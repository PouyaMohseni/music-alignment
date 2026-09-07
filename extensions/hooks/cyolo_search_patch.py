"""Install a search decoder (beam or banded Viterbi) into cyolo's eval path.

Same two patch points as C2, and deliberately the same coordinate handling:
`get_max_box` returns boxes in the network's scale and the caller then does
`pred_boxes *= scale_factors` before compute_batch_stats unrolls them, so the
decoder must apply the scale itself to decide in the metric's own space.

CLASS 0 ONLY. eval_class calls this same function for the bar (1) and system (2)
readouts; C2 originally let the filter run on all three over one shared
per-piece state, which corrupted them -- bar accuracy fell monotonically in lam
(0.829 / 0.791 / 0.695 / 0.496) while the timing on those very frames stayed
exact. State is also keyed per (piece, class) so enabling another class cannot
silently reintroduce the sharing.
"""
from __future__ import annotations

import os

from extensions.hooks.cyolo_z_capture import LAST_Z as _Z

_BATCH = {'file_names': None, 'add_per_staff': None, 'scale_factors': None,
          'frames': None}
_CLASSES = {int(c) for c in os.environ.get('C2_CLASSES', '0').split(',') if c != ''}
# P-SYS: pin the note to the predicted system box (slack in pixels, 0 = off)
_SYS_SLACK = float(os.environ.get('SYS_SLACK', '0'))
_BAR_SLACK = float(os.environ.get('BAR_SLACK', '0'))


def patch_batch_frames():
    """Carry the spectrogram frame index into the batch.

    SequenceDataset.__getitem__ already emits `t` (the audio sample the frame
    ends at) but CustomBatch.__init__ never copies it, so nothing downstream can
    tell how much time separates two scored steps. Under --only_onsets that gap
    is the whole point: non-onset frames are dropped from the dataset, so
    consecutive steps are one to sixty-four frames apart.
    """
    import cyolo_score_following.dataset as ds_mod
    from cyolo_score_following.utils.data_utils import FRAME_SIZE, HOP_SIZE

    if getattr(ds_mod.CustomBatch, '_frames_patched', False):
        return
    _orig_init = ds_mod.CustomBatch.__init__

    def __init__(self, batch):
        _orig_init(self, batch)
        self.frames = [int((x['t'] - FRAME_SIZE) // HOP_SIZE) for x in batch]

    ds_mod.CustomBatch.__init__ = __init__
    ds_mod.CustomBatch._frames_patched = True
    print('[SEARCH] batch carries frame index', flush=True)


def patch_cyolo_search(kind='beam', **kw):
    import torch

    import cyolo_score_following.dataset as ds_mod
    import cyolo_score_following.utils.general as gen_mod
    from extensions.heads.cyolo_beam_decode import BandedViterbi, BeamDecoder

    patch_batch_frames()

    if kind == 'beam':
        dec = BeamDecoder(**kw)
    elif kind == 'viterbi':
        dec = BandedViterbi(**kw)
    elif kind == 'scorer':
        from extensions.heads.cyolo_beam_decode import ScorerDecoder
        dec = ScorerDecoder(**kw)
    else:
        raise ValueError(f'unknown decoder {kind!r}')

    _WANT_FEAT = getattr(dec, 'model', None) is not None and \
        getattr(dec.model, 'fenc', None) is not None
    if _WANT_FEAT:
        from extensions.hooks.cyolo_feat_capture import gather, patch_capture_feat
        patch_capture_feat(scale=0)
        print('[SEARCH] scorer needs backbone features; capture installed',
              flush=True)

    def _best_of_class(x, cls, sf):
        rows = x[x[:, -1] == cls]
        if rows.shape[0] == 0:
            return None
        b = rows[int(rows[:, 4].argmax())].detach().cpu().numpy()
        return [b[0] * sf, b[1] * sf, b[2] * sf, b[3] * sf, b[4]]

    _orig_get_max_box = gen_mod.get_max_box

    def get_max_box(prediction, class_id=0):
        names = _BATCH['file_names']
        if names is None or class_id not in _CLASSES:
            return _orig_get_max_box(prediction, class_id=class_id)

        aps, sfs = _BATCH['add_per_staff'], _BATCH['scale_factors']
        out = []
        for xi, x in enumerate(prediction):
            m_cls = x[:, -1] == class_id
            sel = x[m_cls]
            if sel.shape[0] == 0:
                out.append(x.new_zeros(4))
                continue
            sf = float(sfs[xi]) if sfs is not None else 1.0
            ncol = sel.shape[1]
            if _WANT_FEAT and class_id == 0:
                # carry each candidate's FLAT index through the filters, so the
                # feature rows cannot drift out of step with the rows the
                # decoder ends up scoring. system_filter/bar_filter return
                # filtered ROWS rather than a mask, and neither reads the last
                # column of sel, so an appended index column rides along.
                fi = torch.nonzero(m_cls).squeeze(-1).to(sel.dtype)
                sel = torch.cat([sel, fi.unsqueeze(1)], 1)
            staff_coords, add_per_staff = (aps[xi] if aps is not None else (None, None))
            if class_id == 0 and _SYS_SLACK > 0:
                from extensions.hooks.cyolo_probe_patch import system_filter
                sel = system_filter(sel, x, _SYS_SLACK)
            if class_id == 0 and _BAR_SLACK > 0:
                from extensions.hooks.cyolo_probe_patch import bar_filter
                sel = bar_filter(sel, x, _BAR_SLACK)
            fv = None
            if _WANT_FEAT and class_id == 0:
                fv = gather(xi, sel[:, -1].long().cpu().numpy())
                sel = sel[:, :ncol]
                if fv is None:
                    raise RuntimeError('feature capture returned nothing; '
                                       'Detect.forward hook did not fire')
            frames = _BATCH['frames']
            # the learned scorer uses the frame's bar and system boxes as
            # features; the hand-tuned decoders ignore them
            chosen = dec.decode(sel[:, :4] * sf, sel[:, 4], f'{names[xi]}::{class_id}',
                                staff_coords=staff_coords, add_per_staff=add_per_staff,
                                frame=(frames[xi] if frames is not None else None),
                                bar=_best_of_class(x, 1, sf),
                                sys=_best_of_class(x, 2, sf),
                                ntot=int(sel.shape[0]),
                                z=(_Z['z'][xi] if _Z.get('z') is not None
                                   and xi < len(_Z['z']) else None),
                                feat=fv)
            out.append(chosen / sf)
        return torch.stack(out)

    gen_mod.get_max_box = get_max_box
    gen_mod._search_patched = True
    if hasattr(ds_mod, 'get_max_box'):
        ds_mod.get_max_box = get_max_box

    _orig_iterate = ds_mod.iterate_dataset

    def iterate_dataset(network, dataloader, criterion, optimizer=None, **kwargs):
        dec.reset()

        class _Wrapped:
            def __init__(self, dl):
                self.dl = dl

            def __len__(self):
                return len(self.dl)

            def __iter__(self):
                for data in self.dl:
                    _BATCH['file_names'] = data.file_names
                    _BATCH['add_per_staff'] = data.add_per_staff
                    _BATCH['scale_factors'] = data.scale_factors
                    _BATCH['frames'] = getattr(data, 'frames', None)
                    yield data

        try:
            return _orig_iterate(network, _Wrapped(dataloader), criterion,
                                 optimizer=optimizer, **kwargs)
        finally:
            _BATCH['file_names'] = None

    ds_mod.iterate_dataset = iterate_dataset
    ds_mod._search_iterate_patched = True
    print(f'[SEARCH] {kind} decoder ACTIVE: {kw}', flush=True)
    return dec
