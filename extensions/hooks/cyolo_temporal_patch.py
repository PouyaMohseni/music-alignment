"""C2 -- install the causal temporal decode into CYOLO's evaluation path.

Two patches, no third-party file edited:

  1. `iterate_dataset` stashes the current batch's file names, staff geometry
     and scale factors on a module-level holder.
  2. `get_max_box` consults the temporal filter instead of taking a bare
     per-frame argmax over objectness.

NO TRAINING. This changes only how a position is read out of the network's
existing predictions, so it runs against the RELEASED cyolo_sb checkpoint
(79.9) and costs one CPU evaluation.

COORDINATE SPACE, which is the thing most likely to go wrong here.
`get_max_box` returns boxes in the network's own scale; the caller then does
`pred_boxes *= scale_factors` before compute_batch_stats unrolls them against
`add_per_staff`. So to make a transition decision in the SAME space the metric
uses, the filter must apply the scale factor itself before unrolling. Getting
this wrong would not crash -- it would just make the transition prior compare
pixels in two different units, which is exactly the class of silent error that
has cost this project several runs.
"""
from __future__ import annotations

import os

_BATCH = {'file_names': None, 'add_per_staff': None, 'scale_factors': None}

# Which detection classes the temporal filter is allowed to decode. Class 0 is
# the tracked position -- the only thing this method is about. Classes 1 and 2
# are the bar/system readouts eval_class scores separately.
_CLASSES = {int(c) for c in os.environ.get('C2_CLASSES', '0').split(',') if c != ''}


def patch_cyolo_temporal(lam: float = 1.0, fwd_px: float = 6.0, sigma_px: float = 18.0,
                         jump_logp: float = -6.0, topk: int = 32, warmup: int = 3):
    import torch

    import cyolo_score_following.dataset as ds_mod
    import cyolo_score_following.utils.general as gen_mod
    from extensions.heads.cyolo_temporal_decode import TemporalDecoder

    dec = TemporalDecoder(lam=lam, fwd_px=fwd_px, sigma_px=sigma_px,
                          jump_logp=jump_logp, topk=topk, warmup=warmup)

    _orig_get_max_box = gen_mod.get_max_box

    def get_max_box(prediction, class_id=0):
        names = _BATCH['file_names']
        if names is None:
            return _orig_get_max_box(prediction, class_id=class_id)

        # ONLY the tracked position (class 0). eval_class calls this same
        # function for class 1 (bar) and class 2 (system) to score its auxiliary
        # readouts, so the unfixed patch ran the temporal filter three times per
        # frame over ONE shared per-piece state: the note tracker's "previous
        # position" was really the system box from the previous call, a bar box
        # got a prior tuned for note motion, and the warmup counter advanced 3x
        # too fast. That is why bar accuracy fell monotonically in lam
        # (0.829 -> 0.791 -> 0.695 -> 0.496) while timing on those same frames
        # stayed exact -- the boxes were right and the read-out was corrupted.
        if class_id not in _CLASSES:
            return _orig_get_max_box(prediction, class_id=class_id)

        aps = _BATCH['add_per_staff']
        sfs = _BATCH['scale_factors']
        out = []
        for xi, x in enumerate(prediction):
            sel = x[x[:, -1] == class_id]
            if sel.shape[0] == 0:
                out.append(x.new_zeros(4))
                continue
            sf = float(sfs[xi]) if sfs is not None else 1.0
            staff_coords, add_per_staff = (aps[xi] if aps is not None else (None, None))
            # decide in the METRIC's coordinate space: boxes here are pre-scale,
            # and compute_batch_stats unrolls post-scale.
            boxes = sel[:, :4]
            # key state per (piece, class) even though only class 0 reaches here,
            # so enabling another class cannot silently reintroduce the sharing
            chosen = dec.decode(boxes * sf, sel[:, 4], f'{names[xi]}::{class_id}',
                                staff_coords=staff_coords, add_per_staff=add_per_staff)
            out.append(chosen / sf)
        return torch.stack(out)

    gen_mod.get_max_box = get_max_box
    gen_mod._c2_patched = True
    # dataset.py imported the symbol directly, so rebind it there too
    if hasattr(ds_mod, 'get_max_box'):
        ds_mod.get_max_box = get_max_box

    _orig_iterate = ds_mod.iterate_dataset

    def iterate_dataset(network, dataloader, criterion, optimizer=None, **kw):
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
                    yield data

        try:
            return _orig_iterate(network, _Wrapped(dataloader), criterion,
                                 optimizer=optimizer, **kw)
        finally:
            _BATCH['file_names'] = None

    ds_mod.iterate_dataset = iterate_dataset
    ds_mod._c2_iterate_patched = True
    print(f'[C2] causal temporal decode ACTIVE (lam={lam}, fwd={fwd_px}px, '
          f'sigma={sigma_px}px, jump_logp={jump_logp}, topk={topk})', flush=True)
    return dec
