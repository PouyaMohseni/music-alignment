"""A4 eval -- MERT-aware, with the BOUNDARY decode applied at inference.

WHY THIS FILE HAS TO EXIST. A4's first evaluation used run_eval_native_mert.py,
which runs stock eval_model.py. That scores `model_return['segmentation']` =
sigmoid(conv_out(...)) and DROPS the boundary/staff tensors as unexpected keys
(lenient_load prints, does not raise). But A4 trains with A4_DICE=0, so the
dense conv_out head receives no gradient at all: the number that came back
(31.7 on room) measured how far a frozen readout drifted while the upstream
half was fine-tuned for a different objective. It said nothing about boundary
prediction.

Train-time and test-time decode must be the same function. Here the heads are
constructed so their weights load, and forward() is wrapped so 'segmentation'
IS the boundary decode at the staff row the coarse head selects. Everything
downstream -- thresholding, centre of mass, staff assignment, unrolling,
interpol_c2o -- stays byte-identical to every other experiment.
"""
import os, runpy, sys

_THIS = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'third_party', 'cpjku_unet'))

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

EMB = os.environ.get('MERT_TEST_EMB_ROOT')
if not EMB:
    raise RuntimeError('MERT_TEST_EMB_ROOT must be set')
TEST_DIR = os.environ.get('MERT_EVAL_TEST_DIR', '../data/msmd/msmd_test')
STAGE = int(os.environ.get('A4_STAGE', '6'))
BINS = int(os.environ.get('A4_BINS', '16'))

patch_mert_pipeline(path_to_emb_root={TEST_DIR: EMB})
patch_lenient_load_state_dict()


def patch_boundary_decode(decoder_stage=STAGE, n_bins=BINS):
    import torch
    from audio_conditioned_unet.network import ConditionalUNet
    from extensions.heads.boundary_head import BoundaryHead, decode_mask
    from extensions.heads.staff_coarse_head import StaffCoarseHead
    from extensions.hooks.film_feature_extractor import (FeatureCapture,
                                                         decoder_index_for_stage)
    from extensions.hooks.boundary_patch import _staff_rows_from_logits

    if getattr(ConditionalUNet, '_a4_decode_patched', False):
        return
    orig_init = ConditionalUNet.__init__
    orig_forward = ConditionalUNet.forward

    def __init__(self, config, *a, **kw):
        orig_init(self, config, *a, **kw)
        # Build the heads at construction so load_state_dict finds their keys.
        # The channel count must match training: decoder[idx]'s out_channels.
        idx = decoder_index_for_stage(self.n_encoder_layers, decoder_stage)
        ch = self.decoder[idx].conv_block[0].out_channels \
            if hasattr(self.decoder[idx], 'conv_block') else None
        if ch is None:      # fall back to a probe-free attribute walk
            for m in self.decoder[idx].modules():
                if isinstance(m, torch.nn.Conv2d):
                    ch = m.out_channels
        self.add_module('_ext_bnd_head', BoundaryHead(in_ch=ch))
        self.add_module('_ext_staff_head', StaffCoarseHead(in_ch=ch, n_bins=n_bins))
        self._a4_capture = FeatureCapture(self, idx)

    def forward(self, score, perf, hidden):
        ret = orig_forward(self, score, perf, hidden)
        feat = self._a4_capture.feature
        if feat is None:
            raise RuntimeError('A4 decoder feature hook never fired')
        a, dl, dr = self._ext_bnd_head(feat)
        s = self._ext_staff_head(feat)
        H, W = ret['segmentation'].shape[-2], ret['segmentation'].shape[-1]
        rows = _staff_rows_from_logits(s, H)
        dec = decode_mask(a, dl, dr, H, staff_row=rows)
        if dec.shape[-1] != W:
            dec = torch.nn.functional.interpolate(dec, size=(H, W), mode='nearest')
        ret['segmentation'] = dec
        return ret

    ConditionalUNet.__init__ = __init__
    ConditionalUNet.forward = forward
    ConditionalUNet._a4_decode_patched = True
    print(f'[A4-eval] boundary decode ACTIVE (stage={decoder_stage}, bins={n_bins})',
          flush=True)


patch_boundary_decode()
from audio_conditioned_unet.network import ConditionalUNet as _CU
if not getattr(_CU, '_a4_decode_patched', False):
    raise RuntimeError('A4 decode patch did not take; eval would be invalid')

runpy.run_path(os.path.join(REPO_ROOT, 'third_party', 'cpjku_unet',
                            'audio_conditioned_unet', 'eval_model.py'),
               run_name='__main__')
