"""P1 eval -- MERT-aware, with the BUCKETED decode applied at inference.

WHY THIS FILE HAS TO EXIST. eval_model.py reads `model_return['segmentation']`,
i.e. sigmoid(conv_out(x)), and thresholds it at 0.5.  A P1 checkpoint was never
trained to make that map exceed 0.5 anywhere -- its objective only ever
constrained a softmax over x columns.  Scoring a P1 model through the stock
path would therefore threshold to near-nothing and report a catastrophic
number that says nothing about the model.  Train-time and test-time decode
must be the same function.

HOW. We wrap ConditionalUNet.forward so that the returned 'segmentation' IS
the bucketed decode: softmax over x, peak-normalised to 1.0, broadcast over
height.  Everything downstream -- thresholding, centre of mass, staff mapping,
unrolling, interpol_c2o -- is then byte-identical to every other experiment, so
the resulting pct@0.5s sits on exactly the same axis as R2r_realir's 56.6 and
cyolo_sb's 79.9.
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)
# Same explicit package path as the training entry point -- see the comment
# there. This module imports audio_conditioned_unet.network inside
# patch_bucketed_decode, which needs it too.
sys.path.insert(0, os.path.join(REPO_ROOT, 'third_party', 'cpjku_unet'))

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.lenient_load import patch_lenient_load_state_dict

MERT_TEST_EMB_ROOT = os.environ.get('MERT_TEST_EMB_ROOT')
if not MERT_TEST_EMB_ROOT:
    raise RuntimeError('MERT_TEST_EMB_ROOT env var must be set, e.g. '
                       '/scratch/pmohseni/mert_emb_zenodo/msmd_test')

TEST_DIR = os.environ.get('MERT_EVAL_TEST_DIR', '../data/msmd/msmd_test')
POOL = os.environ.get('P1_POOL', 'logsumexp')

patch_mert_pipeline(path_to_emb_root={TEST_DIR: MERT_TEST_EMB_ROOT})
patch_lenient_load_state_dict()


def patch_bucketed_decode(pool='logsumexp'):
    """Make forward() return the bucketed decode as 'segmentation'."""
    import torch
    from audio_conditioned_unet.network import ConditionalUNet
    from extensions.heads.bucketed_softmax import decode_mask

    if getattr(ConditionalUNet, '_p1_decode_patched', False):
        return
    orig_forward = ConditionalUNet.forward
    captured = {}

    def _hook(module, inp, out):
        captured['logits'] = out

    def forward(self, score, perf, hidden):
        if not getattr(self, '_p1_hooked', False):
            self.conv_out.register_forward_hook(_hook)
            self._p1_hooked = True
        ret = orig_forward(self, score, perf, hidden)
        logits = captured.get('logits')
        if logits is None:
            raise RuntimeError('conv_out hook never fired; cannot apply P1 decode')
        h = ret['segmentation'].shape[-2]
        ret['segmentation'] = decode_mask(logits, h, pool=pool)
        return ret

    ConditionalUNet.forward = forward
    ConditionalUNet._p1_decode_patched = True
    print(f'[P1-eval] bucketed decode ACTIVE at inference (pool={pool})', flush=True)


patch_bucketed_decode(pool=POOL)

# Verify rather than assume: an unpatched forward would silently score the P1
# checkpoint through the sigmoid path and report a meaningless collapse.
from audio_conditioned_unet.network import ConditionalUNet as _CU
if not getattr(_CU, '_p1_decode_patched', False):
    raise RuntimeError('P1 decode patch did not take; eval would be invalid')

_EVAL_MODEL_PATH = os.path.join(
    REPO_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet', 'eval_model.py')

runpy.run_path(_EVAL_MODEL_PATH, run_name='__main__')
