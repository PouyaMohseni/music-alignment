"""B2/B3/B5's auxiliary heads (e.g. network._ext_b2_pitch_head) are attached
as plain attributes on the network during training (so optimizer.add_param_
group sees them -- see iterate_dataset_ext.py), which makes them registered
submodules included in state_dict. That's fine at fresh-start time, but a
weights-only warm-start (--param_path, since CPJKU's train_model.py has no
true resume) calls network.load_state_dict(...) on a FRESHLY constructed
ConditionalUNet, BEFORE any extension's aux head has been lazily created --
so the checkpoint's extension-only keys are "unexpected" to strict loading
and train_model.py's own (unmodified) load call crashes immediately.

CPJKU's own call site (train_model.py) has no strict= kwarg to override, so
this monkey-patches nn.Module.load_state_dict itself to default to
strict=False -- confirmed key error already reproduced running B2/B5 warm-
starts (jobs 64785181, 64785183), both immediate crashes.
"""
import torch.nn as nn

_original_load_state_dict = nn.Module.load_state_dict

# Keys allowed to be absent from a warm-start checkpoint because they are MEANT
# to begin at their initial value. adaLN-Zero (extensions/heads/adaln_zero.py)
# replaces FiLM's gamma/beta Linears with a single zero-initialised `proj`, so a
# FiLM-trained checkpoint cannot contain those keys -- and must not, since the
# method's entire premise is that conditioning starts as an exact no-op. Job
# 780909 died on precisely this, correctly: without an explicit allowance, a
# silently zero-filled tensor is indistinguishable from a broken load.
_ZERO_INIT_OK = ('film_layer.proj.weight', 'film_layer.proj.bias')


def _lenient_load_state_dict(self, state_dict, strict=True):
    missing, unexpected = _original_load_state_dict(self, state_dict, strict=False)
    if unexpected:
        print(f'[lenient_load] Ignoring extension-only checkpoint keys: {unexpected}', flush=True)
    if missing:
        allowed = [k for k in missing if k.endswith(_ZERO_INIT_OK)]
        hard = [k for k in missing if k not in allowed]
        if allowed:
            print(f'[lenient_load] {len(allowed)} zero-init conditioning keys absent '
                  f'from the checkpoint, left at their zero initialisation BY DESIGN '
                  f'(e.g. {allowed[:2]})', flush=True)
        if hard:
            raise RuntimeError(f'Checkpoint is missing base-network keys: {hard}')
    return missing, unexpected


def patch_lenient_load_state_dict():
    nn.Module.load_state_dict = _lenient_load_state_dict
    print('[lenient_load] Patched nn.Module.load_state_dict (strict=False, '
          'still hard-fails on missing base keys)', flush=True)
