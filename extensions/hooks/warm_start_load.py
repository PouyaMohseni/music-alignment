"""Warm-start loader for architectures that ADD new modules to CB_TA.

extensions/hooks/lenient_load.py tolerates *unexpected* checkpoint keys
(extension heads saved by a previous run) but hard-fails on any *missing*
key -- which is exactly right for B2/B3/B5, whose aux heads are created
lazily AFTER load_state_dict, so at load time the network has no extra
parameters and nothing is ever missing.

The N1/N2/N3 architectures (long-context temporal core, gated memory
retrieval, gated belief propagation) instead declare their new modules in
__init__, because they participate in the forward pass on every step and
must be in the optimizer from step one. Warm-starting those from B1a's
converged checkpoint therefore reports the new modules as "missing", and
lenient_load would abort.

Blanket strict=False would also silently swallow a genuinely missing BASE
key (e.g. a renamed encoder layer loading nothing and training from noise) --
the exact failure lenient_load exists to catch. So this keeps that check and
only whitelists prefixes the caller explicitly declares as new:

    patch_warm_start_load(allow_missing_prefixes=('temporal_core.', 'mem_read.'))

Missing keys under a declared prefix are reported and allowed; any other
missing key still raises.
"""
import torch.nn as nn

_original_load_state_dict = nn.Module.load_state_dict


def patch_warm_start_load(allow_missing_prefixes=()):
    allow = tuple(allow_missing_prefixes)

    def _warm_start_load_state_dict(self, state_dict, strict=True):
        missing, unexpected = _original_load_state_dict(self, state_dict, strict=False)
        if unexpected:
            print(f'[warm_start_load] Ignoring extension-only checkpoint keys: '
                  f'{sorted(unexpected)}', flush=True)
        allowed = [k for k in missing if k.startswith(allow)]
        blocked = [k for k in missing if not k.startswith(allow)]
        if allowed:
            print(f'[warm_start_load] {len(allowed)} new-module keys not in checkpoint '
                  f'(expected -- training these from scratch on top of the warm-started '
                  f'base): prefixes={allow}', flush=True)
        if blocked:
            raise RuntimeError(
                f'Checkpoint is missing base-network keys (not covered by '
                f'allow_missing_prefixes={allow}): {sorted(blocked)}')
        return missing, unexpected

    nn.Module.load_state_dict = _warm_start_load_state_dict
    print(f'[warm_start_load] Patched nn.Module.load_state_dict '
          f'(strict=False; missing keys allowed only under {allow})', flush=True)
