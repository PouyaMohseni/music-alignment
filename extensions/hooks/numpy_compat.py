"""Restore the numpy aliases cyolo's data loader still uses.

`cyolo_score_following/utils/data_utils.py` calls np.int (line 52) and np.float
(lines 170-171). Those aliases were removed in numpy 1.24, so the venv this
project was built against must have been on 1.23.x, and any numpy upgrade breaks
dataset loading with

    AttributeError: module 'numpy' has no attribute 'float'

Restoring them is exact rather than approximate: they were never distinct types,
only spellings of the builtins, so np.float = float reproduces the old behaviour
byte for byte. That is why this is preferable to pinning an old numpy -- it
removes the fragility instead of freezing the environment around it, and it
leaves the numbers unchanged, which the control arms verify.
"""
from __future__ import annotations

import numpy as np

# EXACTLY what cyolo uses, nothing more: restoring np.bool / np.object / np.str
# is unnecessary here and numpy warns that it intends to reuse those names for
# real scalar types, so claiming them would be borrowing trouble.
_ALIASES = dict(float=float, int=int)


def patch():
    restored = []
    for name, builtin in _ALIASES.items():
        if not hasattr(np, name):
            setattr(np, name, builtin)
            restored.append(name)
    if restored:
        print(f'[NUMPY] restored removed aliases on numpy {np.__version__}: '
              f'{", ".join("np." + r for r in restored)}', flush=True)
