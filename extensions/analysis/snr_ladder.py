"""Every rung of the decoder ladder on the held-out pieces at four noise levels.

The noise table on the pages compares the shipped scorer with the featureless
one. The paper's robustness figure needs the whole ladder -- the detector's own
argmax, the zero-parameter prior, the featureless scorer and the full scorer --
on the same 80 held-out pieces, so a reader can see how much of the gain at
each noise level comes from the prior alone.
"""
from __future__ import annotations

import gc
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.heldout_compare import rollout_hits
from extensions.analysis.ladder_ci import rollout_argmax, rollout_hand
from extensions.heads.cand_scorer import load as load_ckpt

M = '/scratch/pmohseni/omr/scorer'


def main():
    flat = load_ckpt(f'{M}/ir_only.pt')[0]
    full = load_ckpt(f'{M}/nbr/nbrp64_s0.pt')[0]
    print(f'{"snr":>5s} {"onsets":>7s} {"argmax":>7s} {"prior":>7s} {"flat":>7s} {"full":>7s}')
    for snr in ('12', '6', '3', '0.5'):
        pages = load_with_feat(f'/scratch/pmohseni/omr/candhv256/valid_snr{snr}.npz')
        a, _ = rollout_argmax(pages)
        h, _ = rollout_hand(pages)
        f, _ = rollout_hits(flat, pages)
        s, _ = rollout_hits(full, pages)
        print(f'{snr:>5s} {len(a):7d} {100 * a.mean():7.2f} {100 * h.mean():7.2f} '
              f'{100 * f.mean():7.2f} {100 * s.mean():7.2f}', flush=True)
        del pages
        gc.collect()


if __name__ == '__main__':
    main()
