"""The encoder ablation: audio encoder x image encoder, our decoder on each.

                  image features the scorer reads
                  cyolo backbone (FiLM,        DINOv2-base
                  audio-conditioned)           (visual only, frozen)
  spectrogram     shipped nbrp64 config        dino
  MERT            h1                           h1dino

Every cell uses the shipped decoder recipe retrained on that cell's own
candidates. Numbers are room pct@0.5s from the offline rollout, which
reproduces the harness exactly for cyolo_sb (93.42 / 92.91 / 94.89), reported
as the mean over seeds with every seed listed. Each detector also gets its own
argmax (what that detector scores alone) and the zero-parameter hand decoder,
so the decoder's gain can be read per detector.
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.heldout_compare import rollout_hits
from extensions.analysis.ladder_ci import rollout_argmax, rollout_hand
from extensions.heads.cand_scorer import load as load_ckpt

O = '/scratch/pmohseni/omr'
M = f'{O}/scorer'
CELLS = [
    ('spectrogram', 'cyolo backbone', f'{O}/candf256/room.npz', f'{M}/nbr/nbrp64_s[0-9].pt'),
    ('MERT', 'cyolo backbone', f'{O}/candh1/room.npz', f'{M}/abl/h1_s[0-9].pt'),
    ('spectrogram', 'DINOv2', f'{O}/candf_dino/room.npz', f'{M}/abl/dino_s[0-9].pt'),
    ('MERT', 'DINOv2', f'{O}/candh1_dino/room.npz', f'{M}/abl/h1dino_s[0-9].pt'),
]


def main():
    detector, rows = {}, []
    for audio, image, dump, pat in CELLS:
        if not glob.glob(dump):
            print(f'{audio} / {image}: dump missing ({dump})')
            continue
        pages = load_with_feat(dump)
        if audio not in detector:
            a, _ = rollout_argmax(pages)
            h, _ = rollout_hand(pages)
            detector[audio] = (len(a), 100.0 * a.mean(), 100.0 * h.mean())
        accs = []
        for p in sorted(glob.glob(pat)):
            hits, _ = rollout_hits(load_ckpt(p)[0], pages)
            accs.append(100.0 * hits.mean())
            print(f'  {audio:11s} {image:15s} {p.split("/")[-1]:16s} {accs[-1]:6.2f}',
                  flush=True)
        rows.append((audio, image, accs))

    print(f'\n{"detector audio":15s} {"onsets":>7s} {"argmax":>7s} {"hand":>7s}')
    for audio, (n, a, h) in detector.items():
        print(f'{audio:15s} {n:7d} {a:7.2f} {h:7.2f}')

    print(f'\n{"audio":12s} {"image features":15s} {"seeds":>5s} {"mean":>7s} {"sd":>5s} '
          f'{"min":>6s} {"max":>6s} {"s0-2":>6s} {"vs argmax":>9s}')
    for audio, image, accs in rows:
        if not accs:
            print(f'{audio:12s} {image:15s}  no checkpoints')
            continue
        v = np.array(accs)
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        print(f'{audio:12s} {image:15s} {len(v):5d} {v.mean():7.2f} {sd:5.2f} '
              f'{v.min():6.2f} {v.max():6.2f} {v[:3].mean():6.2f} '
              f'{v.mean() - detector[audio][1]:+9.2f}')


if __name__ == '__main__':
    main()
