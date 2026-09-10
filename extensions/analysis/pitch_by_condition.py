"""Do the pitch heads survive each acoustic condition, or only the one they saw?

The pitch scorer is the best configuration on room (93.54 +/- 0.60, the tightest
seed spread we have) and BELOW the featureless baseline on held-out (-0.52).
That inversion needs an explanation before the direction can be trusted either
way, and there is a candidate: the heads were fitted on IR-augmented training
audio, room is a real room microphone, and held-out is ADDITIVE-NOISE degraded.
If the audio head is the fragile part, it should hold up on room and fall over
on noise -- which is exactly the shape of the result.

Measured per condition, so the claim is decomposed rather than asserted:

  audio head  can it name a sounding pitch from z alone
  score head  can it name the pitch of the notehead a box sits on, from image
              features. These are FiLM-conditioned by audio, so they are not
              automatically condition-independent either
  agreement   the ratio that actually feeds the scorer, correct vs wrong
              candidates

valid is IR-augmented and matches training, so it is the control: if the heads
are weak there too, the problem is the heads and not the transfer.
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.offline_decode import TH
from extensions.analysis.pitch_heads import FPS, LO, NP_, load_heads, piece_notes

CONDS = [
    ('valid  (IR-augmented, = training)', '/scratch/pmohseni/omr/candf/valid.npz'),
    ('room   (real room microphone)', '/scratch/pmohseni/omr/candf/room.npz'),
    ('heldout(additive noise, 12 dB)', '/scratch/pmohseni/omr/candhv/valid_snr12.npz'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heads', default='/scratch/pmohseni/omr/scorer/pitch/heads_s0.pt')
    ap.add_argument('--tol', type=float, default=0.05)
    a = ap.parse_args()
    torch.set_num_threads(4)
    sh, ah = load_heads(a.heads)

    print(f'{"condition":34s} {"audio top1":>10s} {"audio AP":>9s} '
          f'{"score top1":>10s} {"agree ratio":>12s}')
    for name, dump in CONDS:
        if not glob.glob(dump):
            print(f'{name:34s}  (missing)'); continue
        pages = load_with_feat(dump)
        a_hit = a_n = s_hit = s_n = 0
        ap_num, ap_den = 0.0, 0
        tr = tw = nn = 0.0
        for p in pages:
            nt = piece_notes(p['name'])
            if nt is None or p.get('feat') is None or p.get('z') is None:
                continue
            for i, c in enumerate(p['cand']):
                if c.shape[0] == 0:
                    continue
                t_now = float(p['t_gt'][i]) / FPS
                snd = nt[np.abs(nt[:, 0] - t_now) <= a.tol, 1].astype(int)
                fv = np.asarray(p['feat'][i], np.float32)
                k = min(64, c.shape[0], fv.shape[0])
                if k == 0:
                    continue
                with torch.no_grad():
                    ps = torch.softmax(sh(torch.from_numpy(fv[:k])), -1).numpy()
                    pa = torch.sigmoid(ah(torch.from_numpy(
                        np.asarray(p['z'][i], np.float32)[None]))).numpy()[0]
                if len(snd):
                    tgt = set(np.clip(snd - LO, 0, NP_ - 1))
                    a_hit += int(int(pa.argmax()) in tgt); a_n += 1
                    # average precision of the sounding set under the head's ranking
                    order = np.argsort(-pa)
                    hits = np.array([1.0 if j in tgt else 0.0 for j in order])
                    if hits.sum():
                        cum = np.cumsum(hits) / (np.arange(len(hits)) + 1)
                        ap_num += float((cum * hits).sum() / hits.sum()); ap_den += 1
                    # score head, against the notehead each box sits on
                    ct = c[:k, 5] / FPS
                    near = nt[np.argmin(np.abs(nt[:, 0][None, :] - ct[:, None]), 1),
                              1].astype(int)
                    s_hit += int((ps.argmax(1) == np.clip(near - LO, 0, NP_ - 1)).sum())
                    s_n += k
                    agree = ps @ pa
                    corr = np.abs(c[:k, 5] - p['t_gt'][i]) <= TH
                    if corr.any() and (~corr).any():
                        tr += agree[corr].mean(); tw += agree[~corr].mean(); nn += 1
        print(f'{name:34s} {100.0 * a_hit / max(a_n, 1):9.1f}% '
              f'{ap_num / max(ap_den, 1):9.3f} '
              f'{100.0 * s_hit / max(s_n, 1):9.1f}% '
              f'{tr / max(tw, 1e-9):11.2f}x', flush=True)
    print('\naudio top1 = the head\'s highest-scoring pitch is actually sounding')
    print('score top1 = the head names the exact pitch of the box\'s notehead')
    print('agree ratio = correct-vs-wrong separation, the quantity the scorer sees')


if __name__ == '__main__':
    main()
