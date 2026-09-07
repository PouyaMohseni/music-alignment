"""Does the note-index coordinate stack on the 91.4 selector, or is it redundant?

The index prior is worth +0.7 on room over the pixel prior on the HAND decoder
(87.2 vs 86.5) and +1.0 on `do`. But the shipped model is the hand prior blended
0.3 with a learned selector at 91.4, and the selector already sees per-candidate
displacement -- so the index may be information it has effectively learned.

Two ways the index can enter, tested separately:
  prior   -- the blended hand term decodes in index space
  feature -- the candidate's index displacement is handed to the selector too
             (not tested here; needs a retrain, and this says whether it is worth it)
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
torch.set_num_threads(1)
from extensions.analysis.blend_rollout import load_with_feat
from extensions.analysis.note_index import build_bins, to_index
from extensions.analysis.offline_decode import TH, prior_logp
from extensions.heads.cand_features import build
from extensions.heads.cand_scorer import load as load_ckpt

SETS = {'room': '/scratch/pmohseni/omr/cand_test/room.npz',
        'do': '/scratch/pmohseni/omr/cand_test/do.npz'}


def rollout(model, pages, blend=0.7, use_index=False, bin_px=8.0, fwd=0.5,
            sigma=1.5, fwd_px=6.0, sigma_px=18.0, jump=-6.0, lam=1.0, ref=5.0,
            mu_pow=1.0, topk=256):
    bins = build_bins(pages, bin_px=bin_px) if use_index else None
    hit = tot = 0
    for p in pages:
        ctr = bins[p['name']] if use_index else None
        xp = yp = xp2 = fp = fp2 = None
        ip = None
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            cs = c[:topk]
            tot += 1
            fr = int(p['frame'][i])
            dfr = fr - fp if fp is not None and fr > fp else None
            dfp = fp - fp2 if fp is not None and fp2 is not None and fp > fp2 else None
            f = build(cs, p['bar'][i], p['sys'][i], xp, yp, dfr,
                      ntot=int(p['ntot'][i]), use_abs_obj=model.use_abs_obj,
                      x_prev2=xp2, dframes_prev=dfp)[:, :model.nf]
            with torch.no_grad():
                s = model(torch.from_numpy(f).unsqueeze(0))[0].numpy()
            if blend < 1.0:
                lo = np.log(np.clip(cs[:, 4], 1e-8, None))
                if xp is None:
                    hand = lo
                elif use_index:
                    pos = to_index(cs[:, 0], ctr)
                    k = np.clip((dfr or ref) / ref, 0.2, 8.0) ** mu_pow
                    hand = lo + lam * prior_logp(pos - ip, fwd * k, sigma, jump)
                else:
                    k = np.clip((dfr or ref) / ref, 0.2, 8.0) ** mu_pow
                    hand = lo + lam * prior_logp(cs[:, 0] - xp, fwd_px * k,
                                                 sigma_px, jump)
                s = blend * s + (1.0 - blend) * hand
            j = int(np.argmax(s))
            hit += abs(float(cs[j, 5]) - float(p['t_gt'][i])) <= TH
            xp2, fp2 = xp, fp
            xp, yp, fp = float(cs[j, 0]), float(cs[j, 1]), fr
            if use_index:
                ip = float(to_index(cs[j:j + 1, 0], ctr)[0])
    return 100.0 * hit / max(tot, 1), tot


m, _ = load_ckpt('/scratch/pmohseni/omr/scorer/ir_only.pt')
pages = load_with_feat(SETS['room'])
print('GATE on room:', flush=True)
for b, want in ((0.0, 86.5), (0.7, 91.4)):
    got, _ = rollout(m, pages, blend=b)
    ok = abs(got - want) < 0.15
    print(f'  blend {b} pixel: {got:.2f} vs {want}  {"OK" if ok else "MISMATCH"}',
          flush=True)
    if not ok:
        raise SystemExit('gate failed')

print('\n%-10s %8s %10s %10s %8s' % ('tier', 'blend', 'pixel', 'index', 'delta'),
      flush=True)
for nm, path in SETS.items():
    pg = load_with_feat(path) if nm != 'room' else pages
    for b in (0.0, 0.3, 0.5, 0.7):
        vp, _ = rollout(m, pg, blend=b, use_index=False)
        vi, _ = rollout(m, pg, blend=b, use_index=True)
        print('%-10s %8.1f %10.1f %10.1f %+8.1f' % (nm, b, vp, vi, vi - vp),
              flush=True)
