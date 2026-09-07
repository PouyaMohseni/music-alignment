import sys
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
import numpy as np
from extensions.analysis.index_decode import score
from extensions.analysis.note_index import build_bins, to_index
from extensions.analysis.offline_decode import load

SETS = {'room': '/scratch/pmohseni/omr/cand_test/room.npz',
        'do': '/scratch/pmohseni/omr/cand_test/do.npz',
        'rp_synth': '/scratch/pmohseni/omr/cand_test/rp_synth.npz'}
pages = load(SETS['room'])

print('GATE: pixel mode must reproduce the shipped decoder', flush=True)
for mu, want in ((0.0, 85.9), (1.0, 86.5)):
    got, n = score(pages, use_index=False, fwd=6.0, sigma=18.0, mu_pow=mu, topk=100000)
    ok = abs(got - want) < 0.15
    print(f'  mu_pow={mu}: {got:.2f} vs {want}  {"OK" if ok else "MISMATCH"}', flush=True)
    if not ok:
        raise SystemExit('pixel mode does not reproduce; nothing below is valid')

# how large is a true step, in the index we just built?
b = build_bins(pages, bin_px=12.0)
steps = []
for p in pages:
    ix = to_index(p['cand'][0][:1, 0] * 0, b[p['name']])
    prev = None
    for i, c in enumerate(p['cand']):
        if c.shape[0] == 0:
            continue
        j = int(np.argmin(np.abs(c[:, 5] - p['t_gt'][i])))
        v = float(to_index(c[j:j + 1, 0], b[p['name']])[0])
        if prev is not None and v > prev:
            steps.append(v - prev)
        prev = v
steps = np.array(steps)
print(f'\ntrue step in note index: median {np.median(steps):.2f}  '
      f'mean {steps.mean():.2f}  p90 {np.percentile(steps, 90):.2f}', flush=True)

print('\nINDEX-SPACE sweep on room  (pixel reference: 86.5)', flush=True)
print('%6s %7s %7s %8s' % ('bin_px', 'fwd', 'sigma', 'room'), flush=True)
best = (0, None)
for bp in (8.0, 12.0, 20.0):
    for fwd in (0.5, 1.0, 2.0):
        for sg in (1.5, 3.0, 6.0):
            v, _ = score(pages, use_index=True, bin_px=bp, fwd=fwd, sigma=sg,
                         topk=100000)
            print('%6.0f %7.1f %7.1f %8.1f' % (bp, fwd, sg, v), flush=True)
            if v > best[0]:
                best = (v, (bp, fwd, sg))
print(f'\nbest on room: {best[0]:.1f} at bin_px/fwd/sigma = {best[1]}', flush=True)

if best[1]:
    bp, fwd, sg = best[1]
    print('\nsame setting on the other two tiers:', flush=True)
    for nm in ('do', 'rp_synth'):
        pg = load(SETS[nm])
        vi, _ = score(pg, use_index=True, bin_px=bp, fwd=fwd, sigma=sg, topk=100000)
        vp, _ = score(pg, use_index=False, fwd=6.0, sigma=18.0, topk=100000)
        print(f'  {nm:<10} index {vi:5.1f}   pixel {vp:5.1f}   {vi-vp:+.1f}', flush=True)
