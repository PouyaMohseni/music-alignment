"""Detector argmax accuracy at every reported threshold, on any candidate dump.

The dump is sorted by objectness, so row 0 of each frame is the deployed rule.
Column 5 is the time the candidate's page position maps to, t_gt the truth.
"""
import sys, numpy as np
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat

FPS = 22050 / 1102
THR = (0.05, 0.1, 0.5, 1.0, 5.0)

print(f'{"dump":24s} {"n":>5s} ' + ' '.join(f'{"<="+str(t):>7s}' for t in THR))
for path in sys.argv[1:]:
    pages = load_with_feat(path)
    e = np.array([abs(float(c[0, 5]) - float(p['t_gt'][i])) / FPS
                  for p in pages for i, c in enumerate(p['cand']) if len(c)])
    name = path.split('/')[-2]
    print(f'{name:24s} {len(e):5d} ' +
          ' '.join(f'{100*np.mean(e<=t):7.2f}' for t in THR))
