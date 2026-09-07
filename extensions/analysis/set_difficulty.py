"""Detector argmax per dump: difficulty of a set, independent of any decoder.

Lives in the repo, not the session scratchpad -- /tmp is local to the login node
and invisible from compute nodes, which is why the first attempt at this failed
in two seconds with "can't open file".
"""
import glob
import sys

import numpy as np

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.offline_decode import TH, load

SETS = [('ROOM (target)', '/scratch/pmohseni/omr/cand_test/room.npz'),
        ('do', '/scratch/pmohseni/omr/cand_test/do.npz'),
        ('rp_synth', '/scratch/pmohseni/omr/cand_test/rp_synth.npz'),
        ('old valid 19pc, room', '/scratch/pmohseni/omr/cand_ir/valid_c0.npz'),
        ('held-out 80pc, no noise', '/scratch/pmohseni/omr/candhv/valid_snr0.npz'),
        ('held-out 80pc, 12 dB', '/scratch/pmohseni/omr/candhv/valid_snr12.npz'),
        ('held-out 80pc, 6 dB', '/scratch/pmohseni/omr/candhv/valid_snr6.npz'),
        ('held-out 80pc, 3 dB', '/scratch/pmohseni/omr/candhv/valid_snr3.npz'),
        ('held-out 80pc, 0.5 dB', '/scratch/pmohseni/omr/candhv/valid_snr0.5.npz')]

print('%-26s %8s %9s %9s' % ('set', 'frames', 'argmax', 'oracle'), flush=True)
for nm, p in SETS:
    if not glob.glob(p):
        print('%-26s  (missing)' % nm, flush=True)
        continue
    pages = load(p)
    n = a = o = 0
    for pg in pages:
        for i, c in enumerate(pg['cand']):
            if c.shape[0] == 0:
                continue
            n += 1
            e = np.abs(c[:, 5] - pg['t_gt'][i])
            a += e[0] <= TH
            o += e.min() <= TH
    print('%-26s %8d %8.1f%% %8.1f%%'
          % (nm, n, 100 * a / max(n, 1), 100 * o / max(n, 1)), flush=True)
print('\n  a validation set is only a proxy for room if its argmax is near 80.0',
      flush=True)
