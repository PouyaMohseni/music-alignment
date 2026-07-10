"""Smoke test for the new particle_filter_decode causal decoder on D1/D2's
similarity matrix, using D2's REAL trained checkpoint (the strong result) on a
real test piece. Compares all three decoders' pct@0.5s on a handful of pieces
-- not a full eval (that's a GPU SLURM job), just a fast sanity check that the
new decoder runs, produces finite/monotonic-ish output, and is at least in the
right ballpark (better than the old greedy oltw_decode would be the hoped-for
outcome, checked for real here rather than assumed)."""
import sys, time
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

import numpy as np
import torch
import yaml

from mymodel.d1_align_matrix.model import D1Model
from mymodel.d1_align_matrix import data as d1data
from mymodel.d1_align_matrix.dtw import dtw_decode, oltw_decode, particle_filter_decode

cfg = yaml.safe_load(open('configs/d2_midi_privileged.yaml'))
dc = cfg['data']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, flush=True)

pieces = d1data.load_split('test', dc['processed_root'], dc['cpjku_data'],
                           dc['mert_roots'], dc['scale_factor'],
                           cfg['model']['w_downsample'], limit=3)
assert len(pieces) >= 1

model = D1Model(**cfg['model']).to(device)
ck = torch.load('results/d2_midi_privileged/best_model.pt', map_location=device)
model.load_state_dict(ck['model'])
model.eval()
print('Loaded D2 checkpoint', flush=True)

FPS = 20

def score_piece(path_cols, piece):
    wd = piece.w_downsample
    diffs = []
    for f in piece.onset_frames:
        if f >= len(path_cols):
            continue
        x_pred = path_cols[f] * wd + wd / 2.0 + piece.add_per_staff
        x_gt = float(piece.gt_cols[f].item()) * wd + wd / 2.0 + piece.add_per_staff
        t_pred = float(piece.interpol_c2o(x_pred))
        t_gt = float(piece.interpol_c2o(x_gt))
        diffs.append(abs(t_pred - t_gt) / FPS)
    return diffs

all_dtw, all_oltw, all_pf = [], [], []
for p in pieces:
    with torch.no_grad():
        S = model(p.mert.to(device), p.strip.to(device))
    S_np = S.float().cpu().numpy()
    print(f'{p.piece_name}: S shape={S_np.shape}', flush=True)

    t0 = time.time()
    path_dtw = dtw_decode(S_np, band_frac=0.05)
    print(f'  dtw_decode: {time.time()-t0:.2f}s, monotonic={np.all(np.diff(path_dtw)>=0)}', flush=True)

    t0 = time.time()
    path_oltw = oltw_decode(S_np)
    print(f'  oltw_decode: {time.time()-t0:.2f}s', flush=True)

    t0 = time.time()
    path_pf = particle_filter_decode(S_np, process_noise_std=1.0, init_std=2.0)
    print(f'  particle_filter_decode: {time.time()-t0:.2f}s, finite={np.all(np.isfinite(path_pf))}', flush=True)
    assert np.all(np.isfinite(path_pf)), 'FAIL: non-finite particle filter path'

    all_dtw += score_piece(path_dtw, p)
    all_oltw += score_piece(path_oltw, p)
    all_pf += score_piece(path_pf, p)

def pct(diffs, th=0.5):
    arr = np.array(diffs)
    return 100.0 * (arr <= th).mean() if len(arr) else float('nan')

print(f'\n3-piece pct@0.5s -- dtw(offline)={pct(all_dtw):.1f}%  '
      f'oltw(greedy)={pct(all_oltw):.1f}%  particle_filter={pct(all_pf):.1f}%', flush=True)
print(f'mean err -- dtw={np.mean(all_dtw):.2f}s  oltw={np.mean(all_oltw):.2f}s  '
      f'pf={np.mean(all_pf):.2f}s', flush=True)

print('\nSMOKE TEST PASSED', flush=True)
