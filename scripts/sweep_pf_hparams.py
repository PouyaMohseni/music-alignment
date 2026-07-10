"""Quick hyperparameter sweep for particle_filter_decode on 3 real test
pieces using D2's real checkpoint -- cheap CPU-only search before committing
a setting to the full 94-piece GPU eval."""
import sys
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

import numpy as np
import torch
import yaml

from mymodel.d1_align_matrix.model import D1Model
from mymodel.d1_align_matrix import data as d1data
from mymodel.d1_align_matrix.dtw import particle_filter_decode

cfg = yaml.safe_load(open('configs/d2_midi_privileged.yaml'))
dc = cfg['data']
device = torch.device('cpu')

pieces = d1data.load_split('test', dc['processed_root'], dc['cpjku_data'],
                           dc['mert_roots'], dc['scale_factor'],
                           cfg['model']['w_downsample'], limit=3)

model = D1Model(**cfg['model']).to(device)
ck = torch.load('results/d2_midi_privileged/best_model.pt', map_location=device)
model.load_state_dict(ck['model'])
model.eval()

FPS = 20
mats = []
with torch.no_grad():
    for p in pieces:
        S = model(p.mert.to(device), p.strip.to(device)).float().cpu().numpy()
        mats.append(S)

def score(path_cols, piece):
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

settings = [
    (2.0, 2.0), (3.0, 2.0), (4.0, 2.0), (5.0, 2.0), (6.0, 2.0),
    (3.0, 1.0), (4.0, 1.0), (3.0, 3.0), (4.0, 3.0), (8.0, 2.0),
]
for pns, ist in settings:
    all_d = []
    for S, p in zip(mats, pieces):
        path = particle_filter_decode(S, process_noise_std=pns, init_std=ist)
        all_d += score(path, p)
    arr = np.array(all_d)
    print(f'process_noise_std={pns:.1f} init_std={ist:.1f}  '
          f'pct@0.5s={100*(arr<=0.5).mean():.1f}%  mean_err={arr.mean():.2f}s  median_err={np.median(arr):.2f}s',
          flush=True)
