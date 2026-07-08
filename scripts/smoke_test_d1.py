"""D1 smoke test on REAL data (test pieces -- their MERT embeddings already
exist). Exercises: dataset load, forward (similarity matrix), both losses,
backward + gradient flow into BOTH towers, and both decoders (DTW + OLTW)
producing finite monotonic paths. Not fabricated -- actually runs."""
import sys, time
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

import numpy as np
import torch
import yaml

from mymodel.d1_align_matrix.model import D1Model
from mymodel.d1_align_matrix.losses import dense_ce_loss, soft_dtw_matrix_loss
from mymodel.d1_align_matrix import data as d1data
from mymodel.d1_align_matrix.dtw import dtw_decode, oltw_decode

cfg = yaml.safe_load(open('configs/d1_align_matrix.yaml'))
dc = cfg['data']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, flush=True)

# test pieces have MERT in cpjku_fmt_test_eval (already on disk)
pieces = d1data.load_split('test', dc['processed_root'], dc['cpjku_data'],
                           dc['mert_roots'], dc['scale_factor'],
                           cfg['model']['w_downsample'], limit=2)
assert len(pieces) >= 1, 'FAIL: no test pieces loaded'
p = pieces[0]
print(f'piece={p.piece_name}  mert={tuple(p.mert.shape)}  strip={tuple(p.strip.shape)}  '
      f'gt_cols={tuple(p.gt_cols.shape)}  n_onsets={len(p.onset_frames)}', flush=True)

model = D1Model(**cfg['model']).to(device)
print('params:', sum(x.numel() for x in model.parameters()), flush=True)

# --- forward ---
t0 = time.time()
S = model(p.mert.to(device), p.strip.to(device))
print(f'similarity S shape={tuple(S.shape)}  (T x W_col)  [{time.time()-t0:.2f}s]', flush=True)
assert S.shape[0] == p.mert.shape[0], 'FAIL: T mismatch'
W_col = S.shape[1]
gt = p.gt_cols.to(device).clamp(max=W_col - 1)
assert int(gt.max()) < W_col and int(gt.min()) >= 0, 'FAIL: gt col out of range'
print(f'gt_cols range: [{int(gt.min())}, {int(gt.max())}] of W_col={W_col}', flush=True)

# --- losses ---
ce = dense_ce_loss(S, gt, sigma_cols=cfg['loss']['ce_sigma_cols'])
dtw = soft_dtw_matrix_loss(S, gamma=cfg['loss']['dtw_gamma'],
                           max_t=cfg['loss']['dtw_max_t'], max_w=cfg['loss']['dtw_max_w'])
print(f'ce={float(ce):.4f} (init ~ln(W_col)={np.log(W_col):.2f})   soft_dtw={float(dtw):.4f}', flush=True)
assert np.isfinite(float(ce)) and np.isfinite(float(dtw)), 'FAIL: non-finite loss'

loss = ce + cfg['loss']['dtw_weight'] * dtw
loss.backward()

# --- gradient flow into BOTH towers ---
def gsum(module):
    return sum(float(pr.grad.abs().sum()) for pr in module.parameters()
               if pr.grad is not None)
ga = gsum(model.audio_tower); gsc = gsum(model.score_tower)
print(f'grad abs-sum: audio_tower={ga:.4f}  score_tower={gsc:.4f}', flush=True)
assert ga > 0, 'FAIL: no gradient into audio tower'
assert gsc > 0, 'FAIL: no gradient into score tower'

# --- decoders ---
S_np = S.detach().float().cpu().numpy()
t0 = time.time(); path_dtw = dtw_decode(S_np); t_dtw = time.time() - t0
t0 = time.time(); path_oltw = oltw_decode(S_np); t_oltw = time.time() - t0
print(f'DTW  path: len={len(path_dtw)} monotonic={bool(np.all(np.diff(path_dtw)>=0))} '
      f'range=[{path_dtw.min()},{path_dtw.max()}] [{t_dtw:.2f}s]', flush=True)
print(f'OLTW path: len={len(path_oltw)} monotonic={bool(np.all(np.diff(path_oltw)>=0))} '
      f'range=[{path_oltw.min()},{path_oltw.max()}] [{t_oltw:.2f}s]', flush=True)
assert np.all(np.diff(path_dtw) >= 0), 'FAIL: DTW path not monotonic'
assert np.all(np.diff(path_oltw) >= 0), 'FAIL: OLTW path not monotonic'
assert len(path_dtw) == S_np.shape[0], 'FAIL: DTW path length != T'

# --- sanity: DTW path should correlate with GT columns (both monotone rising) ---
corr = np.corrcoef(path_dtw, p.gt_cols.numpy()[:len(path_dtw)])[0, 1]
print(f'corr(DTW path, GT cols) = {corr:.3f}  (untrained model, just a sanity range check)', flush=True)

print('\nSMOKE TEST PASSED', flush=True)
