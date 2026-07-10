"""D3 smoke test on REAL data + REAL v13 checkpoint. Verifies: warm-start
loads v13's exact perf_encoder weights with no missing/unexpected keys,
forward pass produces finite/correctly-shaped embeddings, full D2 loss stack
(repeat-aware CE + soft-DTW + MIDI distillation) backprops into BOTH towers,
and DTW decode is still monotonic. Also compares the WARM-STARTED (untrained
otherwise) D3 model's similarity-matrix quality against D1's original small
audio tower at the same (untrained score-tower) starting point, via
corr(DTW path, GT cols) -- same sanity metric D1's own first smoke test used
-- as an early signal of whether the swap is promising before any GPU training."""
import sys
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

import numpy as np
import torch
import yaml

from mymodel.d3_hybrid.model import D3Model
from mymodel.d1_align_matrix.model import D1Model
from mymodel.d1_align_matrix.losses import dense_ce_loss, soft_dtw_matrix_loss
from mymodel.d1_align_matrix.dtw import dtw_decode
from mymodel.d2_midi_privileged.midi_encoder import MidiEncoder
from mymodel.d2_midi_privileged.losses import soft_multi_target_ce_loss, midi_distill_loss
from mymodel.d2_midi_privileged import data as d2data

cfg = yaml.safe_load(open('configs/d3_hybrid.yaml'))
dc = cfg['data']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, flush=True)

pieces = d2data.load_split('test', dc['processed_root'], dc['cpjku_data'],
                           dc['mert_roots'], dc['scale_factor'],
                           cfg['model']['w_downsample'], repeat_k=dc['repeat_k'], limit=2)
assert len(pieces) >= 1
p = pieces[0]
print(f'piece={p.piece_name}  mert={tuple(p.mert.shape)}  strip={tuple(p.strip.shape)}', flush=True)

# --- warm-start ---
model_kwargs = {k: v for k, v in cfg['model'].items() if k != 'warm_start_ckpt'}
model = D3Model(**model_kwargs).to(device)
missing, unexpected = model.audio_tower.load_pretrained_v13(cfg['model']['warm_start_ckpt'], device=device)
print(f'warm-start: missing={missing}  unexpected={unexpected}', flush=True)
assert missing == [] and unexpected == [], 'FAIL: warm-start key mismatch'
print('Confirmed exact key match, no missing/unexpected keys.', flush=True)

# --- forward ---
A, B = model.encode(p.mert.to(device), p.strip.to(device))
S = model.similarity(A, B)
print(f'A shape={tuple(A.shape)}  B shape={tuple(B.shape)}  S shape={tuple(S.shape)}', flush=True)
assert torch.isfinite(A).all() and torch.isfinite(B).all() and torch.isfinite(S).all(), 'FAIL: non-finite forward'
W_col = S.shape[1]
gt_cols = p.gt_cols.to(device).clamp(max=W_col - 1)

# --- early sanity: warm-started D3 vs D1's original tower, both untrained score towers ---
path_d3 = dtw_decode(S.detach().float().cpu().numpy(), band_frac=0.05)
corr_d3 = np.corrcoef(path_d3, p.gt_cols.numpy()[:len(path_d3)])[0, 1]
print(f'corr(DTW path, GT cols) -- D3 (v13 warm-start, untrained score tower): {corr_d3:.3f}', flush=True)

torch.manual_seed(0)
d1_model_kwargs = {k: v for k, v in model_kwargs.items() if k != 'spec_enc'}
d1 = D1Model(**d1_model_kwargs).to(device)
with torch.no_grad():
    S_d1 = d1(p.mert.to(device), p.strip.to(device))
path_d1 = dtw_decode(S_d1.detach().float().cpu().numpy(), band_frac=0.05)
corr_d1 = np.corrcoef(path_d1, p.gt_cols.numpy()[:len(path_d1)])[0, 1]
print(f'corr(DTW path, GT cols) -- D1 original tower (untrained, fresh seed):   {corr_d1:.3f}', flush=True)
print('(both score towers untrained here -- this isolates the audio tower only; '
      'not a claim about final trained quality, just an early direction check)', flush=True)

# --- full D2 loss stack + gradient flow into BOTH towers ---
midi_encoder = MidiEncoder(d_model=cfg['model']['d_model']).to(device)
ce = soft_multi_target_ce_loss(S, gt_cols, repeat_alt_cols=p.repeat_alt_cols,
                               sigma_cols=cfg['loss']['ce_sigma_cols'],
                               alt_weight=cfg['loss']['repeat_alt_weight'])
dtw = soft_dtw_matrix_loss(S, gamma=cfg['loss']['dtw_gamma'],
                           max_t=cfg['loss']['dtw_max_t'], max_w=cfg['loss']['dtw_max_w'])
T = A.shape[0]
sub = min(T, 300)
idx = torch.from_numpy(np.random.default_rng(0).choice(T, size=sub, replace=False)).to(device)
M = midi_encoder(p.pitch_roll.to(device)[idx])
distill = midi_distill_loss(A[idx], M, temperature=cfg['loss']['distill_temperature'])
print(f'ce={float(ce):.4f}  dtw={float(dtw):.4f}  distill={float(distill):.4f}', flush=True)
assert all(np.isfinite(float(x)) for x in (ce, dtw, distill))

loss = ce + cfg['loss']['dtw_weight'] * dtw + cfg['loss']['distill_weight'] * distill
loss.backward()

def gsum(module):
    return sum(float(pr.grad.abs().sum()) for pr in module.parameters() if pr.grad is not None)
ga, gsc, gm = gsum(model.audio_tower), gsum(model.score_tower), gsum(midi_encoder)
print(f'grad abs-sum: audio_tower={ga:.4f}  score_tower={gsc:.4f}  midi_encoder={gm:.4f}', flush=True)
assert ga > 0, 'FAIL: no gradient into (warm-started) audio tower'
assert gsc > 0, 'FAIL: no gradient into score tower'
assert gm > 0, 'FAIL: no gradient into midi encoder'

# --- decode sanity ---
assert np.all(np.diff(path_d3) >= 0), 'FAIL: D3 DTW path not monotonic'
print(f'D3 DTW path monotonic, range=[{path_d3.min()},{path_d3.max()}] of W_col={W_col}', flush=True)

print('\nSMOKE TEST PASSED', flush=True)
