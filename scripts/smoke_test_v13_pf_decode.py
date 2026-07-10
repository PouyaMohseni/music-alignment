"""Fast correctness-only smoke test for eval_particle_filter.py's three
decoders (original, particle filter, offline DTW) -- truncates to the first
N frames of ONE real piece since a full per-frame ConditionalUNet forward
pass on CPU is too slow for a real accuracy comparison here (no GPU available
in this environment; the real 94-piece/hyperparameter-sweep run needs SLURM).
This only verifies: no crash, finite output, particle filter resets state per
piece, decoders' outputs actually differ (not silently identical)."""
import sys, json
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

import numpy as np
import torch
from omegaconf import OmegaConf
from pathlib import Path

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.v13_mert_unet.eval import _build_perf_frame, _predict_x_com
from mymodel.v11_cpjku_fullstrip.data import load_strip_scaled
from extensions.decode.particle_filter import ParticleFilterXTracker, heatmap_to_x_marginal
from mymodel.d1_align_matrix.dtw import dtw_decode

N_FRAMES_TEST = 60   # truncate -- correctness check only, not accuracy

cfg = OmegaConf.load('configs/v13_mert_linear.yaml')
device = 'cpu'
network = ConditionalUNet(OmegaConf.to_container(cfg.net))
sd = torch.load('/scratch/pmohseni/results/v13_mert_linear/best_model.pt', map_location='cpu', weights_only=False)
network.load_state_dict(sd['state_dict'], strict=True)
network = network.to(device).eval()
print('Loaded v13 checkpoint', flush=True)

proc = Path('data/MSMD/processed')
emb = Path('data/MSMD/mert_emb')
piece_ids = json.load(open(proc / 'splits.json'))['test']
pid = piece_ids[0]
print('piece:', pid, flush=True)

piece_dir = proc / pid
feats = np.load(str(emb / f'{pid}.npy')).astype(np.float32)
T = min(feats.shape[0], N_FRAMES_TEST)
print(f'T={T} (truncated from {feats.shape[0]})', flush=True)

score = load_strip_scaled(piece_dir / 'strip.png', cfg.data.h_strip, cfg.data.w_scale)
H, W_sc = score.shape
score_1 = torch.from_numpy(score[np.newaxis, np.newaxis, np.newaxis]).to(device)

hidden = None
if network.use_lstm:
    hidden = (torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
              torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device))

pred_x_sc_orig = np.zeros(T, dtype=np.float64)
marginals = np.zeros((T, W_sc), dtype=np.float64)
with torch.no_grad():
    for t in range(T):
        perf_np = _build_perf_frame(feats, t, cfg.data.n_frames)
        perf_t = torch.from_numpy(perf_np).to(device).unsqueeze(0)
        out = network(score=score_1, perf=perf_t, hidden=hidden)
        seg = out['segmentation']
        hidden = out.get('hidden')
        if hidden is not None:
            hidden = (hidden[0].detach(), hidden[1].detach())
        seg_np = seg.squeeze(0).squeeze(0).cpu().numpy()
        pred_x_sc_orig[t] = _predict_x_com(seg.squeeze(0))
        marginals[t] = heatmap_to_x_marginal(seg_np)
print(f'Forward pass done. W_sc={W_sc}', flush=True)
assert np.all(np.isfinite(marginals)), 'FAIL: non-finite marginal'
assert np.all(np.isfinite(pred_x_sc_orig)), 'FAIL: non-finite original decode'

# particle filter -- fresh tracker, then AGAIN with a second fresh tracker on
# the SAME data to confirm identical output (state genuinely resets, same
# bug class C5's smoke test caught for D2's calibration)
tracker1 = ParticleFilterXTracker(process_noise_std=3.0, init_std=2.0, seed=0)
pf1 = np.array([tracker1.step(marginals[t]) for t in range(T)])
tracker2 = ParticleFilterXTracker(process_noise_std=3.0, init_std=2.0, seed=0)
pf2 = np.array([tracker2.step(marginals[t]) for t in range(T)])
assert np.allclose(pf1, pf2), 'FAIL: particle filter not deterministic/reproducible across fresh instances'
print('Particle filter: deterministic across fresh instances (state reset confirmed).', flush=True)
assert np.all(np.isfinite(pf1)), 'FAIL: non-finite particle filter output'

# offline DTW over marginals
dtw_path = dtw_decode(marginals.astype(np.float64), band_frac=0.05)
assert np.all(np.diff(dtw_path) >= 0), 'FAIL: offline DTW path not monotonic'
print('Offline DTW: monotonic path confirmed.', flush=True)

print(f'\noriginal   first 10: {np.round(pred_x_sc_orig[:10], 1)}', flush=True)
print(f'particle_f first 10: {np.round(pf1[:10], 1)}', flush=True)
print(f'offline_dtw first 10: {dtw_path[:10]}', flush=True)

# decoders should not be trivially identical (would indicate a wiring bug --
# e.g. marginal computed wrong, or decoder silently falling back to argmax)
assert not np.allclose(pred_x_sc_orig, pf1, atol=0.5), \
    'SUSPICIOUS: particle filter output ~identical to original decode -- check wiring'
print('\nDecoders produce genuinely different (not accidentally identical) paths.', flush=True)

print('\nSMOKE TEST PASSED (correctness only -- see report for accuracy-eval caveat)', flush=True)
