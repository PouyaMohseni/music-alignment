"""F1 smoke test on REAL data. Verifies: soft-DTW term finite on a real chunk,
gradient flows from the NEW dtw loss term specifically (compare grad norm at
dtw_weight=0 vs >0), the seq_len==1 (last-chunk) edge case doesn't NaN, and
the combined loss stays dice-dominant in scale."""
import sys
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

import numpy as np
import torch
from omegaconf import OmegaConf

import json
from pathlib import Path

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.v13_midi_privileged.data import load_piece as _load_one_piece
from mymodel.d2_midi_privileged.midi_encoder import MidiEncoder
from mymodel.v13_f1_combined.train import _forward_chunk, PerfEncoderCapture, _init_hidden

cfg = OmegaConf.load('configs/v13_f1_combined.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, flush=True)

# Load ONE real test piece directly (bypassing the whole-split loader, which
# is too slow to finish in this CPU-only sandbox's time budget -- same
# compute constraint prior forks hit on this exact architecture).
root = Path(cfg.data.processed_root)
piece_ids = json.load(open(root / 'splits.json'))['test']
p = None
for pid in piece_ids:
    p = _load_one_piece(root / pid, cfg.data.mert_emb_root, cfg.data.h_strip,
                        cfg.data.w_scale, cfg.data.fps, cfg.loss.repeat_k)
    if p is not None and any(p['repeat_alt_cols'].values()):
        break
assert p is not None, 'FAIL: could not load any real test piece'
print(f'piece loaded: T={p["T"]}  score shape={p["score"].shape}  '
      f'n_repeat_cols={sum(1 for v in p["repeat_alt_cols"].values() if v)}', flush=True)

net_config = OmegaConf.to_container(cfg.net)
network = ConditionalUNet(net_config)
network.perf_encoder.set_stats(None, None)
network = network.to(device)
midi_encoder = MidiEncoder(d_model=cfg.net.spec_enc).to(device)
perf_capture = PerfEncoderCapture(network)

score_1 = torch.from_numpy(p['score'][np.newaxis, np.newaxis, np.newaxis]).to(device)
feats = p['feats']
seq_len = cfg.train.seq_len
n_frames = cfg.data.n_frames
gt_width = cfg.data.gt_width

def run(frames, dtw_weight):
    hidden = _init_hidden(network, device)
    for p_ in network.parameters():
        p_.grad = None
    for p_ in midi_encoder.parameters():
        p_.grad = None
    loss, acc, _, logs = _forward_chunk(
        network, midi_encoder, perf_capture, score_1, feats, p['pitch_roll'],
        p['strip_x_sc'], frames, n_frames, gt_width, p['repeat_alt_cols'],
        cfg.loss.repeat_alt_weight, cfg.loss.distill_weight,
        dtw_weight, cfg.loss.dtw_gamma, hidden, device)
    loss.backward()
    gsum = sum(float(pr.grad.abs().sum()) for pr in network.parameters() if pr.grad is not None)
    return float(loss), logs, gsum

# --- normal chunk, seq_len frames ---
frames = list(range(0, min(seq_len, p['T'])))
loss0, logs0, gsum0 = run(frames, dtw_weight=0.0)
loss1, logs1, gsum1 = run(frames, dtw_weight=cfg.loss.dtw_weight)
print(f'dtw_weight=0:   loss={loss0:.4f}  dice={logs0["dice"]:.4f}  distill={logs0["distill"]:.4f}  '
      f'dtw={logs0["dtw"]:.4f}  grad_sum={gsum0:.4f}', flush=True)
print(f'dtw_weight=0.1: loss={loss1:.4f}  dice={logs1["dice"]:.4f}  distill={logs1["distill"]:.4f}  '
      f'dtw={logs1["dtw"]:.4f}  grad_sum={gsum1:.4f}', flush=True)

assert np.isfinite(loss1), 'FAIL: non-finite combined loss'
assert np.isfinite(logs1['dtw']), 'FAIL: non-finite dtw term'
assert logs1['dtw'] != 0.0, 'FAIL: dtw term is exactly zero, likely not wired in'
assert gsum1 != gsum0, 'FAIL: gradient identical with dtw_weight=0 vs >0 -- dtw term not contributing gradient'
assert cfg.loss.dtw_weight * logs1['dtw'] < logs1['dice'], \
    'FAIL: dtw term is NOT dice-dominant (B3 lesson violated)'
print('Gradient changes with dtw_weight, dtw term is dice-subordinate in scale: PASS', flush=True)

# --- seq_len==1 edge case (last chunk of a piece) ---
last_frame = [p['T'] - 1]
loss_last, logs_last, _ = run(last_frame, dtw_weight=cfg.loss.dtw_weight)
print(f'seq_len=1 chunk: loss={loss_last:.4f}  dtw={logs_last["dtw"]:.4f}', flush=True)
assert np.isfinite(loss_last), 'FAIL: seq_len=1 produced non-finite loss (the exact NaN class that bit B4)'
print('seq_len=1 edge case: PASS (no NaN)', flush=True)

print('\nSMOKE TEST PASSED', flush=True)
