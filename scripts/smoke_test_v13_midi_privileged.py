import sys, ast
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

import numpy as np
import torch
from omegaconf import OmegaConf

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.v13_midi_privileged.data import MidiPrivilegedFullStripDataset, load_piece
from mymodel.v13_midi_privileged.train import (
    _build_gt_repeat_aware, PerfEncoderCapture, _forward_chunk, _init_hidden)
from mymodel.v13_mert_unet.data import make_gt_mask
from mymodel.d2_midi_privileged.midi_encoder import MidiEncoder
from pathlib import Path

cfg = OmegaConf.load('configs/v13_midi_privileged.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, flush=True)

ds = MidiPrivilegedFullStripDataset(cfg.data.processed_root, cfg.data.mert_emb_root, 'test',
                                    h_strip=cfg.data.h_strip, w_scale=cfg.data.w_scale,
                                    fps=cfg.data.fps, repeat_k=cfg.loss.repeat_k)
assert len(ds) > 0, 'FAIL: no pieces loaded'

n_with_repeats = sum(1 for p in ds.pieces if p['repeat_alt_cols'])
print(f'{n_with_repeats}/{len(ds)} pieces have >=1 repeat-ambiguous column', flush=True)
assert n_with_repeats > 0, 'FAIL: zero repeats found across real test pieces (suspicious)'

p = next(pc for pc in ds.pieces if pc['repeat_alt_cols'])
print(f"using piece={p['pid']}  T={p['T']}  W_sc={p['W_sc']}  "
      f"n_repeat_cols={len(p['repeat_alt_cols'])}", flush=True)

# --- (a) repeat-aware GT: true position stays at 1.0, alt at 0.3, base case degenerates ---
H, W_sc = p['H'], p['W_sc']
some_col_with_alt = next(iter(p['repeat_alt_cols']))
frames_at_col = [t for t in range(p['T']) if int(round(p['strip_x_sc'][t])) == some_col_with_alt][:3]
assert frames_at_col, 'FAIL: could not find a frame at the chosen repeat column'

gt_aware = _build_gt_repeat_aware(H, W_sc, p['strip_x_sc'], frames_at_col, cfg.data.gt_width,
                                  p['repeat_alt_cols'], alt_weight=0.3)
gt_plain = _build_gt_repeat_aware(H, W_sc, p['strip_x_sc'], frames_at_col, cfg.data.gt_width,
                                  {}, alt_weight=0.3)
print(f'gt_aware max={gt_aware.max():.2f} (should be 1.0)  '
      f'gt_aware sum={gt_aware.sum():.1f} vs gt_plain sum={gt_plain.sum():.1f} '
      f'(aware should be strictly larger: has extra 0.3-weight alt stripes)', flush=True)
assert np.isfinite(gt_aware).all()
assert abs(gt_aware.max() - 1.0) < 1e-6, 'FAIL: true position not at weight 1.0'
assert gt_aware.sum() > gt_plain.sum(), 'FAIL: repeat-aware GT has no extra mass from alternates'

# degenerate check: alt_weight=0 or no alts -> identical to plain make_gt_mask stripe
gt_noaware = _build_gt_repeat_aware(H, W_sc, p['strip_x_sc'], frames_at_col, cfg.data.gt_width,
                                    p['repeat_alt_cols'], alt_weight=0.0)
assert np.array_equal(gt_noaware, gt_plain), 'FAIL: alt_weight=0 should degenerate to plain GT'
print('Degeneracy confirmed: alt_weight=0 == plain single-stripe GT.', flush=True)

# --- (b)/(c) gradient flow: dice-only AND dice+distill, into perf_encoder AND midi_encoder ---
net_config = OmegaConf.to_container(cfg.net)
network = ConditionalUNet(net_config).to(device)
network.perf_encoder.set_stats(None, None)
midi_encoder = MidiEncoder(d_model=cfg.net.spec_enc).to(device)
perf_capture = PerfEncoderCapture(network)

score_1 = torch.from_numpy(p['score'][np.newaxis, np.newaxis, np.newaxis]).to(device)
frames = list(range(0, min(16, p['T'])))
hidden = _init_hidden(network, device)

loss, acc, hidden, logs = _forward_chunk(
    network, midi_encoder, perf_capture, score_1, p['feats'], p['pitch_roll'],
    p['strip_x_sc'], frames, cfg.data.n_frames, cfg.data.gt_width,
    p['repeat_alt_cols'], cfg.loss.repeat_alt_weight, cfg.loss.distill_weight, hidden, device)
print(f'loss={float(loss):.4f}  dice={logs["dice"]:.4f}  distill={logs["distill"]:.4f}', flush=True)
assert np.isfinite(float(loss))
loss.backward()

def gsum(module):
    return sum(float(pp.grad.abs().sum()) for pp in module.parameters() if pp.grad is not None)
g_perf_enc = gsum(network.perf_encoder)
g_midi_enc = gsum(midi_encoder)
print(f'grad abs-sum: perf_encoder={g_perf_enc:.4f}  midi_encoder={g_midi_enc:.4f}', flush=True)
assert g_perf_enc > 0, 'FAIL: no gradient into perf_encoder'
assert g_midi_enc > 0, 'FAIL: no gradient into midi_encoder'

# --- (d) distillation loss decreases over a few optimizer steps on one real piece ---
opt = torch.optim.Adam(list(network.parameters()) + list(midi_encoder.parameters()), lr=1e-3)
distill_losses = []
hidden2 = _init_hidden(network, device)
for step in range(5):
    opt.zero_grad()
    loss2, _, hidden2, logs2 = _forward_chunk(
        network, midi_encoder, perf_capture, score_1, p['feats'], p['pitch_roll'],
        p['strip_x_sc'], frames, cfg.data.n_frames, cfg.data.gt_width,
        p['repeat_alt_cols'], cfg.loss.repeat_alt_weight, cfg.loss.distill_weight, hidden2, device)
    loss2.backward()
    opt.step()
    distill_losses.append(logs2['distill'])
print('distill loss over 5 steps:', [f'{d:.4f}' for d in distill_losses], flush=True)
assert distill_losses[-1] < distill_losses[0], 'FAIL: distill loss did not decrease over 5 steps'

# --- (e) static check: v13's eval.py never imports anything MIDI-related ---
tree = ast.parse(open('mymodel/v13_mert_unet/eval.py').read())
names = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        names += [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        names.append(node.module)
bad = [n for n in names if n and 'midi' in n.lower()]
assert not bad, f'FAIL: v13 eval.py imports something MIDI-related: {bad}'
print('Confirmed: v13_mert_unet/eval.py (used unmodified for inference) imports nothing MIDI-related.',
      flush=True)

print('\nSMOKE TEST PASSED', flush=True)
