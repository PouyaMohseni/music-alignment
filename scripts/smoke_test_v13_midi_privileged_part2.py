import sys, ast
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

import numpy as np
import torch
from omegaconf import OmegaConf

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.v13_midi_privileged.data import MidiPrivilegedFullStripDataset
from mymodel.v13_midi_privileged.train import PerfEncoderCapture, _forward_chunk, _init_hidden
from mymodel.d2_midi_privileged.midi_encoder import MidiEncoder

cfg = OmegaConf.load('configs/v13_midi_privileged.yaml')
device = torch.device('cpu')

ds = MidiPrivilegedFullStripDataset(cfg.data.processed_root, cfg.data.mert_emb_root, 'test',
                                    h_strip=cfg.data.h_strip, w_scale=cfg.data.w_scale,
                                    fps=cfg.data.fps, repeat_k=cfg.loss.repeat_k)
# smallest piece with repeats, for a fast multi-step check
candidates = [p for p in ds.pieces if p['repeat_alt_cols']]
p = min(candidates, key=lambda pc: pc['T'])
print(f"using piece={p['pid']}  T={p['T']}  W_sc={p['W_sc']}", flush=True)

net_config = OmegaConf.to_container(cfg.net)
network = ConditionalUNet(net_config).to(device)
network.perf_encoder.set_stats(None, None)
midi_encoder = MidiEncoder(d_model=cfg.net.spec_enc).to(device)
perf_capture = PerfEncoderCapture(network)

score_1 = torch.from_numpy(p['score'][np.newaxis, np.newaxis, np.newaxis]).to(device)
frames = list(range(0, min(16, p['T'])))

opt = torch.optim.Adam(list(network.parameters()) + list(midi_encoder.parameters()), lr=1e-3)
distill_losses = []
hidden = _init_hidden(network, device)
for step in range(3):
    opt.zero_grad()
    loss, _, hidden, logs = _forward_chunk(
        network, midi_encoder, perf_capture, score_1, p['feats'], p['pitch_roll'],
        p['strip_x_sc'], frames, cfg.data.n_frames, cfg.data.gt_width,
        p['repeat_alt_cols'], cfg.loss.repeat_alt_weight, cfg.loss.distill_weight, hidden, device)
    loss.backward()
    opt.step()
    distill_losses.append(logs['distill'])
    print(f'  step {step}: distill={logs["distill"]:.4f}', flush=True)

assert distill_losses[-1] < distill_losses[0], 'FAIL: distill loss did not decrease'
print('Distill loss decreased over 3 steps.', flush=True)

tree = ast.parse(open('mymodel/v13_mert_unet/eval.py').read())
names = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        names += [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        names.append(node.module)
bad = [n for n in names if n and 'midi' in n.lower()]
assert not bad, f'FAIL: {bad}'
print('Confirmed: v13 eval.py imports nothing MIDI-related.', flush=True)
print('\nSMOKE TEST PART 2 PASSED', flush=True)
