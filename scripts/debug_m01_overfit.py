"""Diagnostic: can M01 overfit a single fixed training sample?
If loss doesn't drop sharply within ~200 steps on ONE fixed sample, something
is broken in the loss/model/optimizer wiring (not a data-scale/generalization
issue, since there's nothing to generalize to here).
"""
import random, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mymodel.cadp.dataset import CADPDataset
from mymodel.cadp.m01_model import M01FrozenBaseline
from mymodel.cadp.m01_train import _build_training_sample
from mymodel.shared.losses import expected_distance_loss

random.seed(0); np.random.seed(0); torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, flush=True)

ds = CADPDataset('data/MSMD/processed', 'data/MSMD/mert_emb',
                  '/scratch/pmohseni/dinov2_emb', split='train', fps=20.0)
piece = None
for pid in ds.piece_ids:
    piece = ds.load_piece(pid)
    if piece is not None and piece['col_idx'].shape[0] > 10:
        break
print('piece', piece['pid'], 'n_cols', piece['d2_feats'].shape[0], flush=True)

model = M01FrozenBaseline().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

random.seed(42)
sample = _build_training_sample(piece, 20, 20.0, 5.0, device)
audio_t, score_t, pos_tile, pos_target, valid_mask = sample
print('valid_mask sum', valid_mask.sum().item(), '/', len(valid_mask), flush=True)

for step in range(300):
    out = model(audio_t, score_t)
    sim = out['sim'].squeeze(0)
    loss, parts = expected_distance_loss(sim, pos_tile, pos_target, valid_mask, temperature=0.07)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 20 == 0 or step == 299:
        print(f'step {step:3d}: loss={loss.item():.4f}  exp_dist={parts["exp_dist"].item():.4f}',
              flush=True)
print('DONE', flush=True)
