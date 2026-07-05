"""Diagnostic: does resampling a FRESH random window every epoch (matching
real training) break convergence, vs. fixing the window once per piece
(previous diagnostic, which converged fine up to 50 pieces)?
"""
import random, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mymodel.cadp.dataset import CADPDataset
from mymodel.cadp.m01_model import M01FrozenBaseline
from mymodel.cadp.m01_train import _build_training_sample
from mymodel.shared.losses import expected_distance_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, flush=True)

ds = CADPDataset('data/MSMD/processed', 'data/MSMD/mert_emb',
                  '/scratch/pmohseni/dinov2_emb', split='train', fps=20.0)

for N_PIECES in [5, 10, 20]:
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    pieces = []
    for pid in ds.piece_ids:
        p = ds.load_piece(pid)
        if p is not None and p['col_idx'].shape[0] > 10:
            pieces.append(p)
        if len(pieces) >= N_PIECES:
            break

    model = M01FrozenBaseline().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(300):
        step_losses = []
        for p in pieces:
            # FRESH random window every call -- matches real m01_train.py
            audio_t, score_t, pos_tile, pos_target, valid_mask = \
                _build_training_sample(p, 20, 20.0, 5.0, device)
            out = model(audio_t, score_t)
            sim = out['sim'].squeeze(0)
            loss, _ = expected_distance_loss(sim, pos_tile, pos_target, valid_mask, temperature=0.07)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_losses.append(loss.item())
        losses.append(np.mean(step_losses))

    # report mean of first/last 20 steps to average out per-step window noise
    first20 = np.mean(losses[:20])
    last20 = np.mean(losses[-20:])
    print(f'N_PIECES={N_PIECES:3d}: loss[first20]={first20:.4f}  loss[last20]={last20:.4f}  '
          f'ratio={last20/max(first20,1e-9):.3f}', flush=True)

print('DONE', flush=True)
