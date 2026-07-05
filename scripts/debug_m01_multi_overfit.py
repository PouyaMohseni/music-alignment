"""Diagnostic: can M01 fit a SMALL FIXED set of pieces (not just one)?
The single-piece overfit test passed (loss 0.36 -> 0.015). Real training
(354 pieces, fresh random window each step) never moves. This finds where
generalization breaks down: 2 pieces? 5? 20?
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

for N_PIECES in [1, 2, 5, 10, 20, 50]:
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    pieces = []
    for pid in ds.piece_ids:
        p = ds.load_piece(pid)
        if p is not None and p['col_idx'].shape[0] > 10:
            pieces.append(p)
        if len(pieces) >= N_PIECES:
            break

    # Fix ONE window per piece up front (same samples reused every step).
    fixed_samples = []
    random.seed(42)
    for p in pieces:
        fixed_samples.append(_build_training_sample(p, 20, 20.0, 5.0, device))

    model = M01FrozenBaseline().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(300):
        step_losses = []
        for audio_t, score_t, pos_tile, pos_target, valid_mask in fixed_samples:
            out = model(audio_t, score_t)
            sim = out['sim'].squeeze(0)
            loss, _ = expected_distance_loss(sim, pos_tile, pos_target, valid_mask, temperature=0.07)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_losses.append(loss.item())
        losses.append(np.mean(step_losses))

    print(f'N_PIECES={N_PIECES:3d}: loss[0]={losses[0]:.4f}  loss[299]={losses[-1]:.4f}  '
          f'ratio={losses[-1]/max(losses[0],1e-9):.3f}', flush=True)

print('DONE', flush=True)
