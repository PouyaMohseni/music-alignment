"""Reproduce the batched-training NaN by mimicking m01_train.py's exact
accumulation loop with instrumentation after every single backward() call,
to find the precise piece/step where things go NaN.
"""
import random, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mymodel.cadp.dataset import CADPDataset
from mymodel.cadp.m01_model import M01FrozenBaseline
from mymodel.cadp.m01_train import _build_training_sample
from mymodel.shared.losses import expected_distance_loss

random.seed(42); np.random.seed(42); torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ds = CADPDataset('data/MSMD/processed', 'data/MSMD/mert_emb',
                  '/scratch/pmohseni/dinov2_emb', split='train', fps=20.0)
train_pieces = [p for pid in ds.piece_ids if (p := ds.load_piece(pid)) is not None]
print(f'{len(train_pieces)} pieces', flush=True)

model = M01FrozenBaseline().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 8
n_chunks = 20

random.shuffle(train_pieces)
opt.zero_grad()
for i, piece in enumerate(train_pieces):
    audio_t, score_t, pos_tile, pos_target, valid_mask = \
        _build_training_sample(piece, n_chunks, 20.0, 5.0, device)
    out = model(audio_t, score_t)
    sim = out['sim'].squeeze(0)
    loss, _ = expected_distance_loss(sim, pos_tile, pos_target, valid_mask, temperature=0.07)
    (loss / batch_size).backward()

    # check gradient state right now, before any clipping/stepping
    grad_norm = torch.norm(torch.stack(
        [p.grad.norm() for p in model.parameters() if p.grad is not None]))
    has_nan_grad = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    has_nan_loss = torch.isnan(loss).any() or torch.isinf(loss).any()
    if has_nan_grad or has_nan_loss or i % 50 == 0:
        print(f'i={i:3d}  pid={piece["pid"]}  loss={loss.item()}  '
              f'accumulated_grad_norm={grad_norm.item()}  '
              f'nan_in_grad={has_nan_grad}  nan_in_loss={has_nan_loss}', flush=True)
    if has_nan_grad or has_nan_loss:
        print('FOUND NaN -- stopping', flush=True)
        break

    if (i + 1) % batch_size == 0:
        pre_clip_norm = grad_norm.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # check param state after step
        has_nan_param = any(torch.isnan(p).any() for p in model.parameters())
        print(f'  -- STEP at i={i}: pre_clip_norm={pre_clip_norm:.4f}  nan_in_params_after_step={has_nan_param}',
              flush=True)
        opt.zero_grad()
        if has_nan_param:
            print('NaN detected in params -- stopping', flush=True)
            break

print('DONE', flush=True)
