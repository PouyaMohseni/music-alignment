"""Diagnostic: does accumulating gradients over a batch of pieces (instead of
stepping the optimizer after every single piece) recover real learning under
resampling noise? This tests the hypothesis that M01's real training loop's
effective batch size of 1 is why per-step target diversity swamps the signal.
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

N_PIECES = 50
random.seed(0); np.random.seed(0); torch.manual_seed(0)
pieces = []
for pid in ds.piece_ids:
    p = ds.load_piece(pid)
    if p is not None and p['col_idx'].shape[0] > 10:
        pieces.append(p)
    if len(pieces) >= N_PIECES:
        break
print(f'{len(pieces)} pieces loaded', flush=True)

for BATCH_SIZE in [1, 8, 32]:
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    model = M01FrozenBaseline().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    n_steps = 300
    for step in range(n_steps):
        batch_pieces = random.sample(pieces, min(BATCH_SIZE, len(pieces)))
        opt.zero_grad()
        batch_losses = []
        for p in batch_pieces:
            audio_t, score_t, pos_tile, pos_target, valid_mask = \
                _build_training_sample(p, 20, 20.0, 5.0, device)
            out = model(audio_t, score_t)
            sim = out['sim'].squeeze(0)
            loss, _ = expected_distance_loss(sim, pos_tile, pos_target, valid_mask, temperature=0.07)
            (loss / len(batch_pieces)).backward()
            batch_losses.append(loss.item())
        opt.step()
        losses.append(np.mean(batch_losses))

    first20 = np.mean(losses[:20])
    last20 = np.mean(losses[-20:])
    print(f'BATCH_SIZE={BATCH_SIZE:3d}: loss[first20]={first20:.4f}  loss[last20]={last20:.4f}  '
          f'ratio={last20/max(first20,1e-9):.3f}', flush=True)

print('DONE', flush=True)
