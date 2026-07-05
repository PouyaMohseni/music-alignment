"""Find which training piece(s) produce NaN/Inf loss or gradients, which
would poison an entire accumulated gradient batch under the new batched
training loop (previously only corrupted one single-piece step).
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

ds = CADPDataset('data/MSMD/processed', 'data/MSMD/mert_emb',
                  '/scratch/pmohseni/dinov2_emb', split='train', fps=20.0)
pieces = [(pid, p) for pid in ds.piece_ids if (p := ds.load_piece(pid)) is not None]
print(f'{len(pieces)} pieces loaded', flush=True)

model = M01FrozenBaseline().to(device)
n_bad = 0
for pid, p in pieces:
    # try a few random windows per piece since window sampling is random
    for trial in range(5):
        audio_t, score_t, pos_tile, pos_target, valid_mask = \
            _build_training_sample(p, 20, 20.0, 5.0, device)
        out = model(audio_t, score_t)
        sim = out['sim'].squeeze(0)
        if torch.isnan(sim).any() or torch.isinf(sim).any():
            print(f'BAD SIM: {pid} trial {trial}', flush=True)
            n_bad += 1
            continue
        loss, _ = expected_distance_loss(sim, pos_tile, pos_target, valid_mask, temperature=0.07)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f'BAD LOSS: {pid} trial {trial}  n_cols={p["d2_feats"].shape[0]}  '
                  f'valid={valid_mask.sum().item()}', flush=True)
            n_bad += 1
            continue
        model.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f'BAD GRAD: {pid} trial {trial}  param={name}', flush=True)
                n_bad += 1
                break

print(f'DONE — {n_bad} bad (piece, trial) combinations out of {len(pieces)*5}', flush=True)
