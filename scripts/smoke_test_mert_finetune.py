"""Smoke test for v11-mert-finetune BEFORE submitting any real training job:
loads MERTLive, runs one BPTT-style forward+backward chunk on a real piece,
and verifies gradients actually reach MERT's unfrozen parameters (not just
that the code runs, but that fine-tuning is actually happening).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, '.')

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.v10_mert_unet.mert_live import MERTLive
from mymodel.v11_mert_finetune.data import load_piece
from mymodel.v11_mert_finetune.train import _forward_chunk, _init_hidden


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}', flush=True)

    cfg = OmegaConf.load('configs/v11_mert_finetune.yaml')
    net_config = OmegaConf.to_container(cfg.net)

    t0 = time.time()
    network = ConditionalUNet(net_config)
    print(f'network built ({time.time()-t0:.1f}s)', flush=True)

    t0 = time.time()
    mert_live = MERTLive(mert_id=cfg.mert.mert_id, unfreeze_last_n=cfg.mert.get('unfreeze_last_n', None))
    print(f'MERTLive built ({time.time()-t0:.1f}s)', flush=True)

    network = network.to(device)
    mert_live = mert_live.to(device)

    splits = json.load(open('data/MSMD/processed/splits.json'))
    pid = splits['train'][0]
    print(f'test piece: {pid}', flush=True)

    t0 = time.time()
    piece = load_piece(Path('data/MSMD/processed') / pid,
                       h_strip=cfg.data.h_strip, w_scale=cfg.data.w_scale, fps=cfg.data.fps)
    print(f'piece loaded ({time.time()-t0:.1f}s): T={piece["T"]} audio_samples={piece["audio"].shape[0]}',
          flush=True)

    score_1 = torch.from_numpy(piece['score'][None, None, None]).to(device)
    audio_24k = torch.from_numpy(piece['audio']).to(device)
    hidden = _init_hidden(network, device)

    frames = list(range(0, min(cfg.train.seq_len, piece['T'])))
    print(f'running _forward_chunk on frames {frames[0]}..{frames[-1]}', flush=True)

    trainable_params = list(network.parameters()) + mert_live.trainable_parameters()
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss, acc, new_hidden = _forward_chunk(
        network, mert_live, score_1, audio_24k, piece['strip_x_sc'],
        frames, cfg.data.fps, cfg.train.window_sec, cfg.data.gt_width, hidden, device)
    print(f'forward done ({time.time()-t0:.1f}s): loss={float(loss):.4f} acc={acc:.3f}', flush=True)

    t0 = time.time()
    loss.backward()
    print(f'backward done ({time.time()-t0:.1f}s)', flush=True)

    optimizer.step()

    mert_trainable = mert_live.trainable_parameters()
    n_with_grad = sum(1 for p in mert_trainable if p.grad is not None and p.grad.abs().sum().item() > 0)
    print(f'MERT trainable params: {len(mert_trainable)}, '
          f'with nonzero grad: {n_with_grad}', flush=True)

    net_grad_ok = all(p.grad is not None for p in network.parameters() if p.requires_grad)
    print(f'network params all have grad: {net_grad_ok}', flush=True)

    assert n_with_grad > 0, 'FAIL: no gradient reached MERT -- fine-tuning is not actually happening!'
    assert net_grad_ok, 'FAIL: network params missing gradient'
    print('SMOKE TEST PASSED', flush=True)


if __name__ == '__main__':
    main()
