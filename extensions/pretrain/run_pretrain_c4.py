"""C4 entry point: standalone tempo-invariant contrastive pretraining for
CBEncoder (see tempo_contrastive.py for the method). This does NOT go
through CPJKU's train_model.py -- it pretrains CBEncoder in isolation, not
the full ConditionalUNet, since there's no FiLM/score/RNN involved in this
self-supervised stage.

Later usage (NOT implemented here -- follow-up integration step): to
warm-start a real CB_TA training run from this pretrained encoder, load
just the perf_encoder sub-state-dict into a freshly constructed
ConditionalUNet before calling train_model.py, e.g.:

    network = ConditionalUNet(net_config)
    pretrained = torch.load('c4_checkpoint.pt')
    network.perf_encoder.load_state_dict(pretrained['encoder_state_dict'])
    # then proceed with normal training (optionally freezing perf_encoder
    # for the first N epochs)

    python extensions/pretrain/run_pretrain_c4.py \
        --train_dir /scratch/pmohseni/msmd_train_full \
        --sound_font third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2 \
        --fluidsynth /scratch/pmohseni/micromamba/envs/fluidsynth/bin/fluidsynth \
        --out_dir /scratch/pmohseni/results/c4_tempo_contrastive
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet'))

from extensions.pretrain.tempo_contrastive import sample_tempo_pair_batch, info_nce_loss

TEMPO_FACTORS = [500, 750, 950, 1000, 1050, 1250, 1500]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', required=True, help='e.g. /scratch/pmohseni/msmd_train_full')
    p.add_argument('--sound_font', required=True)
    p.add_argument('--fluidsynth', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--spec_enc', type=int, default=32, help='matches CB_TA net_config spec_enc')
    p.add_argument('--batch_size', type=int, default=32, help='onset pairs per step')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--save_every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    from audio_conditioned_unet.audio_encoder import CBEncoder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = CBEncoder(a.spec_enc).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=a.lr)

    start_step = 0
    resume_path = Path(a.out_dir) / 'c4_encoder_latest.pt'
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        start_step = ckpt['step']
        print(f'Resuming from {resume_path} at step {start_step}', flush=True)

    perf_dir = Path(a.train_dir) / 'performance'
    piece_stems = sorted({
        f.name.rsplit('_tempo_', 1)[0]
        for f in perf_dir.glob('*_tempo_*.mid')
    })
    print(f'{len(piece_stems)} pieces available for tempo-pair sampling', flush=True)

    rng = np.random.default_rng(a.seed)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    running_loss = 0.0
    n_logged = 0
    for step in range(start_step + 1, a.steps + 1):
        piece = piece_stems[rng.integers(len(piece_stems))]
        tempo_a, tempo_b = rng.choice(TEMPO_FACTORS, size=2, replace=False)

        batch_a, batch_b, k = sample_tempo_pair_batch(
            piece, str(perf_dir), int(tempo_a), int(tempo_b),
            a.sound_font, a.fluidsynth, a.batch_size, rng)
        if k < 2:
            continue   # need >=2 onsets for a meaningful in-batch-negative contrastive loss

        t_a = torch.from_numpy(batch_a).to(device)
        t_b = torch.from_numpy(batch_b).to(device)

        emb_a = encoder(t_a)   # (k, spec_enc)
        emb_b = encoder(t_b)

        loss = info_nce_loss(emb_a, emb_b, temperature=a.temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_logged += 1

        if step % 100 == 0:
            print(f'step {step}/{a.steps}  loss={running_loss / max(n_logged, 1):.4f}  '
                  f'piece={piece}  tempo_pair=({tempo_a},{tempo_b})  k={k}', flush=True)
            running_loss = 0.0
            n_logged = 0

        if step % a.save_every == 0:
            ckpt_path = out_dir / f'c4_encoder_step{step}.pt'
            torch.save({'encoder_state_dict': encoder.state_dict(), 'step': step}, ckpt_path)
            torch.save({'encoder_state_dict': encoder.state_dict(), 'step': step},
                       out_dir / 'c4_encoder_latest.pt')
            print(f'  saved {ckpt_path}', flush=True)

    print('Pretraining finished.', flush=True)


if __name__ == '__main__':
    main()
