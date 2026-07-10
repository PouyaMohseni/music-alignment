"""E2/E3: v13/v14/v15's full-strip BPTT training + two MIDI-privileged
train-time-only signals (never used at inference):

1. Repeat-aware GT: make_gt_mask's single stripe -> ALSO stamp lower-weight
   (alt_weight, default 0.3) stripes at MIDI-discovered repeat-alternate
   columns, via np.maximum (true position stays at 1.0, dice loss no longer
   fully penalizes activation at a genuinely ambiguous repeat). Simpler than
   D2's Gaussian-mixture-then-renormalize CE target -- dice loss compares
   directly against graded target values, no renormalization needed.
2. MIDI->audio distillation: a forward hook captures network.perf_encoder's
   per-frame output (BEFORE the LSTM -- same integration point D2 used, the
   audio tower's own per-frame embedding, not a temporally-mixed one) and
   pulls it toward MidiEncoder(pitch_roll) via InfoNCE (mymodel.d2_midi_privileged
   .losses.midi_distill_loss, reused unchanged). MidiEncoder is a pure
   training scaffold, discarded at inference -- eval.py (v13's own, unmodified)
   never imports it.

Usage:
    python -m mymodel.v13_midi_privileged.train --config configs/v13_midi_privileged.yaml
"""
from __future__ import annotations
import argparse, os, random, time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.v13_mert_unet.data import make_gt_mask
from mymodel.v13_midi_privileged.data import MidiPrivilegedFullStripDataset
from mymodel.d2_midi_privileged.midi_encoder import MidiEncoder
from mymodel.d2_midi_privileged.losses import midi_distill_loss

EPS = 1e-8


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def dice_loss(inputs, targets, smoothing=0.0):
    iflat = inputs.reshape(-1)
    tflat = targets.reshape(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2.0 * intersection + smoothing) /
                (iflat.pow(2).sum() + tflat.pow(2).sum() + smoothing + EPS))


def _build_perf(feats, frames, n_frames):
    T, mert_dim = feats.shape
    sl = len(frames)
    out = np.zeros((sl, 1, mert_dim, n_frames), dtype=np.float32)
    for i, t in enumerate(frames):
        t = min(t, T - 1)
        t0 = max(0, t - n_frames + 1)
        window = feats[t0:t + 1]
        actual_len = window.shape[0]
        if actual_len < n_frames:
            pad = np.zeros((n_frames - actual_len, mert_dim), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        out[i, 0] = window.T
    return out


def _build_gt_repeat_aware(H, W_sc, strip_x_sc, frames, gt_width, repeat_alt_cols, alt_weight):
    sl = len(frames)
    out = np.zeros((sl, 1, H, W_sc), dtype=np.float32)
    for i, t in enumerate(frames):
        cx = int(np.clip(round(strip_x_sc[t]), 0, W_sc - 1))
        base = make_gt_mask(H, W_sc, cx=cx, gt_width=gt_width)
        alts = repeat_alt_cols.get(cx, []) if alt_weight > 0 else []
        if alts:
            for ac in alts:
                ac = int(np.clip(ac, 0, W_sc - 1))
                alt_mask = make_gt_mask(H, W_sc, cx=ac, gt_width=gt_width) * alt_weight
                base = np.maximum(base, alt_mask)
        out[i, 0] = base
    return out


class PerfEncoderCapture:
    """Forward hook capturing network.perf_encoder's per-frame output --
    the audio tower's OWN embedding, before any LSTM temporal mixing, same
    integration point D2's distillation used on its analogous audio tower."""
    def __init__(self, network):
        self.output = None
        self._handle = network.perf_encoder.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.output = output


def _init_hidden(network, device):
    if not network.use_lstm:
        return None
    return (torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
            torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device))


def _detach_hidden(hidden):
    return tuple(h.detach() for h in hidden) if hidden else None


def _forward_chunk(network, midi_encoder, perf_capture, score_1frame, feats, pitch_roll,
                   strip_x_sc, frames, n_frames, gt_width, repeat_alt_cols, alt_weight,
                   distill_weight, hidden, device):
    sl = len(frames)
    H = score_1frame.shape[-2]
    W_sc = score_1frame.shape[-1]

    score_batch = score_1frame.expand(sl, -1, -1, -1, -1)
    perf_np = _build_perf(feats, frames, n_frames)
    perf_batch = torch.from_numpy(perf_np).to(device).unsqueeze(1)

    gt_np = _build_gt_repeat_aware(H, W_sc, strip_x_sc, frames, gt_width, repeat_alt_cols, alt_weight)
    gt_batch = torch.from_numpy(gt_np).to(device)

    out = network(score=score_batch, perf=perf_batch, hidden=hidden)
    pred = out['segmentation']
    new_hidden = _detach_hidden(out.get('hidden'))

    loss = dice_loss(pred, gt_batch, smoothing=0.0)
    logs = {'dice': float(loss.detach()), 'distill': 0.0}

    if distill_weight > 0:
        audio_emb = perf_capture.output.reshape(sl, -1)     # (sl, spec_enc)
        pr = torch.from_numpy(np.stack([pitch_roll[min(f, pitch_roll.shape[0] - 1)] for f in frames]))
        pr = pr.to(device)
        midi_emb = midi_encoder(pr)                          # (sl, spec_enc)
        distill = midi_distill_loss(audio_emb, midi_emb, temperature=0.1)
        loss = loss + distill_weight * distill
        logs['distill'] = float(distill.detach())

    with torch.no_grad():
        col = pred.squeeze(1).sum(dim=1)
        pred_x = col.argmax(dim=-1).float()
        gt_x = torch.tensor([strip_x_sc[f] for f in frames], device=device, dtype=torch.float32)
        acc = ((pred_x - gt_x).abs() <= W_sc * 0.1).float().mean()

    return loss, float(acc), new_hidden, logs


def _train_epoch(network, midi_encoder, perf_capture, dataset, optimizer, seq_len, n_frames,
                 gt_width, alt_weight, distill_weight, device):
    network.train(); midi_encoder.train()
    order = list(range(len(dataset)))
    np.random.shuffle(order)
    total_loss = total_acc = total_distill = n_seen = 0

    for idx in order:
        p = dataset[idx]
        score_1 = torch.from_numpy(p['score'][np.newaxis, np.newaxis, np.newaxis]).to(device)
        feats = p['feats']
        T = p['T']

        hidden = _init_hidden(network, device)
        t = 0
        while t < T:
            frames = list(range(t, min(t + seq_len, T)))
            sl = len(frames)
            optimizer.zero_grad(set_to_none=True)
            loss, acc, hidden, logs = _forward_chunk(
                network, midi_encoder, perf_capture, score_1, feats, p['pitch_roll'],
                p['strip_x_sc'], frames, n_frames, gt_width, p['repeat_alt_cols'],
                alt_weight, distill_weight, hidden, device)
            loss.backward()
            optimizer.step()
            total_loss += logs['dice'] * sl
            total_distill += logs['distill'] * sl
            total_acc += acc * sl
            n_seen += sl
            t += sl

    denom = max(1, n_seen)
    return total_loss / denom, total_acc / denom, total_distill / denom


@torch.no_grad()
def _val_epoch(network, midi_encoder, perf_capture, dataset, seq_len, n_frames, gt_width,
              alt_weight, distill_weight, device):
    network.eval(); midi_encoder.eval()
    total_loss = n_seen = 0
    for idx in range(len(dataset)):
        p = dataset[idx]
        score_1 = torch.from_numpy(p['score'][np.newaxis, np.newaxis, np.newaxis]).to(device)
        feats = p['feats']
        T = p['T']
        hidden = _init_hidden(network, device)
        t = 0
        while t < T:
            frames = list(range(t, min(t + seq_len, T)))
            loss, _, hidden, logs = _forward_chunk(
                network, midi_encoder, perf_capture, score_1, feats, p['pitch_roll'],
                p['strip_x_sc'], frames, n_frames, gt_width, p['repeat_alt_cols'],
                alt_weight, distill_weight, hidden, device)
            total_loss += logs['dice'] * len(frames)
            n_seen += len(frames)
            t += len(frames)
    network.train(); midi_encoder.train()
    return total_loss / max(1, n_seen)


def _save(network, midi_encoder, optimizer, scheduler, epoch, best_val, wait, cfg, out_dir, is_best):
    payload = {
        'epoch': epoch, 'state_dict': network.state_dict(),
        'midi_encoder_state_dict': midi_encoder.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'best_val_loss': best_val, 'wait': wait,
        'net_config': OmegaConf.to_container(cfg.net),
        'cfg': OmegaConf.to_container(cfg),
    }
    path = out_dir / f'checkpoint_epoch{epoch:03d}.pt'
    torch.save(payload, path)
    if is_best:
        torch.save(payload, out_dir / 'best_model.pt')
    old = out_dir / f'checkpoint_epoch{epoch - 2:03d}.pt'
    if old.exists():
        old.unlink()
    return path


def main(cfg: DictConfig, resume: str | None = None):
    _seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(os.getcwd()) / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device={device}  out={out_dir}', flush=True)

    train_ds = MidiPrivilegedFullStripDataset(cfg.data.processed_root, cfg.data.mert_emb_root,
                                              'train', h_strip=cfg.data.h_strip,
                                              w_scale=cfg.data.w_scale, fps=cfg.data.fps,
                                              repeat_k=cfg.loss.repeat_k)
    val_ds = MidiPrivilegedFullStripDataset(cfg.data.processed_root, cfg.data.mert_emb_root,
                                            'val', h_strip=cfg.data.h_strip,
                                            w_scale=cfg.data.w_scale, fps=cfg.data.fps,
                                            repeat_k=cfg.loss.repeat_k)
    print(f'train={len(train_ds)}  val={len(val_ds)}', flush=True)

    net_config = OmegaConf.to_container(cfg.net)
    network = ConditionalUNet(net_config)
    network.perf_encoder.set_stats(None, None)
    network = network.to(device)
    midi_encoder = MidiEncoder(d_model=cfg.net.spec_enc).to(device)
    perf_capture = PerfEncoderCapture(network)
    print(f'network params: {sum(p.numel() for p in network.parameters() if p.requires_grad):,}  '
          f'midi_encoder params: {sum(p.numel() for p in midi_encoder.parameters()):,} '
          f'(train-only, discarded at inference)', flush=True)

    optimizer = torch.optim.Adam(
        list(network.parameters()) + list(midi_encoder.parameters()),
        lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=cfg.optim.patience, factor=0.5)

    max_epochs = cfg.train.max_epochs
    early_patience = cfg.optim.patience * 2
    best_val = float('inf')
    wait = 0
    start_epoch = 1

    ckpt_to_load = None
    if resume:
        ckpt_to_load = resume if resume != 'auto' else None
        if ckpt_to_load is None:
            candidates = sorted(out_dir.glob('checkpoint_epoch*.pt'))
            if candidates:
                ckpt_to_load = str(candidates[-1])
                print(f'Auto-resume: {ckpt_to_load}', flush=True)

    if ckpt_to_load and Path(ckpt_to_load).exists():
        ckpt = torch.load(ckpt_to_load, map_location='cpu', weights_only=False)
        network.load_state_dict(ckpt['state_dict'])
        midi_encoder.load_state_dict(ckpt['midi_encoder_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_val = ckpt['best_val_loss']
        wait = ckpt['wait']
        start_epoch = ckpt['epoch'] + 1
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f'Resumed epoch={ckpt["epoch"]}  best={best_val:.5f}  wait={wait}', flush=True)

    t0 = time.time()
    n_frames = cfg.data.n_frames
    gt_width = cfg.data.gt_width
    alt_weight = cfg.loss.repeat_alt_weight
    distill_weight = cfg.loss.distill_weight

    for epoch in range(start_epoch, max_epochs + 1):
        tr_loss, tr_acc, tr_distill = _train_epoch(
            network, midi_encoder, perf_capture, train_ds, optimizer,
            seq_len=cfg.train.seq_len, n_frames=n_frames, gt_width=gt_width,
            alt_weight=alt_weight, distill_weight=distill_weight, device=device)
        val_loss = _val_epoch(
            network, midi_encoder, perf_capture, val_ds,
            seq_len=cfg.train.seq_len, n_frames=n_frames, gt_width=gt_width,
            alt_weight=alt_weight, distill_weight=distill_weight, device=device)
        scheduler.step(val_loss)

        is_best = val_loss < best_val
        print(f'epoch {epoch:3d}/{max_epochs}  '
              f'train_dice={tr_loss:.4f}  acc={tr_acc:.3f}  train_distill={tr_distill:.4f}  '
              f'val_dice={val_loss:.4f}  '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}  '
              f'{time.time()-t0:.0f}s', flush=True)

        if is_best:
            best_val = val_loss
            wait = 0
            _save(network, midi_encoder, optimizer, scheduler, epoch, best_val, wait, cfg, out_dir, True)
            print(f'  -> new best (val={best_val:.5f})', flush=True)
        else:
            wait += 1
            _save(network, midi_encoder, optimizer, scheduler, epoch, best_val, wait, cfg, out_dir, False)
            print(f'  no improvement ({wait}/{early_patience})', flush=True)
            if wait >= early_patience:
                print(f'Early stopping at epoch {epoch}.', flush=True)
                break

    print(f'Done. best_val_loss={best_val:.5f}', flush=True)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', nargs='?', const='auto', default=None)
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    return cfg, a.resume


if __name__ == '__main__':
    cfg, resume = _parse()
    main(cfg, resume=resume)
