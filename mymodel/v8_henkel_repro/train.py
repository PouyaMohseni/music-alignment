"""v8 training: faithful Henkel repro with Dice loss on position heatmap.

    python -m mymodel.v8_henkel_repro.train --config configs/v8_henkel_repro.yaml
"""
from __future__ import annotations
import argparse, math, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .model import HenkelRepro, HenkelReproConfig
from .data import HenkelDataset


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _lr(step, warmup, total, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * prog))


def dice_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss. pred, gt: (B, 1, W) in [0,1]."""
    pred = pred.view(pred.shape[0], -1)
    gt   = gt.view(gt.shape[0], -1)
    inter = (pred * gt).sum(dim=-1)
    union = pred.sum(dim=-1) + gt.sum(dim=-1)
    return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()


def _step_loss(model, b, device):
    cqt   = b["audio_cqt"].to(device)          # (B, 1, n_bins, T)
    strip = b["strip_win"].to(device)           # (B, 1, W)
    gt    = b["gt_mask"].unsqueeze(1).to(device) # (B, 1, W)

    pos_map, _ = model(cqt, strip)              # (B, 1, W)
    loss = dice_loss(pos_map, gt)
    # Accuracy proxy: is argmax within 10% of strip width from centre?
    half = pos_map.shape[-1] // 2
    margin = pos_map.shape[-1] // 10
    pred_center = pos_map.squeeze(1).argmax(dim=-1)  # (B,)
    acc = ((pred_center - half).abs() <= margin).float().mean()
    return loss, float(loss.detach()), float(acc.detach())


def _save(model, step, cfg, out_dir):
    path = Path(out_dir) / f"checkpoint_{step:06d}.pt"
    torch.save({"step": step,
                "state_dict": model.state_dict(),
                "cfg": OmegaConf.to_container(cfg)}, path)
    return path


@torch.no_grad()
def _validate(model, loader, device):
    model.eval(); losses = []
    for b in loader:
        _, ce, _ = _step_loss(model, b, device)
        losses.append(ce)
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main(cfg: DictConfig):
    _seed(cfg.seed)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(os.getcwd()) / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}", flush=True)

    def _loader(split, shuffle):
        ds = HenkelDataset(
            cfg.data.processed_root, split,
            window_sec=cfg.data.window_sec,
            tile_width=cfg.model.tile_width,
            n_bins=cfg.model.n_bins,
            hop=cfg.data.hop,
            sr=cfg.data.sr,
            sigma_px=cfg.data.sigma_px)
        return DataLoader(ds, batch_size=cfg.data.batch_size,
                          shuffle=shuffle, num_workers=cfg.data.num_workers,
                          pin_memory=device == "cuda")

    tl = _loader("train", shuffle=True)
    try:    vl = _loader("val", shuffle=False)
    except: vl = None
    print(f"train: {len(tl.dataset)}  val: {len(vl.dataset) if vl else 0}", flush=True)

    hc = HenkelReproConfig(
        n_bins=cfg.model.n_bins,
        cnn_channels=list(cfg.model.cnn_channels),
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        lstm_bidirectional=cfg.model.get("lstm_bidirectional", False),
        unet_channels=list(cfg.model.unet_channels),
        tile_width=cfg.model.tile_width)
    model = HenkelRepro(hc).to(device)
    print(f"trainable params: {model.num_trainable_params():,}", flush=True)

    optim = torch.optim.Adam(model.parameters(),
                              lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    accum = cfg.train.get("grad_accum_steps", 1)
    it    = iter(tl); t0 = time.time(); optim.zero_grad(set_to_none=True)

    for step in range(1, cfg.train.steps + 1):
        tot_loss = 0.0; tot_acc = 0.0
        for _ in range(accum):
            try: b = next(it)
            except StopIteration: it = iter(tl); b = next(it)
            loss, lv, acc = _step_loss(model, b, device)
            (loss / accum).backward()
            tot_loss += lv / accum; tot_acc += acc / accum

        lr = _lr(step, cfg.optim.warmup_steps, cfg.train.steps, cfg.optim.lr)
        for g in optim.param_groups: g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optim.step(); optim.zero_grad(set_to_none=True)

        if step % cfg.train.log_every == 0 or step == 1:
            print(f"step {step:5d}/{cfg.train.steps}  dice={tot_loss:.4f}  "
                  f"acc={tot_acc:.3f}  lr={lr:.2e}  {time.time()-t0:.1f}s", flush=True)
        if vl and step % cfg.train.eval_every == 0:
            print(f"  [val @ {step}] dice={_validate(model, vl, device):.5f}", flush=True)
        if step % cfg.train.ckpt_every == 0:
            print(f"  -> saved {_save(model, step, cfg, out_dir)}", flush=True)

    print(f"done in {time.time()-t0:.1f}s", flush=True)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/v8_henkel_repro.yaml")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    return cfg


if __name__ == "__main__":
    main(_parse())
