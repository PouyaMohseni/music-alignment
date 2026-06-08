"""v3 full-sequence training with the distance-aware localization loss.

    python -m mymodel.v3_fullseq.train --config configs/v3_fullseq.yaml

One step = one full performance (batch_size 1) with gradient accumulation for
an effective batch. Encoders are NOT in the graph — only the projection /
cross-attention head trains, on cached embeddings.
"""
from __future__ import annotations
import argparse, math, os, random, time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .model import FullSeqAlignmentModel, FullSeqModelConfig
from .data import build_loader
from ..shared.losses import expected_distance_loss


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _lr(step, warmup, total, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * prog))


def _to(batch, device):
    for k in ("audio_emb", "tile_emb", "pos_tile", "pos_target", "valid_mask"):
        batch[k] = batch[k].to(device)
    return batch


@torch.no_grad()
def _validate(model, loader, device, cfg):
    model.eval()
    errs = []
    for b in loader:
        b = _to(b, device)
        out = model(b["audio_emb"].unsqueeze(0), b["tile_emb"].unsqueeze(0))
        sim = out["sim"][0]                                   # (T, N)
        _, parts = expected_distance_loss(
            sim, b["pos_tile"], b["pos_target"], b["valid_mask"],
            temperature=cfg.loss.temperature, power=cfg.loss.power)
        errs.append(parts["exp_dist"].item())
    model.train()
    return float(np.mean(errs)) if errs else float("nan")


def _save(model, optim, step, cfg, out_dir):
    path = Path(out_dir) / f"checkpoint_{step:06d}.pt"
    torch.save({"step": step,
                "trainable_state": {k: v.cpu() for k, v in model.named_parameters()
                                    if v.requires_grad},
                "cfg": OmegaConf.to_container(cfg)}, path)
    return path


def main(cfg: DictConfig):
    _seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cwd = Path(os.getcwd())
    out_dir = cwd / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}  emb={cfg.data.emb_root}")

    train_loader = build_loader(cfg.data.emb_root, cfg.data.processed_root,
                                "train", shuffle=True, num_workers=cfg.data.num_workers)
    try:
        val_loader = build_loader(cfg.data.emb_root, cfg.data.processed_root,
                                  "val", shuffle=False, num_workers=1)
    except ValueError:
        val_loader = None
    print(f"train pieces: {len(train_loader.dataset)}  "
          f"val pieces: {len(val_loader.dataset) if val_loader else 0}")

    model_cfg = FullSeqModelConfig(
        d_audio=cfg.model.d_audio, d_image=cfg.model.d_image,
        shared_dim=cfg.model.shared_dim, n_heads=cfg.model.n_heads,
        n_cross_layers=cfg.model.n_cross_layers, dropout=cfg.model.dropout)
    model = FullSeqAlignmentModel(model_cfg).to(device)
    print(f"trainable params: {model.num_trainable_params():,}")

    optim = torch.optim.AdamW(model.trainable_parameters(),
                              lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    accum = cfg.train.get("grad_accum_steps", 1)

    train_iter = iter(train_loader)
    t0 = time.time()
    optim.zero_grad(set_to_none=True)

    for step in range(1, cfg.train.steps + 1):
        acc_loss = 0.0
        acc_dist = 0.0
        acc_ent = 0.0
        for _ in range(accum):
            try:
                b = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                b = next(train_iter)
            b = _to(b, device)
            out = model(b["audio_emb"].unsqueeze(0), b["tile_emb"].unsqueeze(0))
            sim = out["sim"][0]
            loss, parts = expected_distance_loss(
                sim, b["pos_tile"], b["pos_target"], b["valid_mask"],
                temperature=cfg.loss.temperature, power=cfg.loss.power,
                entropy_weight=cfg.loss.get("entropy_weight", 0.0))
            (loss / accum).backward()
            acc_loss += loss.item() / accum
            acc_dist += parts["exp_dist"].item() / accum
            acc_ent += parts["entropy"].item() / accum

        lr = _lr(step, cfg.optim.warmup_steps, cfg.train.steps, cfg.optim.lr)
        for g in optim.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.optim.grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % cfg.train.log_every == 0 or step == 1:
            dt = time.time() - t0
            print(f"step {step:5d}/{cfg.train.steps}  loss={acc_loss:.5f}  "
                  f"exp_dist={acc_dist:.5f}  entropy={acc_ent:.3f}  "
                  f"lr={lr:.2e}  {dt:.1f}s", flush=True)

        if val_loader is not None and step % cfg.train.eval_every == 0:
            v = _validate(model, val_loader, device, cfg)
            print(f"  [val @ {step}] mean exp_dist = {v:.5f}", flush=True)

        if step % cfg.train.ckpt_every == 0:
            print(f"  -> saved {_save(model, optim, step, cfg, out_dir)}", flush=True)

    print(f"done in {time.time()-t0:.1f}s")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/v3_fullseq.yaml")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    return cfg


if __name__ == "__main__":
    main(_parse())
