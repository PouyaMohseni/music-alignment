"""Variant A training loop.

From the repo root:

    python -m mymodel.v1_baseline.train

Optionally override config knobs on the CLI (Hydra style):

    python -m mymodel.v1_baseline.train train.steps=200 train.batch_size=2
"""
from __future__ import annotations
import math, os, random, time
from pathlib import Path

import argparse
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .data import WindowConfig, build_dataloaders
from .loss import softdtw_anchor_loss
from .model import AlignmentModel, AlignmentModelConfig
from ..shared.losses import infonce_loss


def _pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    # MPS is intentionally skipped: pure-Python SoftDTW backward overflows
    # under MPS with the band-penalty cost. Use cpu on Mac laptops; switch
    # to cuda on a GPU box.
    return "cpu"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warmup_cosine_lr(step: int, *, warmup: int, total: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def _move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _save_checkpoint(model, optim, step, cfg, out_dir):
    path = Path(out_dir) / f"checkpoint_{step:06d}.pt"
    torch.save({
        "step":        step,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()
                        if v.dtype.is_floating_point and v.requires_grad
                        or k.endswith(".weight") or k.endswith(".bias")},
        "trainable_state": {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad},
        "optim_state": optim.state_dict(),
        "cfg":         OmegaConf.to_container(cfg),
    }, path)
    return path


# ---------------------------------------------------------------- train loop ---


def main(cfg: DictConfig) -> None:
    _seed_everything(cfg.seed)
    device = _pick_device(cfg.train.device)
    cwd = Path(os.getcwd())
    out_dir = cwd / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device      = {device}")
    print(f"output dir  = {out_dir}")
    print(f"manifest    = {cfg.data.manifest_path}")

    # ----- data -----
    win = WindowConfig(**OmegaConf.to_container(cfg.window))
    manifest = str(cwd / cfg.data.manifest_path)
    loaders  = build_dataloaders(
        manifest,
        window=win,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
    )
    train_loader = loaders["train"]
    val_loader   = loaders.get("val")
    print(f"train pieces: {len(train_loader.dataset.pieces)}  "
          f"virtual steps/epoch: {len(train_loader)}")
    if val_loader is None:
        print("(no val split)")

    # ----- model -----
    model_cfg = AlignmentModelConfig(
        shared_dim=cfg.model.shared_dim,
        audio_model_id=cfg.model.audio_model_id,
        image_model_id=cfg.model.image_model_id,
        pool_hz=cfg.model.pool_hz,
        tile_size=cfg.model.tile_size,
        tile_stride=cfg.model.tile_stride,
        freeze_audio=cfg.model.freeze_audio,
        freeze_image=cfg.model.freeze_image,
        lora_rank_audio=cfg.model.get("lora_rank_audio", 0),
        lora_rank_image=cfg.model.get("lora_rank_image", 0),
        head_dropout=cfg.model.head_dropout,
    )
    model = AlignmentModel(model_cfg).to(device)
    print(f"trainable params: {model.num_trainable_params():,}")

    # ----- optim -----
    optim = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
    )
    scaler = torch.amp.GradScaler(device, enabled=cfg.train.amp and device == "cuda")

    # ----- train -----
    accum_steps  = cfg.train.get("grad_accum_steps", 1)
    # SoftDTW gate: only add DTW loss once NCE < nce_gate_threshold.
    # Below this value the embedding space has enough structure for DTW to help.
    nce_gate     = cfg.loss.get("nce_gate_threshold", 1.0)
    nce_weight   = cfg.loss.get("nce_weight", 0.5)
    nce_temp     = cfg.loss.get("nce_temperature", 0.07)
    dtw_enabled  = False

    train_iter = iter(train_loader)
    t_start    = time.time()
    optim.zero_grad(set_to_none=True)

    for step in range(1, cfg.train.steps + 1):
        # ---- collect accum_steps micro-batches ----
        accum_loss = torch.tensor(0.0, device=device)
        accum_parts: dict = {"dtw": torch.tensor(0.0), "anchor": torch.tensor(0.0),
                             "nce": torch.tensor(0.0)}
        skipped = 0
        for _ in range(accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = _move_batch(batch, device)

            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                     enabled=cfg.train.amp and device == "cuda"):
                out = model(batch["audio"], batch["image"], batch["image_mask"])

                # InfoNCE — always active
                B = out["audio_embeds"].shape[0]
                t_idx = batch["anchors_t"][:, 0].clamp(0, out["audio_embeds"].shape[1] - 1)
                n_idx = batch["anchors_n"][:, 0].clamp(0, out["image_embeds"].shape[1] - 1)
                a_anc = out["audio_embeds"][torch.arange(B), t_idx]
                s_anc = out["image_embeds"][torch.arange(B), n_idx]
                loss_nce = infonce_loss(a_anc, s_anc, temperature=nce_temp)

                # SoftDTW — only once NCE has fallen below gate threshold
                if dtw_enabled:
                    loss_dtw, dtw_parts = softdtw_anchor_loss(
                        out["sim"],
                        batch["anchors_t"], batch["anchors_n"], batch["anchor_mask"],
                        gamma=cfg.loss.gamma,
                        anchor_weight=cfg.loss.anchor_weight,
                        band_radius_frac=cfg.loss.band_radius_frac,
                    )
                    loss = loss_dtw + nce_weight * loss_nce
                    for k in ("dtw", "anchor"):
                        accum_parts[k] = accum_parts[k] + dtw_parts[k].detach().cpu()
                else:
                    loss = loss_nce

                accum_parts["nce"] = accum_parts["nce"] + loss_nce.detach().cpu()
                micro_loss = loss / accum_steps

            if not torch.isfinite(micro_loss):
                skipped += 1
                continue

            scaler.scale(micro_loss).backward()
            accum_loss = accum_loss + micro_loss.detach()

        if skipped == accum_steps:
            print(f"  WARN step {step}: all micro-batches non-finite, skipping",
                  flush=True)
            optim.zero_grad(set_to_none=True)
            continue

        # Enable DTW once NCE consistently below gate
        nce_val = (accum_parts["nce"] / accum_steps).item()
        if not dtw_enabled and nce_val < nce_gate:
            dtw_enabled = True
            print(f"  INFO step {step}: NCE={nce_val:.4f} < gate={nce_gate} "
                  f"→ SoftDTW enabled", flush=True)

        # set lr from warmup-cosine schedule
        lr = _warmup_cosine_lr(step, warmup=cfg.optim.warmup_steps,
                               total=cfg.train.steps, peak=cfg.optim.lr)
        for g in optim.param_groups:
            g["lr"] = lr

        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.optim.grad_clip)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

        if step % cfg.train.log_every == 0 or step == 1:
            dt = time.time() - t_start
            dtw_str = f"dtw={accum_parts['dtw'].item()/accum_steps:7.4f}  " if dtw_enabled else "dtw=off        "
            print(f"step {step:5d}/{cfg.train.steps}  loss={accum_loss.item():7.4f}  "
                  f"{dtw_str}"
                  f"nce={nce_val:.4f}  "
                  f"lr={lr:.2e}  {dt:.1f}s elapsed")

        if val_loader is not None and step % cfg.train.eval_every == 0:
            _run_window_val(model, val_loader, device, cfg, step)

        if step % cfg.train.ckpt_every == 0:
            path = _save_checkpoint(model, optim, step, cfg, out_dir)
            print(f"  -> saved {path}")

    print(f"done in {time.time() - t_start:.1f}s")


@torch.no_grad()
def _run_window_val(model, val_loader, device, cfg, step):
    model.eval()
    losses = []
    for batch in val_loader:
        batch = _move_batch(batch, device)
        out = model(batch["audio"], batch["image"], batch["image_mask"])
        loss, _ = softdtw_anchor_loss(
            out["sim"], batch["anchors_t"], batch["anchors_n"], batch["anchor_mask"],
            gamma=cfg.loss.gamma, anchor_weight=cfg.loss.anchor_weight,
            band_radius_frac=cfg.loss.band_radius_frac,
        )
        losses.append(loss.item())
    model.train()
    print(f"  [val @ step {step}] window loss = {np.mean(losses):.4f}  "
          f"({len(losses)} batches)")


def _parse_cli() -> DictConfig:
    p = argparse.ArgumentParser(description="Variant A training")
    p.add_argument("--config", default="configs/v1_baseline.yaml",
                   help="path to YAML config (relative to repo root)")
    p.add_argument("overrides", nargs="*",
                   help="dot-key=value overrides, e.g. train.steps=200")
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    return cfg


if __name__ == "__main__":
    main(_parse_cli())
