"""Variant C: end-to-end training with live LoRA encoders + distance-aware loss.

    python -m mymodel.v3_e2e.train --config configs/v3_e2e.yaml

Two-group optimizer:
  - encoder_params (LoRA adapters): lr = cfg.optim.encoder_lr  (~5e-6)
  - head_params (proj + cross-attn): lr = cfg.optim.lr         (~1e-4)
"""
from __future__ import annotations
import argparse, math, os, random, time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .model import E2EAlignmentModel, E2EModelConfig
from .data import build_loaders
from ..shared.losses import expected_distance_loss


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _lr_scale(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def _load_init(model, checkpoint, device):
    sd = torch.load(checkpoint, map_location=device, weights_only=False)
    params = sd.get("trainable_state", sd.get("model_state", {}))
    miss, unexp = model.load_state_dict(params, strict=False)
    print(f"init from {checkpoint}  missing={len(miss)} unexpected={len(unexp)}")


def _save(model, optim, step, cfg, out_dir, filename=None):
    path = Path(out_dir) / (filename or f"checkpoint_{step:06d}.pt")
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
    print(f"device={device}  out={out_dir}", flush=True)

    loaders = build_loaders(
        cfg.data.processed_root,
        split_names=["train", "val"],
        audio_sec=cfg.window.audio_sec,
        audio_sr=cfg.window.audio_sr,
        pool_hz=cfg.window.pool_hz,
        tile_size=cfg.window.tile_size,
        tile_stride=cfg.window.tile_stride,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
    )
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    print(f"train pieces: {len(train_loader.dataset.pieces)}  "
          f"steps/epoch: {len(train_loader)}", flush=True)

    model_cfg = E2EModelConfig(
        audio_model_id=cfg.model.audio_model_id,
        pool_hz=cfg.window.pool_hz,
        lora_rank_audio=cfg.model.lora_rank_audio,
        image_model_id=cfg.model.image_model_id,
        tile_size=cfg.window.tile_size,
        tile_stride=cfg.window.tile_stride,
        lora_rank_image=cfg.model.lora_rank_image,
        shared_dim=cfg.model.shared_dim,
        n_heads=cfg.model.n_heads,
        n_cross_layers=cfg.model.n_cross_layers,
        dropout=cfg.model.dropout,
    )
    model = E2EAlignmentModel(model_cfg).to(device)
    print(f"trainable params: {model.num_trainable_params():,}", flush=True)

    # warm-start: load LoRA encoder weights from NCE run + head from v3 run
    if cfg.train.get("init_encoder_checkpoint"):
        _load_init(model, cwd / cfg.train.init_encoder_checkpoint, device)
    if cfg.train.get("init_head_checkpoint"):
        _load_init(model, cwd / cfg.train.init_head_checkpoint, device)

    # two-group optimizer: very low LR for encoder LoRA, normal for head
    base_lrs = [cfg.optim.encoder_lr, cfg.optim.lr]
    optim = torch.optim.AdamW([
        {"params": list(model.encoder_parameters()),
         "lr": cfg.optim.encoder_lr, "weight_decay": cfg.optim.weight_decay},
        {"params": list(model.head_parameters()),
         "lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay},
    ])
    scaler = torch.amp.GradScaler(device, enabled=cfg.train.amp and device == "cuda")
    accum = cfg.train.get("grad_accum_steps", 1)

    train_iter = iter(train_loader)
    t0 = time.time()
    optim.zero_grad(set_to_none=True)

    # best-checkpoint tracking + early stopping on val loss. e2e LoRA
    # fine-tuning converges slowly (very low encoder LR), so patience is
    # generous by default: cfg.train.eval_every=500 with this patience means
    # ~5000 steps (10 evals) of no improvement before we give up -- enough
    # runway to ride out noisy validation, but well short of the full
    # cfg.train.steps budget on a run that has clearly plateaued/overfit.
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = cfg.train.get("early_stop_patience", 10)

    for step in range(1, cfg.train.steps + 1):
        acc_loss = acc_dist = acc_ent = 0.0

        scale = _lr_scale(step, cfg.optim.warmup_steps, cfg.train.steps)
        for g, base in zip(optim.param_groups, base_lrs):
            g["lr"] = base * scale

        for _ in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            audio = batch["audio_window"].to(device)
            strip = batch["strip"].to(device)
            pos_target = batch["pos_target"].to(device)
            valid = batch["valid_mask"].to(device)

            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                    enabled=cfg.train.amp and device == "cuda"):
                out = model(audio, strip)
                # per-sample loss (variable N per strip width — use first sample's pos_tile)
                # all samples in batch share same strip width after collate padding,
                # but pos_tile is per-sample; use mean across batch for simplicity
                losses = []
                for b in range(audio.shape[0]):
                    N_real = batch["strip"][b].shape[2] // cfg.window.tile_stride  # approx
                    pos_tile_b = batch["pos_tile"][b].to(device)
                    sim_b = out["sim"][b, :, :len(pos_tile_b)]
                    l, parts = expected_distance_loss(
                        sim_b, pos_tile_b, pos_target[b], valid[b],
                        temperature=cfg.loss.temperature,
                        power=cfg.loss.power)
                    losses.append(l)
                    acc_dist += parts["exp_dist"].item() / (accum * audio.shape[0])
                    acc_ent += parts["entropy"].item() / (accum * audio.shape[0])
                loss = torch.stack(losses).mean()

            if torch.isnan(loss):
                optim.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss / accum).backward()
            acc_loss += loss.detach().item() / accum

        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            cfg.optim.grad_clip)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

        if step % cfg.train.log_every == 0 or step == 1:
            enc_lr = optim.param_groups[0]["lr"]
            head_lr = optim.param_groups[1]["lr"]
            print(f"step {step:5d}/{cfg.train.steps}  loss={acc_loss:.5f}  "
                  f"exp_dist={acc_dist:.5f}  entropy={acc_ent:.3f}  "
                  f"enc_lr={enc_lr:.2e}  head_lr={head_lr:.2e}  "
                  f"{time.time()-t0:.1f}s", flush=True)

        if val_loader is not None and step % cfg.train.eval_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_loader:
                    va = vb["audio_window"].to(device)
                    vs = vb["strip"].to(device)
                    vout = model(va, vs)
                    for b in range(va.shape[0]):
                        pt = vb["pos_tile"][b].to(device)
                        sim_b = vout["sim"][b, :, :len(pt)]
                        vl, vp = expected_distance_loss(
                            sim_b, pt, vb["pos_target"][b].to(device),
                            vb["valid_mask"][b].to(device),
                            temperature=cfg.loss.temperature,
                            power=cfg.loss.power)
                        val_losses.append(vp["exp_dist"].item())
            model.train()
            mean_val = float(np.mean(val_losses))
            print(f"  [val @ {step}] mean exp_dist = {mean_val:.5f}", flush=True)

            if mean_val < best_val_loss:
                best_val_loss = mean_val
                patience_counter = 0
                best_path = _save(model, optim, step, cfg, out_dir, filename="best_model.pt")
                print(f"  -> new best val loss {best_val_loss:.5f}, saved {best_path}", flush=True)
            else:
                patience_counter += 1
                print(f"  -> no val improvement (best={best_val_loss:.5f}), "
                      f"patience {patience_counter}/{early_stop_patience}", flush=True)
                if patience_counter >= early_stop_patience:
                    print(f"early stopping at step {step}: no val improvement for "
                          f"{early_stop_patience} consecutive evals "
                          f"(best val loss={best_val_loss:.5f})", flush=True)
                    break

        if step % cfg.train.ckpt_every == 0:
            print(f"  -> saved {_save(model, optim, step, cfg, out_dir)}", flush=True)

    print(f"done in {time.time()-t0:.1f}s")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/v3_e2e.yaml")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    return cfg


if __name__ == "__main__":
    main(_parse())
