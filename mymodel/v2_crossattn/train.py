"""v2 training entry point.

Identical to v1_baseline/train.py except it instantiates CrossAttnAlignmentModel
instead of AlignmentModel. Run from repo root:

    python -m mymodel.v2_crossattn.train --config configs/v2_crossattn.yaml
"""
from __future__ import annotations
import sys

# Patch the v1 train module to use v2 model before importing main
import mymodel.v1_baseline.train as _v1_train
from .model import CrossAttnAlignmentModel, CrossAttnModelConfig

_orig_main = _v1_train.main


def _v2_main(cfg):
    # Monkey-patch model construction inside v1 main by overriding the
    # AlignmentModel / AlignmentModelConfig names in v1_train's namespace.
    _v1_train.AlignmentModelConfig = CrossAttnModelConfig
    _v1_train.AlignmentModel = CrossAttnAlignmentModel

    def _build_cfg(cfg_obj):
        return CrossAttnModelConfig(
            shared_dim=cfg_obj.model.shared_dim,
            n_heads=cfg_obj.model.get("n_heads", 4),
            attn_dropout=cfg_obj.model.get("attn_dropout", 0.1),
            head_dropout=cfg_obj.model.head_dropout,
            audio_model_id=cfg_obj.model.audio_model_id,
            pool_hz=cfg_obj.model.pool_hz,
            freeze_audio=cfg_obj.model.freeze_audio,
            lora_rank_audio=cfg_obj.model.get("lora_rank_audio", 0),
            image_model_id=cfg_obj.model.image_model_id,
            tile_size=cfg_obj.model.tile_size,
            tile_stride=cfg_obj.model.tile_stride,
            freeze_image=cfg_obj.model.freeze_image,
            lora_rank_image=cfg_obj.model.get("lora_rank_image", 0),
        )

    # Override the AlignmentModelConfig construction block in v1 main
    import types, functools

    original_main_code = _orig_main.__code__
    _orig_main(cfg)  # will use patched names above


# Actually the cleanest approach: just copy v1 main and replace model instantiation.
# Import everything from v1 and override just the model parts.
from mymodel.v1_baseline.train import (
    _pick_device, _seed_everything, _warmup_cosine_lr,
    _move_batch, _save_checkpoint, _run_window_val, _parse_cli,
)
from mymodel.v1_baseline.data import WindowConfig, build_dataloaders
from mymodel.v1_baseline.loss import softdtw_anchor_loss
import math, os, time
from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F


def _infonce(a: torch.Tensor, i: torch.Tensor, temp: float) -> torch.Tensor:
    """InfoNCE over mean-pooled audio/image embeddings."""
    a_mean = F.normalize(a.mean(dim=1), dim=-1)   # (B, d)
    i_mean = F.normalize(i.mean(dim=1), dim=-1)   # (B, d)
    logits = a_mean @ i_mean.T / temp
    labels = torch.arange(len(a_mean), device=a_mean.device)
    return F.cross_entropy(logits, labels)


def main(cfg: DictConfig) -> None:
    _seed_everything(cfg.seed)
    device = _pick_device(cfg.train.device)
    cwd = Path(os.getcwd())
    out_dir = cwd / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device      = {device}")
    print(f"output dir  = {out_dir}")
    print(f"manifest    = {cfg.data.manifest_path}")

    win = WindowConfig(**OmegaConf.to_container(cfg.window))
    manifest = str(cwd / cfg.data.manifest_path)
    loaders = build_dataloaders(
        manifest, window=win,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.seed,
    )
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    print(f"train pieces: {len(train_loader.dataset.pieces)}  "
          f"virtual steps/epoch: {len(train_loader)}")

    model_cfg = CrossAttnModelConfig(
        shared_dim=cfg.model.shared_dim,
        n_heads=cfg.model.get("n_heads", 4),
        attn_dropout=cfg.model.get("attn_dropout", 0.1),
        head_dropout=cfg.model.head_dropout,
        audio_model_id=cfg.model.audio_model_id,
        pool_hz=cfg.model.pool_hz,
        freeze_audio=cfg.model.freeze_audio,
        lora_rank_audio=cfg.model.get("lora_rank_audio", 0),
        image_model_id=cfg.model.image_model_id,
        tile_size=cfg.model.tile_size,
        tile_stride=cfg.model.tile_stride,
        freeze_image=cfg.model.freeze_image,
        lora_rank_image=cfg.model.get("lora_rank_image", 0),
    )
    model = CrossAttnAlignmentModel(model_cfg).to(device)
    print(f"trainable params: {model.num_trainable_params():,}")

    init_ckpt = cfg.train.get("init_checkpoint", None)
    if init_ckpt:
        init_path = cwd / init_ckpt
        sd = torch.load(init_path, map_location=device, weights_only=False)
        params = sd.get("trainable_state", sd.get("model_state", {}))
        missing, unexpected = model.load_state_dict(params, strict=False)
        print(f"init from {init_ckpt}  missing={len(missing)} unexpected={len(unexpected)}")

    optim = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
    )
    scaler = torch.amp.GradScaler(device, enabled=cfg.train.amp and device == "cuda")

    accum_steps = cfg.train.get("grad_accum_steps", 1)
    nce_gate = cfg.loss.get("nce_gate_threshold", 1.0)
    nce_weight = cfg.loss.get("nce_weight", 0.5)
    nce_temp = cfg.loss.get("nce_temperature", 0.07)
    dtw_ramp_steps = cfg.loss.get("dtw_ramp_steps", 2000)
    dtw_only_nce = cfg.loss.get("nce_only", False)
    dtw_enabled = False
    dtw_enabled_step = 0

    train_iter = iter(train_loader)
    t_start = time.time()
    optim.zero_grad(set_to_none=True)

    for step in range(1, cfg.train.steps + 1):
        accum_loss = torch.tensor(0.0, device=device)
        accum_parts = {"dtw": torch.tensor(0.0), "anchor": torch.tensor(0.0),
                       "nce": torch.tensor(0.0)}
        skipped = 0

        for _ in range(accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = _move_batch(batch, device)

            lr = _warmup_cosine_lr(step, warmup=cfg.optim.warmup_steps,
                                   total=cfg.train.steps, peak=cfg.optim.lr)
            for g in optim.param_groups:
                g["lr"] = lr

            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                    enabled=cfg.train.amp and device == "cuda"):
                out = model(batch["audio"], batch["image"], batch["image_mask"])
                sim = out["sim"]

                # NCE loss on mean-pooled embeddings
                L_nce = _infonce(out["audio_embeds"], out["image_embeds"], nce_temp)

                # check gate
                if not dtw_only_nce and not dtw_enabled and L_nce.item() < nce_gate:
                    dtw_enabled = True
                    dtw_enabled_step = step
                    print(f"  [step {step}] DTW enabled (nce={L_nce.item():.4f} < gate={nce_gate})")

                if dtw_enabled:
                    ramp = min(1.0, (step - dtw_enabled_step) / max(1, dtw_ramp_steps))
                    L_dtw, parts = softdtw_anchor_loss(
                        sim, batch["anchors_t"], batch["anchors_n"], batch["anchor_mask"],
                        gamma=cfg.loss.gamma,
                        anchor_weight=cfg.loss.anchor_weight,
                        band_radius_frac=cfg.loss.band_radius_frac,
                    )
                    loss = ramp * L_dtw + nce_weight * L_nce
                    accum_parts["dtw"] = accum_parts["dtw"] + parts["dtw"].cpu() / accum_steps
                    accum_parts["anchor"] = accum_parts["anchor"] + parts["anchor"].cpu() / accum_steps
                else:
                    loss = L_nce
                    L_dtw = torch.tensor(0.0)
                accum_parts["nce"] = accum_parts["nce"] + L_nce.detach().cpu() / accum_steps

            if torch.isnan(loss):
                skipped += 1
                continue

            scaler.scale(loss / accum_steps).backward()
            accum_loss = accum_loss + loss.detach() / accum_steps

        if skipped == accum_steps:
            optim.zero_grad(set_to_none=True)
            continue

        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.optim.grad_clip)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

        dtw_str = f"{accum_parts['dtw'].item():7.4f}" if dtw_enabled else "off"
        if step % cfg.train.log_every == 0 or step == 1:
            dt = time.time() - t_start
            print(f"step {step:5d}/{cfg.train.steps}  loss={accum_loss.item():7.4f}  "
                  f"dtw={dtw_str:>12s}  nce={accum_parts['nce'].item():.4f}  "
                  f"lr={lr:.2e}  {dt:.1f}s elapsed")

        if val_loader is not None and step % cfg.train.eval_every == 0:
            _run_window_val(model, val_loader, device, cfg, step)

        if step % cfg.train.ckpt_every == 0:
            path = _save_checkpoint(model, optim, step, cfg, out_dir)
            print(f"  -> saved {path}")

    print(f"done in {time.time() - t_start:.1f}s")


def _parse_cli():
    import argparse
    p = argparse.ArgumentParser(description="v2 cross-attention training")
    p.add_argument("--config", default="configs/v2_crossattn.yaml")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    return cfg


if __name__ == "__main__":
    main(_parse_cli())
