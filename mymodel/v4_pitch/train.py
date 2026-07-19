"""v4 training: pitch-fused full-sequence alignment.

Loss = sharpened expected_distance_loss (localization)
     + pitch_weight * (BCE(audio_pitch) + BCE(score_pitch))   [aux pitch supervision]

    python -m mymodel.v4_pitch.train --config configs/v4_pitch.yaml
"""
from __future__ import annotations
import argparse, math, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .model import PitchFusedModel, PitchFusedConfig
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


def _to(b, device):
    for k in ("audio_emb", "tile_emb", "pos_tile", "pos_target", "valid_mask",
              "audio_pitch_target", "score_pitch_target"):
        b[k] = b[k].to(device)
    return b


def _losses(model, b, cfg):
    out = model(b["audio_emb"].unsqueeze(0), b["tile_emb"].unsqueeze(0))
    sim = out["sim"][0]
    loc, parts = expected_distance_loss(
        sim, b["pos_tile"], b["pos_target"], b["valid_mask"],
        temperature=cfg.loss.temperature, power=cfg.loss.power,
        entropy_weight=cfg.loss.get("entropy_weight", 0.0))
    bce_a = F.binary_cross_entropy_with_logits(out["audio_pitch_logits"][0], b["audio_pitch_target"])
    bce_s = F.binary_cross_entropy_with_logits(out["score_pitch_logits"][0], b["score_pitch_target"])
    pw = cfg.loss.pitch_weight
    loss = loc + pw * (bce_a + bce_s)
    return loss, {"loc": float(loc.detach()), "exp_dist": float(parts["exp_dist"]),
                  "bce_a": float(bce_a.detach()), "bce_s": float(bce_s.detach())}


@torch.no_grad()
def _validate(model, loader, device, cfg):
    model.eval(); errs = []
    for b in loader:
        b = _to(b, device)
        _, d = _losses(model, b, cfg)
        errs.append(d["exp_dist"])
    model.train()
    return float(np.mean(errs)) if errs else float("nan")


def _save_checkpoint(model, step, cfg, path):
    torch.save({"step": step,
                "trainable_state": {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad},
                "cfg": OmegaConf.to_container(cfg)}, path)
    return path


def _save(model, step, cfg, out_dir):
    return _save_checkpoint(model, step, cfg, Path(out_dir) / f"checkpoint_{step:06d}.pt")


def _load_v3_weights(model, ckpt_path, device):
    """Warm-start proj + cross-attn heads from a v3 checkpoint (pitch heads stay random)."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    v3_state = sd.get("trainable_state", sd.get("model_state", {}))
    transferable = {k: v for k, v in v3_state.items()
                    if k.startswith(("audio_proj.", "image_proj.", "audio_layers.", "image_layers."))}
    miss, _ = model.load_state_dict(transferable, strict=False)
    print(f"  warm-start: loaded {len(transferable)} tensors from {ckpt_path} "
          f"({len(miss)} new v4-only keys left random)", flush=True)


def main(cfg: DictConfig):
    _seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cwd = Path(os.getcwd())
    out_dir = cwd / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}  emb={cfg.data.emb_root}", flush=True)

    tl = build_loader(cfg.data.emb_root, cfg.data.processed_root, "train",
                      shuffle=True, num_workers=cfg.data.num_workers, tile_size=cfg.data.tile_size)
    try:
        vl = build_loader(cfg.data.emb_root, cfg.data.processed_root, "val",
                          shuffle=False, num_workers=1, tile_size=cfg.data.tile_size)
    except ValueError:
        vl = None
    print(f"train pieces: {len(tl.dataset)}  val pieces: {len(vl.dataset) if vl else 0}", flush=True)

    mc = PitchFusedConfig(
        d_audio=cfg.model.d_audio, d_image=cfg.model.d_image, shared_dim=cfg.model.shared_dim,
        n_heads=cfg.model.n_heads, n_cross_layers=cfg.model.n_cross_layers, dropout=cfg.model.dropout,
        pitch_fuse_alpha=cfg.model.pitch_fuse_alpha, pitch_hidden=cfg.model.pitch_hidden)
    model = PitchFusedModel(mc).to(device)

    init_ckpt = cfg.get("init_v3_checkpoint", None)
    if init_ckpt:
        _load_v3_weights(model, init_ckpt, device)

    print(f"trainable params: {model.num_trainable_params():,}", flush=True)

    optim = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    accum = cfg.train.get("grad_accum_steps", 1)
    it = iter(tl); t0 = time.time(); optim.zero_grad(set_to_none=True)

    # Best-checkpoint tracking + early stopping on val exp_dist (lower is better).
    # This model family overfits fast (v4_pitch training log: val exp_dist bottomed
    # out at step ~2500/10000 then got 16% worse by step 10000), so we use a fairly
    # tight patience of 4 validation checks (= 4 * eval_every steps) without
    # improvement before stopping.
    best_val = float("inf")
    patience = 0
    patience_limit = cfg.train.get("early_stop_patience", 4)

    for step in range(1, cfg.train.steps + 1):
        agg = {"loc": 0, "exp_dist": 0, "bce_a": 0, "bce_s": 0}; tot = 0.0
        for _ in range(accum):
            try: b = next(it)
            except StopIteration: it = iter(tl); b = next(it)
            b = _to(b, device)
            loss, d = _losses(model, b, cfg)
            (loss / accum).backward()
            tot += float(loss) / accum
            for k in agg: agg[k] += d[k] / accum
        lr = _lr(step, cfg.optim.warmup_steps, cfg.train.steps, cfg.optim.lr)
        for g in optim.param_groups: g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.optim.grad_clip)
        optim.step(); optim.zero_grad(set_to_none=True)

        if step % cfg.train.log_every == 0 or step == 1:
            print(f"step {step:5d}/{cfg.train.steps}  loss={tot:.4f}  exp_dist={agg['exp_dist']:.4f}  "
                  f"bce_a={agg['bce_a']:.3f}  bce_s={agg['bce_s']:.3f}  lr={lr:.2e}  "
                  f"{time.time()-t0:.1f}s", flush=True)
        if vl is not None and step % cfg.train.eval_every == 0:
            val_exp_dist = _validate(model, vl, device, cfg)
            print(f"  [val @ {step}] mean exp_dist = {val_exp_dist:.5f}", flush=True)
            if val_exp_dist < best_val:
                best_val = val_exp_dist
                patience = 0
                best_path = _save_checkpoint(model, step, cfg, Path(out_dir) / "best_model.pt")
                print(f"  -> new best (exp_dist={best_val:.5f}), saved {best_path}", flush=True)
            else:
                patience += 1
                print(f"  -> no improvement ({patience}/{patience_limit}); best exp_dist={best_val:.5f}",
                      flush=True)
                if patience >= patience_limit:
                    print(f"early stopping at step {step}: no val exp_dist improvement in "
                          f"{patience_limit} checks ({patience_limit * cfg.train.eval_every} steps); "
                          f"best={best_val:.5f}", flush=True)
                    if step % cfg.train.ckpt_every != 0:
                        print(f"  -> saved {_save(model, step, cfg, out_dir)}", flush=True)
                    break
        if step % cfg.train.ckpt_every == 0:
            print(f"  -> saved {_save(model, step, cfg, out_dir)}", flush=True)

    print(f"done in {time.time()-t0:.1f}s", flush=True)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/v4_pitch.yaml")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    return cfg


if __name__ == "__main__":
    main(_parse())
