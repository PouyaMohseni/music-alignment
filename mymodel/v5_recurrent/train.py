"""v5 training: recurrent score follower.

Loss: cross-entropy(logits[t], nearest_tile[t]) per valid audio frame.
Warm-start proj + cross-attn from a v3_all checkpoint.

    python -m mymodel.v5_recurrent.train --config configs/v5_recurrent.yaml
"""
from __future__ import annotations
import argparse, math, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .model import RecurrentFollower, RecurrentConfig
from ..v3_fullseq.data import FullSeqTarDataset, FullSeqDataset
from ..v4_pitch.data import PitchFusedDataset


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _lr(step, warmup, total, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * prog))


def _build_loader(emb_root, processed_root, split, shuffle, num_workers, with_pitch=False):
    if with_pitch:
        ds = PitchFusedDataset(emb_root, processed_root, split)
    elif (Path(emb_root) / "index.json").exists():
        ds = FullSeqTarDataset(emb_root, processed_root, split)
    else:
        ds = FullSeqDataset(emb_root, processed_root, split)
    return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=lambda b: b[0], persistent_workers=num_workers > 0)


def _to(b, device):
    keys = ["audio_emb", "tile_emb", "pos_tile", "pos_target", "valid_mask"]
    for k in ("audio_pitch_target", "score_pitch_target"):
        if k in b:
            keys.append(k)
    for k in keys:
        b[k] = b[k].to(device)
    return b


def _tile_labels(pos_target, pos_tile):
    """Nearest tile index for each audio frame."""
    diffs = (pos_target.unsqueeze(1) - pos_tile.unsqueeze(0)).abs()  # (T, N)
    return diffs.argmin(dim=1)                                        # (T,) long


def _step_loss(model, b, pitch_weight=0.0):
    out = model(b["audio_emb"].unsqueeze(0), b["tile_emb"].unsqueeze(0))
    logits = out["logits"][0]                              # (T, N)
    labels = _tile_labels(b["pos_target"], b["pos_tile"])  # (T,)
    mask = b["valid_mask"].bool()
    loss = F.cross_entropy(logits[mask], labels[mask])
    if pitch_weight > 0 and "audio_pitch_logits" in out:
        bce_a = F.binary_cross_entropy_with_logits(out["audio_pitch_logits"][0], b["audio_pitch_target"])
        bce_s = F.binary_cross_entropy_with_logits(out["score_pitch_logits"][0], b["score_pitch_target"])
        loss = loss + pitch_weight * (bce_a + bce_s)
    acc = (logits[mask].argmax(1) == labels[mask]).float().mean()
    return loss, float(loss.detach()), float(acc.detach())


def _load_v3_weights(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    v3_state = sd.get("trainable_state", sd.get("model_state", {}))
    transferable = {k: v for k, v in v3_state.items()
                    if k.startswith(("audio_proj.", "image_proj.",
                                     "audio_layers.", "image_layers."))}
    miss, _ = model.load_state_dict(transferable, strict=False)
    print(f"  warm-start: loaded {len(transferable)} tensors "
          f"({len(miss)} new keys left random)", flush=True)


def _save(model, step, cfg, out_dir):
    path = Path(out_dir) / f"checkpoint_{step:06d}.pt"
    torch.save({"step": step,
                "trainable_state": {k: v.cpu() for k, v in model.named_parameters()
                                    if v.requires_grad},
                "cfg": OmegaConf.to_container(cfg)}, path)
    return path


@torch.no_grad()
def _validate(model, loader, device):
    model.eval(); losses = []
    for b in loader:
        b = _to(b, device)
        _, ce, _ = _step_loss(model, b)
        losses.append(ce)
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main(cfg: DictConfig):
    _seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(os.getcwd()) / cfg.train.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}  emb={cfg.data.emb_root}", flush=True)

    with_pitch = cfg.model.get("pitch_hidden", 0) > 0
    tl = _build_loader(cfg.data.emb_root, cfg.data.processed_root, "train",
                       shuffle=True, num_workers=cfg.data.num_workers, with_pitch=with_pitch)
    try:
        vl = _build_loader(cfg.data.emb_root, cfg.data.processed_root, "val",
                           shuffle=False, num_workers=1, with_pitch=with_pitch)
    except ValueError:
        vl = None
    print(f"train: {len(tl.dataset)}  val: {len(vl.dataset) if vl else 0}", flush=True)

    rc = RecurrentConfig(
        d_audio=cfg.model.d_audio, d_image=cfg.model.d_image,
        shared_dim=cfg.model.shared_dim, n_heads=cfg.model.n_heads,
        n_cross_layers=cfg.model.n_cross_layers, dropout=cfg.model.dropout,
        lstm_hidden=cfg.model.lstm_hidden, lstm_layers=cfg.model.lstm_layers,
        lstm_bidirectional=cfg.model.get("lstm_bidirectional", False),
        residual=cfg.model.get("residual", False),
        pitch_hidden=cfg.model.get("pitch_hidden", 0),
        pitch_on_aligned=cfg.model.get("pitch_on_aligned", False))
    model = RecurrentFollower(rc).to(device)

    if cfg.get("init_v3_checkpoint"):
        _load_v3_weights(model, cfg.init_v3_checkpoint, device)

    print(f"trainable params: {model.num_trainable_params():,}", flush=True)

    optim = torch.optim.AdamW(model.trainable_parameters(),
                               lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    accum = cfg.train.get("grad_accum_steps", 1)
    pitch_weight = cfg.loss.get("pitch_weight", 0.0) if hasattr(cfg, "loss") else 0.0
    it = iter(tl); t0 = time.time(); optim.zero_grad(set_to_none=True)

    for step in range(1, cfg.train.steps + 1):
        tot_ce = 0.0; tot_acc = 0.0
        for _ in range(accum):
            try: b = next(it)
            except StopIteration: it = iter(tl); b = next(it)
            b = _to(b, device)
            loss, ce, acc = _step_loss(model, b, pitch_weight=pitch_weight)
            (loss / accum).backward()
            tot_ce += ce / accum; tot_acc += acc / accum

        lr = _lr(step, cfg.optim.warmup_steps, cfg.train.steps, cfg.optim.lr)
        for g in optim.param_groups: g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.optim.grad_clip)
        optim.step(); optim.zero_grad(set_to_none=True)

        if step % cfg.train.log_every == 0 or step == 1:
            print(f"step {step:5d}/{cfg.train.steps}  ce={tot_ce:.4f}  "
                  f"frame_acc={tot_acc:.3f}  lr={lr:.2e}  {time.time()-t0:.1f}s",
                  flush=True)
        if vl and step % cfg.train.eval_every == 0:
            print(f"  [val @ {step}] ce={_validate(model, vl, device):.5f}", flush=True)
        if step % cfg.train.ckpt_every == 0:
            print(f"  -> saved {_save(model, step, cfg, out_dir)}", flush=True)

    print(f"done in {time.time()-t0:.1f}s", flush=True)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/v5_recurrent.yaml")
    p.add_argument("overrides", nargs="*")
    a = p.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(a.overrides))
    return cfg


if __name__ == "__main__":
    main(_parse())
