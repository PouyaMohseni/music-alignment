"""v10 eval — causal tracking using precomputed MERT embeddings.

Same causal LSTM tracking as v9; only audio source changes:
  spectrogram (78-bin log-mel) → precomputed MERT embeddings (768-dim at 20 Hz).

    python -m mymodel.v10_mert_unet.eval \
        --checkpoint results/v10_mert_unet/checkpoint_050000.pt \
        --config configs/v10_mert_unet.yaml \
        --split test \
        --processed /path/to/data/MSMD/processed \
        --mert_emb  /path/to/data/MSMD/mert_emb
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .network import ConditionalUNet
from ..v9_cpjku.data import load_strip_2d, crop_score
from ..shared.metrics import alignment_metrics, henkel_metrics


def _build(cfg, device):
    net_config = OmegaConf.to_container(cfg.net)
    return ConditionalUNet(net_config).to(device)


def _predict_x(seg: torch.Tensor) -> float:
    arr = seg.squeeze(0).cpu().numpy()
    if arr.sum() < 1e-6:
        return arr.shape[1] / 2.0
    col_weights = arr.sum(axis=0)
    xs = np.arange(arr.shape[1], dtype=np.float32)
    return float((xs * col_weights).sum() / col_weights.sum())


@torch.no_grad()
def eval_split(checkpoint, cfg_path, processed_root, mert_emb_root, split,
               out_dir=None, limit=None, device=None):
    cfg    = OmegaConf.load(cfg_path)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    network = _build(cfg, device)
    sd = torch.load(checkpoint, map_location=device, weights_only=False)
    network.load_state_dict(sd['state_dict'], strict=True)
    network.eval()

    fps = cfg.data.fps   # 20
    W   = cfg.data.tile_width
    H   = cfg.data.h_strip

    proc     = Path(processed_root)
    emb_root = Path(mert_emb_root)
    piece_ids = json.load(open(proc / 'splits.json'))[split]
    if limit:
        piece_ids = piece_ids[:limit]

    out_root = Path(out_dir or str(Path(checkpoint).parent / 'eval')) / split
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out_root / 'per_piece.jsonl', 'w') as fout:
        for k, pid in enumerate(piece_ids):
            try:
                emb_path = emb_root / f'{pid}.npy'
                if not emb_path.exists():
                    raise FileNotFoundError(f'No MERT embedding: {emb_path}')

                piece_dir = proc / pid
                ann   = json.load(open(piece_dir / 'annotations.json'))
                notes = np.load(piece_dir / 'noteheads.npz')
                strip_w   = ann['image']['width_px']
                dur       = float(ann['audio']['duration_sec'])
                px_per_sec = strip_w / dur

                mert_emb = np.load(emb_path).astype(np.float32)   # (T, 768)
                strip    = load_strip_2d(piece_dir / 'strip.png', H)

                T_spec = mert_emb.shape[0]

                hidden = None
                if network.use_lstm:
                    hidden = (torch.zeros(network.rnn_layers, 1, network.rnn_size).to(device),
                              torch.zeros(network.rnn_layers, 1, network.rnn_size).to(device))

                pos_estimate = 0.0
                pred_x_at_frame = np.zeros(T_spec, dtype=np.float64)

                for t in range(T_spec):
                    # MERT embedding at frame t → (1, 1, 1, 768, 1)
                    perf_t = torch.from_numpy(
                        mert_emb[t].reshape(1, 1, 1, 768, 1)
                    ).to(device)

                    cx   = int(round(pos_estimate))
                    crop = crop_score(strip, cx, W)
                    x0   = max(0, min(strip_w - W, cx - W // 2))

                    score_t = torch.from_numpy(
                        crop[np.newaxis, np.newaxis, np.newaxis]
                    ).to(device)   # (1, 1, 1, H, W)

                    out    = network(score=score_t, perf=perf_t, hidden=hidden)
                    seg    = out['segmentation']
                    hidden = out['hidden']
                    if hidden is not None:
                        hidden = (hidden[0].detach(), hidden[1].detach())

                    local_x = _predict_x(seg.squeeze(0))
                    new_pos = float(np.clip(x0 + local_x, 0, strip_w - 1))
                    pos_estimate = new_pos
                    pred_x_at_frame[t] = new_pos

                gt_onset   = notes['onset_sec']
                gt_strip_x = notes['strip_x']
                frame = np.clip(np.round(gt_onset * fps).astype(int), 0, T_spec - 1)
                pred_at_onset = pred_x_at_frame[frame]

                m = alignment_metrics(
                    pred_at_onset, gt_strip_x, px_per_sec,
                    beat_times_sec=ann.get('beat_times_sec') or None,
                    bar_times_sec=ann.get('bar_times_sec') or None,
                    gt_onset_sec=gt_onset,
                    all_strip_x=gt_strip_x,
                    all_onset_sec=gt_onset)
                m.update(henkel_metrics(pred_at_onset, gt_strip_x))
                m['piece_id'] = pid
            except Exception as e:
                import traceback
                m = {'piece_id': pid, 'error': repr(e), 'tb': traceback.format_exc()}

            rows.append(m)
            fout.write(json.dumps(m) + '\n'); fout.flush()
            if (k + 1) % 10 == 0:
                good = [r for r in rows if 'error' not in r]
                if good:
                    print(f'  [{k+1}/{len(piece_ids)}] mean_abs_err_sec='
                          f'{np.mean([r["mean_abs_err_sec"] for r in good]):.3f}',
                          flush=True)

    good = [r for r in rows if 'error' not in r]
    if not good:
        for r in [x for x in rows if 'error' in x][:3]:
            print(f'  {r["piece_id"]}: {r["error"]}\n{r.get("tb","")}')
        return None

    keys = [k for k in good[0]
            if k.startswith(('mean_', 'median_', 'pct_', 'recall_')) or k == 'n']
    summ = {'n_pieces': len(good), 'n_errors': len(rows) - len(good)}
    for k in keys:
        vals = np.asarray([r[k] for r in good if isinstance(r.get(k), (int, float))])
        if len(vals):
            summ[f'mean_{k}'] = float(vals.mean())
            summ[f'median_{k}'] = float(np.median(vals))
    summ['split'] = split; summ['checkpoint'] = checkpoint
    with open(out_root / 'summary.json', 'w') as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    p.add_argument('--config',     default='configs/v10_mert_unet.yaml')
    p.add_argument('--processed',  default='data/MSMD/processed')
    p.add_argument('--mert_emb',   default='data/MSMD/mert_emb')
    p.add_argument('--out_dir',    default=None)
    p.add_argument('--limit',      type=int, default=None)
    p.add_argument('--device',     default=None)
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.mert_emb, a.split,
               out_dir=a.out_dir, limit=a.limit, device=a.device)


if __name__ == '__main__':
    main()
