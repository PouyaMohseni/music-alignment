"""v13/v14/v15 eval — full-strip inference with MERT audio features.

Mirrors v11/eval.py but loads pre-computed MERT features from mert_emb/
instead of computing mel spectrograms on-the-fly.

    python -m mymodel.v13_mert_unet.eval \
        --checkpoint /scratch/pmohseni/results/v13_mert_linear/best_model.pt \
        --config     configs/v13_mert_linear.yaml \
        --split      test
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.shared.metrics import alignment_metrics, henkel_metrics


def _build_perf_frame(feats: np.ndarray, t: int, n_frames: int) -> np.ndarray:
    """Single-frame perf window from MERT features.
    feats: (T, 768). Returns (1, 1, 768, n_frames).
    """
    T, mert_dim = feats.shape
    t = min(t, T - 1)
    t0 = max(0, t - n_frames + 1)
    window = feats[t0:t + 1]
    if window.shape[0] < n_frames:
        pad = np.zeros((n_frames - window.shape[0], mert_dim), dtype=np.float32)
        window = np.concatenate([pad, window], axis=0)
    return window.T[np.newaxis, np.newaxis]   # (1, 1, mert_dim, n_frames)


def _predict_x_com(seg: torch.Tensor, threshold: float = 0.5) -> float:
    """Center of mass along x axis of the thresholded segmentation map.
    seg: (1, H, W). Matches CPJKU's eval_official.py: threshold the raw
    sigmoid output before CoM, else diffuse low-confidence activation
    elsewhere on the strip drags the CoM off target.
    """
    arr = seg.squeeze(0).cpu().numpy()
    arr = (arr >= threshold).astype(np.float32)
    col = arr.sum(axis=0)
    total = col.sum()
    if total < 1e-6:
        return float(arr.shape[1] // 2)
    xs = np.arange(arr.shape[1], dtype=np.float32)
    return float((xs * col).sum() / total)


@torch.no_grad()
def eval_split(checkpoint: str, cfg_path: str, processed_root: str,
               mert_emb_root: str, split: str,
               out_dir: str = None, limit: int = None,
               device: str = None) -> dict | None:
    cfg    = OmegaConf.load(cfg_path)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    net_config = OmegaConf.to_container(cfg.net)
    network = ConditionalUNet(net_config)
    sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
    network.load_state_dict(sd['state_dict'], strict=True)
    network = network.to(device).eval()

    fps      = cfg.data.fps
    n_frames = cfg.data.n_frames
    h_strip  = cfg.data.h_strip
    w_scale  = cfg.data.w_scale

    from mymodel.v11_cpjku_fullstrip.data import load_strip_scaled

    proc = Path(processed_root)
    emb  = Path(mert_emb_root)
    piece_ids = json.load(open(proc / 'splits.json'))[split]
    if limit:
        piece_ids = piece_ids[:limit]

    out_root = Path(out_dir or str(Path(checkpoint).parent / 'eval')) / split
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out_root / 'per_piece.jsonl', 'w') as fout:
        for k, pid in enumerate(piece_ids):
            try:
                piece_dir = proc / pid
                ann   = json.load(open(piece_dir / 'annotations.json'))
                notes = np.load(piece_dir / 'noteheads.npz')

                strip_w    = ann['image']['width_px']
                dur        = float(ann['audio']['duration_sec'])
                px_per_sec = strip_w / dur

                # Load pre-computed MERT features
                feats = np.load(str(emb / f'{pid}.npy')).astype(np.float32)  # (T, 768)
                T = feats.shape[0]

                score = load_strip_scaled(piece_dir / 'strip.png', h_strip, w_scale)
                H, W_sc = score.shape

                score_1 = torch.from_numpy(
                    score[np.newaxis, np.newaxis, np.newaxis]).to(device)  # (1,1,1,H,W_sc)

                hidden = None
                if network.use_lstm:
                    hidden = (torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
                              torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device))

                pred_x_sc = np.zeros(T, dtype=np.float64)
                for t in range(T):
                    perf_np = _build_perf_frame(feats, t, n_frames)
                    perf_t  = torch.from_numpy(perf_np).to(device).unsqueeze(0)  # (1,1,1,768,n)
                    out = network(score=score_1, perf=perf_t, hidden=hidden)
                    seg = out['segmentation']
                    hidden = out.get('hidden')
                    if hidden is not None:
                        hidden = (hidden[0].detach(), hidden[1].detach())
                    pred_x_sc[t] = _predict_x_com(seg.squeeze(0))

                pred_x_orig = pred_x_sc * w_scale
                gt_onset    = notes['onset_sec']
                gt_strip_x  = notes['strip_x']
                frame_idx   = np.clip(np.round(gt_onset * fps).astype(int), 0, T - 1)
                pred_at_onset = pred_x_orig[frame_idx]

                m = alignment_metrics(
                    pred_at_onset, gt_strip_x, px_per_sec,
                    gt_onset_sec=gt_onset,
                    all_strip_x=gt_strip_x,
                    all_onset_sec=gt_onset)
                m.update(henkel_metrics(pred_at_onset, gt_strip_x))
                m['piece_id'] = pid

            except Exception as e:
                import traceback
                m = {'piece_id': pid, 'error': repr(e), 'tb': traceback.format_exc()}

            rows.append(m)
            fout.write(json.dumps(m) + '\n')
            fout.flush()
            if (k + 1) % 10 == 0:
                good = [r for r in rows if 'error' not in r]
                if good:
                    pct = np.mean([r.get('pct_within_0.5s', 0) for r in good])
                    print(f'  [{k+1}/{len(piece_ids)}]  pct@0.5s={pct:.1f}%', flush=True)

    good = [r for r in rows if 'error' not in r]
    if not good:
        for r in [x for x in rows if 'error' in x][:3]:
            print(f'  ERROR {r["piece_id"]}: {r["error"]}')
        return None

    keys = [k for k in good[0]
            if k.startswith(('mean_', 'median_', 'pct_', 'n'))]
    summ = {'n_pieces': len(good), 'n_errors': len(rows) - len(good)}
    for k in keys:
        vals = np.asarray([r[k] for r in good if isinstance(r.get(k), (int, float))])
        if len(vals):
            summ[f'mean_{k}'] = float(vals.mean())

    summ['split'] = split
    summ['checkpoint'] = str(checkpoint)

    with open(out_root / 'summary.json', 'w') as f:
        json.dump(summ, f, indent=2)

    print(f'\n=== {split} results ===')
    for thr in [0.05, 0.1, 0.5, 1.0, 5.0]:
        k = f'mean_pct_within_{thr}s'
        if k in summ:
            print(f'  pct@{thr}s  = {summ[k]:.1f}%')
    print(f'  mean_err  = {summ.get("mean_mean_abs_err_sec", float("nan")):.3f}s')
    print(f'  median_err= {summ.get("mean_median_abs_err_sec", float("nan")):.3f}s')
    print(f'  n_pieces  = {summ["n_pieces"]}')
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    required=True)
    p.add_argument('--config',        required=True)
    p.add_argument('--split',         default='test', choices=['train', 'val', 'test'])
    p.add_argument('--processed',     default='data/MSMD/processed')
    p.add_argument('--mert_emb_root', default='data/MSMD/mert_emb')
    p.add_argument('--out_dir',       default=None)
    p.add_argument('--limit',         type=int, default=None)
    p.add_argument('--device',        default=None)
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.mert_emb_root,
               a.split, out_dir=a.out_dir, limit=a.limit, device=a.device)


if __name__ == '__main__':
    main()
