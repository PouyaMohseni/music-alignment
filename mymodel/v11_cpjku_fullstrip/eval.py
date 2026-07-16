"""v11 eval — full strip inference with LSTM state.

Unlike v9, the model sees the ENTIRE score strip per frame.
The predicted x position comes directly from the CoM of the output heatmap
(no crop tracking, no re-initialisation needed).

    python -m mymodel.v11_cpjku_fullstrip.eval \
        --checkpoint results/v11_cpjku_fullstrip/best_model.pt \
        --config     configs/v11_cpjku_fullstrip.yaml \
        --split      test \
        --processed  data/MSMD/processed
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.v11_cpjku_fullstrip.data import load_spec, load_spec_madmom, load_strip_scaled
from mymodel.shared.metrics import alignment_metrics, henkel_metrics


def _load_spec_for_eval(piece_dir: Path, pid: str, fps: int, n_mels: int,
                        cpjku_fmt_root: str | None) -> np.ndarray:
    if cpjku_fmt_root:
        spec = load_spec_madmom(pid, Path(cpjku_fmt_root))
        if spec is not None:
            return spec
        print(f'WARNING: no cached real-madmom spectrogram for {pid} '
              f'-- falling back to mel-spectrogram approximation.', flush=True)
    return load_spec(piece_dir / 'audio.wav', fps=fps, n_mels=n_mels)


def _build_perf_frame(spec: np.ndarray, t: int, n_frames: int) -> np.ndarray:
    """Context window for one frame. Returns (1, 1, n_mels, n_frames)."""
    t0 = max(0, t - n_frames)
    window = spec[:, t0:t]
    if window.shape[-1] < n_frames:
        window = np.pad(window, ((0, 0), (n_frames - window.shape[-1], 0)))
    return window[np.newaxis, np.newaxis]  # (1, 1, n_mels, n_frames)


def _predict_x_com(seg: torch.Tensor, threshold: float = 0.5) -> float:
    """Center of mass along the x axis of the thresholded segmentation map.
    seg: (1, H, W) in [0, 1]. Matches CPJKU's eval_official.py: threshold
    the raw sigmoid output before computing center-of-mass, else diffuse
    low-confidence activation elsewhere on the strip drags the CoM off target.
    """
    arr = seg.squeeze(0).cpu().numpy()   # (H, W)
    arr = (arr >= threshold).astype(np.float32)
    col = arr.sum(axis=0)               # (W,)
    total = col.sum()
    if total < 1e-6:
        return float(arr.shape[1] // 2)
    xs = np.arange(arr.shape[1], dtype=np.float32)
    return float((xs * col).sum() / total)


@torch.no_grad()
def eval_split(checkpoint: str, cfg_path: str, processed_root: str, split: str,
               out_dir: str = None, limit: int = None, device: str = None,
               cpjku_fmt_root: str = None) -> dict | None:
    cfg    = OmegaConf.load(cfg_path)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    net_config = OmegaConf.to_container(cfg.net)
    network = ConditionalUNet(net_config)
    sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
    network.load_state_dict(sd['state_dict'], strict=True)
    network = network.to(device)
    network.eval()

    fps      = cfg.data.fps
    n_frames = cfg.data.n_frames
    h_strip  = cfg.data.h_strip
    w_scale  = cfg.data.w_scale
    n_mels   = cfg.data.n_mels
    cpjku_fmt_root = cpjku_fmt_root or cfg.data.get('cpjku_fmt_root', None)

    proc = Path(processed_root)
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

                spec  = _load_spec_for_eval(piece_dir, pid, fps, n_mels, cpjku_fmt_root)
                score = load_strip_scaled(piece_dir / 'strip.png', h_strip, w_scale)
                H, W_sc = score.shape
                T = spec.shape[-1]

                # Score tensor: (1, 1, H, W_sc) — same every frame
                score_t = torch.from_numpy(score[np.newaxis, np.newaxis]).to(device)
                # Expand to (sl=1, bs=1, 1, H, W_sc)
                score_1 = score_t.unsqueeze(0)

                # Initialise LSTM hidden state
                hidden = None
                if network.use_lstm:
                    hidden = (torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
                              torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device))

                # Predicted x in scaled-strip coordinates per frame
                pred_x_sc = np.zeros(T, dtype=np.float64)

                for t in range(T):
                    perf_np = _build_perf_frame(spec, t, n_frames)
                    perf_t  = torch.from_numpy(perf_np).to(device)  # (1, 1, n_mels, n_frames)
                    perf_t  = perf_t.unsqueeze(0)  # (1, 1, 1, n_mels, n_frames) → seq=1, bs=1, c=1

                    out    = network(score=score_1, perf=perf_t, hidden=hidden)
                    seg    = out['segmentation']   # (1, 1, H, W_sc)
                    hidden = out.get('hidden')
                    if hidden is not None:
                        hidden = (hidden[0].detach(), hidden[1].detach())

                    pred_x_sc[t] = _predict_x_com(seg.squeeze(0))  # in scaled coords

                # Convert to original pixel coordinates
                pred_x_orig = pred_x_sc * w_scale   # (T,) in strip pixel coords

                # Evaluate at onset frames
                gt_onset   = notes['onset_sec']
                gt_strip_x = notes['strip_x']
                frame_idx  = np.clip(np.round(gt_onset * fps).astype(int), 0, T - 1)
                pred_at_onset = pred_x_orig[frame_idx]

                m = alignment_metrics(
                    pred_at_onset, gt_strip_x, px_per_sec,
                    beat_times_sec=ann.get('beat_times_sec') or None,
                    bar_times_sec=ann.get('bar_times_sec') or None,
                    gt_onset_sec=gt_onset)
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
                    print(f'  [{k+1}/{len(piece_ids)}] mean_err_sec='
                          f'{np.mean([r["mean_abs_err_sec"] for r in good]):.3f}',
                          flush=True)

    good = [r for r in rows if 'error' not in r]
    if not good:
        for r in [x for x in rows if 'error' in x][:3]:
            print(f'  {r["piece_id"]}: {r["error"]}\n{r.get("tb", "")}')
        return None

    keys = [k for k in good[0]
            if k.startswith(('mean_', 'median_', 'pct_', 'recall_')) or k == 'n']
    summ = {'n_pieces': len(good), 'n_errors': len(rows) - len(good)}
    for k in keys:
        vals = np.asarray([r[k] for r in good if isinstance(r.get(k), (int, float))])
        if len(vals):
            summ[f'mean_{k}'] = float(vals.mean())
            summ[f'median_{k}'] = float(np.median(vals))
    summ['split'] = split
    summ['checkpoint'] = str(checkpoint)

    with open(out_root / 'summary.json', 'w') as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2), flush=True)
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    p.add_argument('--config',     default='configs/v11_cpjku_fullstrip.yaml')
    p.add_argument('--processed',  default='data/MSMD/processed')
    p.add_argument('--out_dir',    default=None)
    p.add_argument('--limit',      type=int, default=None)
    p.add_argument('--device',     default=None)
    p.add_argument('--cpjku_fmt_root', default=None,
                   help='If set (or present in config data.cpjku_fmt_root), use cached '
                        'real-madmom spectrograms instead of the mel-spec approximation.')
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.split,
               out_dir=a.out_dir, limit=a.limit, device=a.device,
               cpjku_fmt_root=a.cpjku_fmt_root)


if __name__ == '__main__':
    main()
