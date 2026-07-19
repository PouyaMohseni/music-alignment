"""C1 eval — mirrors mymodel/v11_cpjku_fullstrip/eval.py exactly, swapping
in C1VisualGroundingNet in place of ConditionalUNet.

    python -m mymodel.c1_visual_grounding.eval \
        --checkpoint results/c1_visual_grounding/best_model.pt \
        --config     configs/c1_visual_grounding.yaml \
        --split      test \
        --processed  data/MSMD/processed
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mymodel.c1_visual_grounding.model import C1VisualGroundingNet
from mymodel.v11_cpjku_fullstrip.data import load_spec, load_strip_scaled
from mymodel.shared.metrics import alignment_metrics, henkel_metrics


def _build_perf_frame(spec: np.ndarray, t: int, n_frames: int) -> np.ndarray:
    t0 = max(0, t - n_frames)
    window = spec[:, t0:t]
    if window.shape[-1] < n_frames:
        window = np.pad(window, ((0, 0), (n_frames - window.shape[-1], 0)))
    return window[np.newaxis, np.newaxis]


def _predict_x_com(seg: torch.Tensor) -> float:
    """Weighted center-of-mass over the RAW attention distribution -- no
    threshold. seg is a softmax over score patches (sums to 1 across the
    whole strip), so its per-patch max is structurally capped and rarely
    crosses a sigmoid-style 0.5 threshold (confirmed empirically: only
    ~17% of frames on a real test piece). The previous `>=0.5` threshold
    (copied from mymodel/v11_cpjku_fullstrip/eval.py, appropriate THERE
    because that model's decoder ends in torch.sigmoid, giving independent
    per-pixel probabilities that legitimately saturate near 1.0) made this
    function silently fall back to the literal strip midpoint on the large
    majority of frames, discarding the model's actual prediction. Matches
    train.py's own frame-accuracy diagnostic, which reads the raw
    distribution via argmax with no threshold."""
    arr = seg.squeeze(0).cpu().numpy()
    col = arr.sum(axis=0)
    total = col.sum()
    xs = np.arange(arr.shape[1], dtype=np.float32)
    return float((xs * col).sum() / total)


@torch.no_grad()
def eval_split(checkpoint: str, cfg_path: str, processed_root: str, split: str,
               out_dir: str = None, limit: int = None, device: str = None) -> dict | None:
    cfg = OmegaConf.load(cfg_path)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    network = C1VisualGroundingNet(
        spec_enc=cfg.net.spec_enc, rnn_size=cfg.net.rnn_size, rnn_layers=cfg.net.rnn_layer,
        d_model=cfg.net.d_model, n_heads=cfg.net.n_heads, patch_w=cfg.net.patch_w)
    sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
    # patch_conv is built lazily on first forward -- run one dummy forward at
    # the config's h_strip so load_state_dict finds a matching key already there.
    network.score_encoder._build(cfg.data.h_strip, torch.device('cpu'))
    network.load_state_dict(sd['state_dict'], strict=True)
    network = network.to(device)
    network.eval()

    fps, n_frames = cfg.data.fps, cfg.data.n_frames
    h_strip, w_scale, n_mels = cfg.data.h_strip, cfg.data.w_scale, cfg.data.n_mels

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
                ann = json.load(open(piece_dir / 'annotations.json'))
                notes = np.load(piece_dir / 'noteheads.npz')

                strip_w = ann['image']['width_px']
                dur = float(ann['audio']['duration_sec'])
                px_per_sec = strip_w / dur

                spec = load_spec(piece_dir / 'audio.wav', fps=fps, n_mels=n_mels)
                score = load_strip_scaled(piece_dir / 'strip.png', h_strip, w_scale)
                H, W_sc = score.shape
                T = spec.shape[-1]

                score_t = torch.from_numpy(score[np.newaxis, np.newaxis]).to(device)
                score_1 = score_t.unsqueeze(0)

                hidden = (torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
                         torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device))

                pred_x_sc = np.zeros(T, dtype=np.float64)
                for t in range(T):
                    perf_np = _build_perf_frame(spec, t, n_frames)
                    perf_t = torch.from_numpy(perf_np).to(device).unsqueeze(0)

                    out = network(score=score_1, perf=perf_t, hidden=hidden)
                    seg = out['segmentation']
                    hidden = out.get('hidden')
                    if hidden is not None:
                        hidden = (hidden[0].detach(), hidden[1].detach())

                    pred_x_sc[t] = _predict_x_com(seg.squeeze(0))

                pred_x_orig = pred_x_sc * w_scale

                gt_onset = notes['onset_sec']
                gt_strip_x = notes['strip_x']
                frame_idx = np.clip(np.round(gt_onset * fps).astype(int), 0, T - 1)
                pred_at_onset = pred_x_orig[frame_idx]

                m = alignment_metrics(pred_at_onset, gt_strip_x, px_per_sec,
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
                          f'{np.mean([r["mean_abs_err_sec"] for r in good]):.3f}', flush=True)

    good = [r for r in rows if 'error' not in r]
    if not good:
        for r in [x for x in rows if 'error' in x][:3]:
            print(f'  {r["piece_id"]}: {r["error"]}\n{r.get("tb", "")}')
        return None

    keys = [k for k in good[0] if k.startswith(('mean_', 'median_', 'pct_', 'recall_')) or k == 'n']
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
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--config', default='configs/c1_visual_grounding.yaml')
    p.add_argument('--processed', default='data/MSMD/processed')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--device', default=None)
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.split,
               out_dir=a.out_dir, limit=a.limit, device=a.device)


if __name__ == '__main__':
    main()
