"""CADP M05 evaluation — direct per-frame soft-argmax decode, no DTW.

Processes each piece in win_sec windows (same window size as training, full
score always in context) and reads off each annotated onset's predicted pixel
position directly from that window's output. There is no global monotonic
path to solve and therefore no cascade-failure mode from tempo drift or
repeats — the structural issue that broke M01/M03's whole-piece DTW decode.

    python -m mymodel.cadp.m05_eval \
        --checkpoint results/cadp_m05/best_model.pt \
        --config     configs/cadp_m05.yaml \
        --split      test
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mymodel.cadp.m05_model import M05LearnedPathPredictor
from mymodel.cadp.m04_model import subcol_positions
from mymodel.cadp.dataset import CADPDataset
from mymodel.shared.metrics import alignment_metrics, henkel_metrics


@torch.no_grad()
def eval_split(checkpoint: str, cfg_path: str, split: str,
               out_dir: str = None, device: str = None,
               overrides: list[str] | None = None) -> dict | None:
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model = M05LearnedPathPredictor(
        mert_dim=cfg.model.mert_dim,
        dinov2_dim=cfg.model.dinov2_dim,
        hidden_dim=cfg.model.hidden_dim,
        embed_dim=cfg.model.embed_dim,
        path_channels=cfg.model.path_channels,
        attention_heads=cfg.model.attention_heads,
    ).to(device).eval()
    sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['state_dict'], strict=True)

    ds = CADPDataset(
        cfg.data.processed_root, cfg.data.mert_root, cfg.data.dinov2_root,
        split=split, fps=cfg.data.fps)
    col_stride = float(ds.col_stride)
    col_w = float(ds.col_w)

    out_root = Path(out_dir or str(Path(checkpoint).parent / 'eval')) / split
    out_root.mkdir(parents=True, exist_ok=True)

    win_frames = int(cfg.train.win_sec * cfg.data.fps)
    rows = []
    with open(out_root / 'per_piece.jsonl', 'w') as fout:
        for k, pid in enumerate(ds.piece_ids):
            try:
                piece = ds.load_piece(pid)
                if piece is None:
                    raise FileNotFoundError(f'Missing features for {pid}')

                mert_feats = piece['mert_feats']     # (T, 768)
                d2_feats   = piece['d2_feats']        # (N_cols, 16, 768)
                T = mert_feats.shape[0]
                n_cols = d2_feats.shape[0]

                score_t = torch.from_numpy(d2_feats).to(device)
                pos_subcol = subcol_positions(n_cols, col_stride, col_w).to(device)

                pred_pos_full = torch.zeros(T, device=device)
                for t0 in range(0, T, win_frames):
                    t1 = min(t0 + win_frames, T)
                    audio_t = torch.from_numpy(mert_feats[t0:t1]).to(device)
                    out = model(audio_t, score_t, pos_subcol,
                                temperature=cfg.loss.temperature)
                    pred_pos_full[t0:t1] = out['pred_pos']

                onset_sec  = piece['onset_sec']
                strip_x_gt = piece['strip_x']
                strip_w    = piece['strip_w']
                dur        = piece['dur']
                px_per_sec = strip_w / dur

                fps = cfg.data.fps
                f_idx = np.clip(np.round(onset_sec * fps).astype(np.int64), 0, T - 1)
                pred_strip_x = pred_pos_full[f_idx].cpu().numpy()

                m = alignment_metrics(
                    pred_strip_x, strip_x_gt, px_per_sec,
                    gt_onset_sec=onset_sec,
                    all_strip_x=strip_x_gt,
                    all_onset_sec=onset_sec)
                m.update(henkel_metrics(pred_strip_x, strip_x_gt))
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
                    print(f'  [{k+1}/{len(ds.piece_ids)}]  pct@0.5s={pct:.1f}%', flush=True)

    good = [r for r in rows if 'error' not in r]
    if not good:
        for r in [x for x in rows if 'error' in x][:3]:
            print(f'  ERROR {r["piece_id"]}: {r["error"]}\n{r.get("tb","")}')
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
    print(f'  n_pieces  = {summ["n_pieces"]}')
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     required=True)
    p.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    p.add_argument('--out_dir',    default=None)
    p.add_argument('--device',     default=None)
    p.add_argument('overrides', nargs='*')
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.split, a.out_dir, a.device, a.overrides)


if __name__ == '__main__':
    main()
