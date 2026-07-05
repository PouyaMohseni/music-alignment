"""CADP M04 evaluation — DTW decode over the dense (unpooled) sim matrix.

Unlike M01/M03, audio is used at native (dense) resolution here, so there's
no chunk-count-vs-piece-length mismatch to get wrong — the whole piece's sim
matrix is computed once and DTW runs directly on it. Still uses a hard,
whole-piece monotonic DTW backtrack, so it inherits M01/M03's cascade-failure
mode on tempo-drift/repeat pieces (Chopin nocturne, Satie Gymnopedie, etc.) —
that comparison against M05 is the point of this ablation.

    python -m mymodel.cadp.m04_eval \
        --checkpoint results/cadp_m04/best_model.pt \
        --config     configs/cadp_m04.yaml \
        --split      test
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mymodel.cadp.m04_model import M04DenseTokens, subcol_positions
from mymodel.cadp.dataset import CADPDataset
from mymodel.shared.metrics import alignment_metrics, henkel_metrics, dtw_backtrack


@torch.no_grad()
def eval_split(checkpoint: str, cfg_path: str, split: str,
               out_dir: str = None, device: str = None,
               overrides: list[str] | None = None) -> dict | None:
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model = M04DenseTokens(
        mert_dim=cfg.model.mert_dim,
        dinov2_dim=cfg.model.dinov2_dim,
        hidden_dim=cfg.model.hidden_dim,
        embed_dim=cfg.model.embed_dim,
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

    rows = []
    with open(out_root / 'per_piece.jsonl', 'w') as fout:
        for k, pid in enumerate(ds.piece_ids):
            try:
                piece = ds.load_piece(pid)
                if piece is None:
                    raise FileNotFoundError(f'Missing features for {pid}')

                mert_feats = torch.from_numpy(piece['mert_feats']).to(device)  # (T, 768)
                d2_feats   = torch.from_numpy(piece['d2_feats']).to(device)     # (N_cols, 16, 768)
                T = mert_feats.shape[0]
                n_cols = d2_feats.shape[0]

                out = model(mert_feats, d2_feats)
                sim = out['sim'].cpu().numpy()          # (T, N_s)

                path = dtw_backtrack(sim, band_radius_frac=0.25)  # (P, 2) [frame_idx, subcol_idx]
                frame_to_subcol = np.zeros(T, dtype=np.float64)
                for frame_i, subcol_i in path:
                    frame_to_subcol[frame_i] = subcol_i

                pos_subcol = subcol_positions(n_cols, col_stride, col_w).numpy()  # (N_s,)

                onset_sec  = piece['onset_sec']
                strip_x_gt = piece['strip_x']
                strip_w    = piece['strip_w']
                dur        = piece['dur']
                px_per_sec = strip_w / dur

                fps = cfg.data.fps
                f_idx = np.clip(np.round(onset_sec * fps).astype(np.int64), 0, T - 1)
                pred_subcol = frame_to_subcol[f_idx].astype(np.int64)
                pred_strip_x = pos_subcol[pred_subcol]

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
