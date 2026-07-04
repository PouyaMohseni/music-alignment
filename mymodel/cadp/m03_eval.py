"""CADP M03 evaluation — DTW on sim matrix, reports pct@0.5s.

    python -m mymodel.cadp.m03_eval \
        --checkpoint results/cadp_m03/best_model.pt \
        --config     configs/cadp_m03.yaml \
        --split      test
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mymodel.cadp.m03_model import M03LSTMTemporal
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

    model = M03LSTMTemporal(
        mert_dim=cfg.model.mert_dim,
        dinov2_dim=cfg.model.dinov2_dim,
        embed_dim=cfg.model.embed_dim,
        n_audio_chunks=cfg.model.n_audio_chunks,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        lstm_bidirectional=cfg.model.lstm_bidirectional,
    ).to(device).eval()
    sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['state_dict'], strict=True)

    ds = CADPDataset(
        cfg.data.processed_root, cfg.data.mert_root, cfg.data.dinov2_root,
        split=split, fps=cfg.data.fps)

    out_root = Path(out_dir or str(Path(checkpoint).parent / 'eval')) / split
    out_root.mkdir(parents=True, exist_ok=True)

    # Training only ever saw win_sec/n_audio_chunks-second chunks (e.g. 5s/20=0.25s).
    # Pooling a whole piece to a fixed n_audio_chunks coarsens resolution far below
    # the 0.5s eval threshold, so scale chunk count to match training's time-per-chunk.
    sec_per_chunk = cfg.train.win_sec / cfg.model.n_audio_chunks
    rows = []
    with open(out_root / 'per_piece.jsonl', 'w') as fout:
        for k, pid in enumerate(ds.piece_ids):
            try:
                piece = ds.load_piece(pid)
                if piece is None:
                    raise FileNotFoundError(f'Missing features for {pid}')

                mert_feats = torch.from_numpy(piece['mert_feats']).unsqueeze(0).to(device)  # (1, T, 768)
                d2_feats   = torch.from_numpy(piece['d2_feats']).to(device)                  # (N_cols, 16, 768)

                T_a = mert_feats.shape[1]
                n_chunks = max(1, round((T_a / cfg.data.fps) / sec_per_chunk))

                out = model(mert_feats, d2_feats, n_chunks=n_chunks)
                sim = out['sim'].squeeze(0).cpu().numpy()   # (n_chunks, N_cols)

                path = dtw_backtrack(sim, band_radius_frac=0.25)  # (P, 2) [chunk_idx, col_idx]
                chunk_to_col = np.zeros(n_chunks, dtype=np.float64)
                for chunk_i, col_i in path:
                    chunk_to_col[chunk_i] = col_i

                fps = cfg.data.fps
                dur = piece['dur']
                onset_sec  = piece['onset_sec']
                strip_x_gt = piece['strip_x']
                strip_w    = piece['strip_w']
                px_per_sec = strip_w / dur

                f_idx = np.clip(np.round(onset_sec * fps).astype(np.int64),
                                0, T_a - 1)
                frames_per_chunk = T_a / n_chunks
                chunk_idx = np.clip(np.floor(f_idx / frames_per_chunk).astype(np.int64),
                                    0, n_chunks - 1)
                pred_col = chunk_to_col[chunk_idx]

                col_stride = float(piece['col_stride'])
                col_w      = float(ds.col_w)
                # Matches dataset.py's GT mapping: column i center = i*stride + col_w/2
                pred_strip_x = pred_col * col_stride + col_w / 2.0

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
