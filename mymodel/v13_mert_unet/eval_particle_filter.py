"""E1: decode retrofit for v13/v14/v15 -- particle-filter and offline-DTW
decoders on top of the SAME per-frame heatmap forward pass eval.py already
computes, no retraining. Tests whether D2's decode-only win (particle
filter: 5.1%->23.7% pct@0.5s on the same similarity matrix, zero retrain)
transfers to the project's strongest models.

Runs all three decoders (original threshold+CoM, particle filter, offline
DTW over collected per-frame marginals) in ONE forward pass per piece
(collecting each frame's raw sigmoid heatmap once, decoding it three ways),
so this costs the same GPU time as a single eval.py run, not three.

    python -m mymodel.v13_mert_unet.eval_particle_filter \
        --checkpoint /scratch/pmohseni/results/v13_mert_linear/best_model.pt \
        --config     configs/v13_mert_linear.yaml \
        --split      test
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.shared.metrics import alignment_metrics, henkel_metrics
from mymodel.v13_mert_unet.eval import _build_perf_frame, _predict_x_com
from extensions.decode.particle_filter import ParticleFilterXTracker, heatmap_to_x_marginal
from mymodel.d1_align_matrix.dtw import dtw_decode


def _decode_particle_filter(marginals: np.ndarray, process_noise_std: float, init_std: float) -> np.ndarray:
    """marginals: (T, W_sc) non-negative per-frame column marginals."""
    tracker = ParticleFilterXTracker(process_noise_std=process_noise_std, init_std=init_std)
    T = marginals.shape[0]
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        out[t] = tracker.step(marginals[t])
    return out


def _decode_offline_dtw(marginals: np.ndarray, band_frac: float = 0.05) -> np.ndarray:
    """marginals: (T, W_sc) -> treat as a similarity matrix (already
    non-negative sums, monotone-in-match like D1/D2's cosine-sim matrix) and
    reuse dtw_decode directly -- it only requires "higher = better match"."""
    return dtw_decode(marginals.astype(np.float64), band_frac=band_frac).astype(np.float64)


@torch.no_grad()
def eval_split(checkpoint: str, cfg_path: str, processed_root: str,
               mert_emb_root: str, split: str,
               pf_process_noise_std: float = 3.0, pf_init_std: float = 2.0,
               dtw_band_frac: float = 0.05,
               out_dir: str = None, limit: int = None, device: str = None) -> dict:
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

    out_root = Path(out_dir or str(Path(checkpoint).parent / 'eval_pf')) / split
    out_root.mkdir(parents=True, exist_ok=True)

    decoders = ['original', 'particle_filter', 'offline_dtw']
    rows = {d: [] for d in decoders}

    for k, pid in enumerate(piece_ids):
        try:
            piece_dir = proc / pid
            ann   = json.load(open(piece_dir / 'annotations.json'))
            notes = np.load(piece_dir / 'noteheads.npz')

            strip_w    = ann['image']['width_px']
            dur        = float(ann['audio']['duration_sec'])
            px_per_sec = strip_w / dur

            feats = np.load(str(emb / f'{pid}.npy')).astype(np.float32)
            T = feats.shape[0]

            score = load_strip_scaled(piece_dir / 'strip.png', h_strip, w_scale)
            H, W_sc = score.shape
            score_1 = torch.from_numpy(score[np.newaxis, np.newaxis, np.newaxis]).to(device)

            hidden = None
            if network.use_lstm:
                hidden = (torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
                          torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device))

            pred_x_sc_orig = np.zeros(T, dtype=np.float64)
            marginals = np.zeros((T, W_sc), dtype=np.float64)
            for t in range(T):
                perf_np = _build_perf_frame(feats, t, n_frames)
                perf_t  = torch.from_numpy(perf_np).to(device).unsqueeze(0)
                out = network(score=score_1, perf=perf_t, hidden=hidden)
                seg = out['segmentation']
                hidden = out.get('hidden')
                if hidden is not None:
                    hidden = (hidden[0].detach(), hidden[1].detach())
                seg_np = seg.squeeze(0).squeeze(0).cpu().numpy()   # (H, W_sc)
                pred_x_sc_orig[t] = _predict_x_com(seg.squeeze(0))
                marginals[t] = heatmap_to_x_marginal(seg_np)

            pred_x_sc = {
                'original': pred_x_sc_orig,
                'particle_filter': _decode_particle_filter(marginals, pf_process_noise_std, pf_init_std),
                'offline_dtw': _decode_offline_dtw(marginals, dtw_band_frac),
            }

            gt_onset    = notes['onset_sec']
            gt_strip_x  = notes['strip_x']
            frame_idx   = np.clip(np.round(gt_onset * fps).astype(int), 0, T - 1)

            for dname in decoders:
                pred_x_orig = pred_x_sc[dname] * w_scale
                pred_at_onset = pred_x_orig[frame_idx]
                m = alignment_metrics(pred_at_onset, gt_strip_x, px_per_sec,
                                      gt_onset_sec=gt_onset, all_strip_x=gt_strip_x,
                                      all_onset_sec=gt_onset)
                m.update(henkel_metrics(pred_at_onset, gt_strip_x))
                m['piece_id'] = pid
                rows[dname].append(m)

        except Exception as e:
            import traceback
            err = {'piece_id': pid, 'error': repr(e), 'tb': traceback.format_exc()}
            for dname in decoders:
                rows[dname].append(err)

        if (k + 1) % 10 == 0:
            line = f'  [{k+1}/{len(piece_ids)}]'
            for dname in decoders:
                good = [r for r in rows[dname] if 'error' not in r]
                pct = np.mean([r.get('pct_within_0.5s', 0) for r in good]) if good else float('nan')
                line += f'  {dname}={pct:.1f}%'
            print(line, flush=True)

    summary = {}
    for dname in decoders:
        good = [r for r in rows[dname] if 'error' not in r]
        if not good:
            print(f'{dname}: ALL PIECES ERRORED, first error: {[r for r in rows[dname] if "error" in r][:1]}')
            continue
        keys = [k for k in good[0] if k.startswith(('mean_', 'median_', 'pct_', 'n'))]
        summ = {'n_pieces': len(good), 'n_errors': len(rows[dname]) - len(good)}
        for k in keys:
            vals = np.asarray([r[k] for r in good if isinstance(r.get(k), (int, float))])
            if len(vals):
                summ[f'mean_{k}'] = float(vals.mean())
        summary[dname] = summ

        with open(out_root / f'summary_{dname}.json', 'w') as f:
            json.dump(summ, f, indent=2)

        print(f'\n=== {dname} ({split}) ===', flush=True)
        for thr in [0.05, 0.1, 0.5, 1.0, 5.0]:
            kk = f'mean_pct_within_{thr}s'
            if kk in summ:
                print(f'  pct@{thr}s  = {summ[kk]:.1f}%', flush=True)
        print(f'  mean_err  = {summ.get("mean_mean_abs_err_sec", float("nan")):.3f}s', flush=True)
        print(f'  median_err= {summ.get("mean_median_abs_err_sec", float("nan")):.3f}s', flush=True)
        print(f'  n_pieces  = {summ["n_pieces"]}', flush=True)

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    required=True)
    p.add_argument('--config',        required=True)
    p.add_argument('--split',         default='test', choices=['train', 'val', 'test'])
    p.add_argument('--processed',     default='data/MSMD/processed')
    p.add_argument('--mert_emb_root', default='data/MSMD/mert_emb')
    p.add_argument('--pf_process_noise_std', type=float, default=3.0)
    p.add_argument('--pf_init_std',   type=float, default=2.0)
    p.add_argument('--dtw_band_frac', type=float, default=0.05)
    p.add_argument('--out_dir',       default=None)
    p.add_argument('--limit',         type=int, default=None)
    p.add_argument('--device',        default=None)
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.mert_emb_root, a.split,
              pf_process_noise_std=a.pf_process_noise_std, pf_init_std=a.pf_init_std,
              dtw_band_frac=a.dtw_band_frac, out_dir=a.out_dir, limit=a.limit, device=a.device)


if __name__ == '__main__':
    main()
