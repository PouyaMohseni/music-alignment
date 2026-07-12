"""F3 -- extends F2's zero-retrain heatmap ensemble several ways at once:

1. Optional 4th member: v13_midi_privileged (E2/E3's checkpoint) shares
   identical strip geometry (h_strip=128, w_scale=4, fps=20) with v13/v14/v15
   even though it's from an early, under-converged training attempt (interim
   solo eval: 49.5% pct@0.5s at ~epoch 5-6). Testing whether folding it in
   (down-weighted) helps or hurts the F2 ensemble.

2. Offline-DTW / particle-filter decode on the ensemble-averaged column
   marginals, on top of the existing threshold+CoM decode -- all computed
   from the SAME forward pass per model-set (collect marginals once, decode
   N ways), mirroring v13_mert_unet/eval_particle_filter.py's E1 pattern. E1
   found DTW decode hurt pct@0.5s but helped mean/median error for a
   converged SINGLE model; testing whether that trade still holds once
   heatmaps are pre-averaged across models (which should already reduce
   per-frame noise).

3. Fusion method: mean (F2's original), median (robust to one member being
   confidently wrong on a frame), or max (confident-vote -- if any member is
   sure, trust it).

4. Pairwise subsets (e.g. --models v13,v14) to check which member pairs
   carry the ensemble's gain, diagnostic for where variance reduction comes
   from.

    python -m mymodel.f3_ensemble_decode.eval --models v13,v14,v15 --split test
    python -m mymodel.f3_ensemble_decode.eval --models v13,v14,v15,v13_midi --weights 0.3,0.3,0.3,0.1 --split test
    python -m mymodel.f3_ensemble_decode.eval --models v13,v14 --fusion median --split test
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.v13_mert_unet.eval import _build_perf_frame
from mymodel.v11_cpjku_fullstrip.data import load_strip_scaled
from mymodel.shared.metrics import alignment_metrics, henkel_metrics
from mymodel.d1_align_matrix.dtw import dtw_decode
from extensions.decode.particle_filter import ParticleFilterXTracker

MODEL_REGISTRY = {
    'v13':      ('/scratch/pmohseni/results/v13_mert_linear/best_model.pt', 'configs/v13_mert_linear.yaml'),
    'v14':      ('/scratch/pmohseni/results/v14_mert_bilstm/best_model.pt', 'configs/v14_mert_bilstm.yaml'),
    'v15':      ('/scratch/pmohseni/results/v15_mert_mlp/best_model.pt', 'configs/v15_mert_mlp.yaml'),
    'v13_midi': ('/scratch/pmohseni/results/v13_midi_privileged/best_model.pt', 'configs/v13_midi_privileged.yaml'),
}


def _load_model(checkpoint: str, cfg_path: str, device: str):
    cfg = OmegaConf.load(cfg_path)
    net_config = OmegaConf.to_container(cfg.net)
    network = ConditionalUNet(net_config)
    sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
    network.load_state_dict(sd['state_dict'], strict=True)
    network = network.to(device).eval()
    return network, cfg


def _predict_x_com_from_heatmap(heatmap_hw: np.ndarray, threshold: float = 0.5) -> float:
    arr = (heatmap_hw >= threshold).astype(np.float32)
    col = arr.sum(axis=0)
    total = col.sum()
    if total < 1e-6:
        return float(arr.shape[1] // 2)
    xs = np.arange(arr.shape[1], dtype=np.float32)
    return float((xs * col).sum() / total)


def _decode_offline_dtw(marginals: np.ndarray, band_frac: float = 0.05) -> np.ndarray:
    return dtw_decode(marginals.astype(np.float64), band_frac=band_frac).astype(np.float64)


def _decode_particle_filter(marginals: np.ndarray, process_noise_std: float = 3.0, init_std: float = 2.0) -> np.ndarray:
    tracker = ParticleFilterXTracker(process_noise_std=process_noise_std, init_std=init_std)
    T = marginals.shape[0]
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        out[t] = tracker.step(marginals[t])
    return out


def _fuse(heatmaps_list, weights, fusion: str) -> np.ndarray:
    """heatmaps_list: list of (H, W_sc) arrays, one per model, for a single frame."""
    if fusion == 'mean':
        out = np.zeros_like(heatmaps_list[0])
        for w, hm in zip(weights, heatmaps_list):
            out += w * hm
        return out
    stacked = np.stack(heatmaps_list, axis=0)  # (M, H, W_sc)
    if fusion == 'median':
        return np.median(stacked, axis=0)
    if fusion == 'max':
        return np.max(stacked, axis=0)
    raise ValueError(f'unknown fusion: {fusion}')


@torch.no_grad()
def eval_split(names, weights, processed_root: str, mert_emb_root: str, split: str,
              decoders=('original', 'offline_dtw'), fusion: str = 'mean', dtw_band_frac: float = 0.05,
              pf_process_noise_std: float = 3.0, pf_init_std: float = 2.0,
              out_dir: str = None, limit: int = None, device: str = None) -> dict | None:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    weights = weights or [1.0 / len(names)] * len(names)
    assert abs(sum(weights) - 1.0) < 1e-6, 'ensemble weights must sum to 1'
    assert len(weights) == len(names)

    loaded = []
    for name in names:
        ckpt, cfg_path = MODEL_REGISTRY[name]
        network, cfg = _load_model(ckpt, cfg_path, device)
        loaded.append((name, network, cfg))
        print(f'Loaded {name} from {ckpt}', flush=True)

    h_strips = {cfg.data.h_strip for _, _, cfg in loaded}
    w_scales = {cfg.data.w_scale for _, _, cfg in loaded}
    fps_set = {cfg.data.fps for _, _, cfg in loaded}
    assert len(h_strips) == 1 and len(w_scales) == 1 and len(fps_set) == 1, \
        f'FAIL: ensemble members have mismatched geometry: h_strip={h_strips} w_scale={w_scales} fps={fps_set}'
    fps = loaded[0][2].data.fps
    h_strip = loaded[0][2].data.h_strip
    w_scale = loaded[0][2].data.w_scale

    proc = Path(processed_root)
    emb = Path(mert_emb_root)
    piece_ids = json.load(open(proc / 'splits.json'))[split]
    if limit:
        piece_ids = piece_ids[:limit]

    out_root = Path(out_dir or f'results/f3_ensemble_decode/{"+".join(names)}') / split
    out_root.mkdir(parents=True, exist_ok=True)

    rows = {d: [] for d in decoders}
    with open(out_root / 'per_piece.jsonl', 'w') as fout:
        for k, pid in enumerate(piece_ids):
            try:
                piece_dir = proc / pid
                ann = json.load(open(piece_dir / 'annotations.json'))
                notes = np.load(piece_dir / 'noteheads.npz')

                strip_w = ann['image']['width_px']
                dur = float(ann['audio']['duration_sec'])
                px_per_sec = strip_w / dur

                feats = np.load(str(emb / f'{pid}.npy')).astype(np.float32)
                T = feats.shape[0]

                score = load_strip_scaled(piece_dir / 'strip.png', h_strip, w_scale)
                H, W_sc = score.shape
                score_1 = torch.from_numpy(score[np.newaxis, np.newaxis, np.newaxis]).to(device)

                hiddens = []
                for _, network, _ in loaded:
                    if network.use_lstm:
                        hiddens.append((torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
                                        torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device)))
                    else:
                        hiddens.append(None)

                pred_x_sc_orig = np.zeros(T, dtype=np.float64)
                marginals = np.zeros((T, W_sc), dtype=np.float64)
                for t in range(T):
                    member_heatmaps = []
                    for i, (name, network, cfg) in enumerate(loaded):
                        n_frames = cfg.data.n_frames
                        perf_np = _build_perf_frame(feats, t, n_frames)
                        perf_t = torch.from_numpy(perf_np).to(device).unsqueeze(0)
                        out = network(score=score_1, perf=perf_t, hidden=hiddens[i])
                        seg = out['segmentation']
                        new_hidden = out.get('hidden')
                        if new_hidden is not None:
                            hiddens[i] = (new_hidden[0].detach(), new_hidden[1].detach())
                        member_heatmaps.append(seg.squeeze(0).squeeze(0).cpu().numpy())
                    fused_heatmap = _fuse(member_heatmaps, weights, fusion)
                    pred_x_sc_orig[t] = _predict_x_com_from_heatmap(fused_heatmap)
                    marginals[t] = fused_heatmap.sum(axis=0)

                pred_x_sc = {'original': pred_x_sc_orig}
                if 'offline_dtw' in decoders:
                    pred_x_sc['offline_dtw'] = _decode_offline_dtw(marginals, dtw_band_frac)
                if 'particle_filter' in decoders:
                    pred_x_sc['particle_filter'] = _decode_particle_filter(marginals, pf_process_noise_std, pf_init_std)

                gt_onset = notes['onset_sec']
                gt_strip_x = notes['strip_x']
                frame_idx = np.clip(np.round(gt_onset * fps).astype(int), 0, T - 1)

                per_decoder_m = {}
                for dname in decoders:
                    pred_x_orig = pred_x_sc[dname] * w_scale
                    pred_at_onset = pred_x_orig[frame_idx]
                    m = alignment_metrics(pred_at_onset, gt_strip_x, px_per_sec,
                                          gt_onset_sec=gt_onset, all_strip_x=gt_strip_x,
                                          all_onset_sec=gt_onset)
                    m.update(henkel_metrics(pred_at_onset, gt_strip_x))
                    m['piece_id'] = pid
                    per_decoder_m[dname] = m
                    rows[dname].append(m)

                fout.write(json.dumps({'piece_id': pid, **per_decoder_m}) + '\n')
                fout.flush()

            except Exception as e:
                import traceback
                err = {'piece_id': pid, 'error': repr(e), 'tb': traceback.format_exc()}
                for dname in decoders:
                    rows[dname].append(err)
                fout.write(json.dumps(err) + '\n')
                fout.flush()

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
        summ = {'n_pieces': len(good), 'n_errors': len(rows[dname]) - len(good), 'members': names, 'weights': weights}
        for k in keys:
            vals = np.asarray([r[k] for r in good if isinstance(r.get(k), (int, float))])
            if len(vals):
                summ[f'mean_{k}'] = float(vals.mean())
        summ['split'] = split
        summary[dname] = summ

        with open(out_root / f'summary_{dname}.json', 'w') as f:
            json.dump(summ, f, indent=2)

        print(f'\n=== F3 ({"+".join(names)}) decoder={dname} on {split} ===')
        for thr in [0.05, 0.1, 0.5, 1.0, 5.0]:
            kk = f'mean_pct_within_{thr}s'
            if kk in summ:
                print(f'  pct@{thr}s  = {summ[kk]:.1f}%')
        print(f'  mean_err  = {summ.get("mean_mean_abs_err_sec", float("nan")):.3f}s')
        print(f'  median_err= {summ.get("mean_median_abs_err_sec", float("nan")):.3f}s')
        print(f'  n_pieces  = {summ["n_pieces"]}')

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', default='v13,v14,v15', help='comma-separated names from MODEL_REGISTRY')
    p.add_argument('--weights', default=None, help='comma-separated weights, must sum to 1 (default: uniform)')
    p.add_argument('--decoders', default='original,offline_dtw', help='comma-separated: original,offline_dtw,particle_filter')
    p.add_argument('--fusion', default='mean', choices=['mean', 'median', 'max'])
    p.add_argument('--dtw_band_frac', type=float, default=0.05)
    p.add_argument('--pf_process_noise_std', type=float, default=3.0)
    p.add_argument('--pf_init_std', type=float, default=2.0)
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--processed', default='data/MSMD/processed')
    p.add_argument('--mert_emb_root', default='data/MSMD/mert_emb')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--device', default=None)
    a = p.parse_args()
    names = a.models.split(',')
    weights = [float(w) for w in a.weights.split(',')] if a.weights else None
    decoders = tuple(a.decoders.split(','))
    eval_split(names, weights, a.processed, a.mert_emb_root, a.split,
              decoders=decoders, fusion=a.fusion, dtw_band_frac=a.dtw_band_frac,
              pf_process_noise_std=a.pf_process_noise_std, pf_init_std=a.pf_init_std,
              out_dir=a.out_dir, limit=a.limit, device=a.device)


if __name__ == '__main__':
    main()
