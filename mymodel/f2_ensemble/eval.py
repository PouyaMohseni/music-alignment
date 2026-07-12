"""F2 -- zero-retrain heatmap ensemble across v13/v14/v15 (all converged:
66.1%/66.0%/66.4% pct@0.5s individually). Classic deep-ensemble technique
(Lakshminarayanan et al. 2017): three independently-trained, architecturally-
diverse models (differ only in audio-encoder head -- MERTProjector/
MERTBiLSTM/MERTMlpProjector; identical h_strip/w_scale/gt_width/fps, so their
per-frame heatmaps are directly comparable and averageable) should have
partially decorrelated errors, so averaging their per-frame sigmoid heatmaps
before decode should reduce variance and improve accuracy -- at ZERO
retraining cost, just 3x eval-time compute.

No MIDI, no new training signal -- this is purely a decode-time combination
of three checkpoints that already exist.

    python -m mymodel.f2_ensemble.eval --split test
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

DEFAULT_MODELS = [
    ('v13', '/scratch/pmohseni/results/v13_mert_linear/best_model.pt', 'configs/v13_mert_linear.yaml'),
    ('v14', '/scratch/pmohseni/results/v14_mert_bilstm/best_model.pt', 'configs/v14_mert_bilstm.yaml'),
    ('v15', '/scratch/pmohseni/results/v15_mert_mlp/best_model.pt', 'configs/v15_mert_mlp.yaml'),
]


def _load_model(checkpoint: str, cfg_path: str, device: str):
    cfg = OmegaConf.load(cfg_path)
    net_config = OmegaConf.to_container(cfg.net)
    network = ConditionalUNet(net_config)
    sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
    network.load_state_dict(sd['state_dict'], strict=True)
    network = network.to(device).eval()
    return network, cfg


def _predict_x_com_from_heatmap(heatmap_hw: np.ndarray, threshold: float = 0.5) -> float:
    """heatmap_hw: (H, W) averaged sigmoid probability. Same thresholded
    center-of-mass convention as v13's own _predict_x_com."""
    arr = (heatmap_hw >= threshold).astype(np.float32)
    col = arr.sum(axis=0)
    total = col.sum()
    if total < 1e-6:
        return float(arr.shape[1] // 2)
    xs = np.arange(arr.shape[1], dtype=np.float32)
    return float((xs * col).sum() / total)


@torch.no_grad()
def eval_split(models_spec, processed_root: str, mert_emb_root: str, split: str,
              weights=None, out_dir: str = None, limit: int = None, device: str = None) -> dict | None:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    names = [m[0] for m in models_spec]
    weights = weights or [1.0 / len(models_spec)] * len(models_spec)
    assert abs(sum(weights) - 1.0) < 1e-6, 'ensemble weights must sum to 1'

    loaded = []
    for name, ckpt, cfg_path in models_spec:
        network, cfg = _load_model(ckpt, cfg_path, device)
        loaded.append((name, network, cfg))
        print(f'Loaded {name} from {ckpt}', flush=True)

    # sanity: all models must share the same strip geometry for averaging to be valid
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

    out_root = Path(out_dir or f'results/f2_ensemble/{"+".join(names)}') / split
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

                pred_x_sc = np.zeros(T, dtype=np.float64)
                for t in range(T):
                    avg_heatmap = np.zeros((H, W_sc), dtype=np.float32)
                    for i, (name, network, cfg) in enumerate(loaded):
                        n_frames = cfg.data.n_frames
                        perf_np = _build_perf_frame(feats, t, n_frames)
                        perf_t = torch.from_numpy(perf_np).to(device).unsqueeze(0)
                        out = network(score=score_1, perf=perf_t, hidden=hiddens[i])
                        seg = out['segmentation']
                        new_hidden = out.get('hidden')
                        if new_hidden is not None:
                            hiddens[i] = (new_hidden[0].detach(), new_hidden[1].detach())
                        avg_heatmap += weights[i] * seg.squeeze(0).squeeze(0).cpu().numpy()
                    pred_x_sc[t] = _predict_x_com_from_heatmap(avg_heatmap)

                pred_x_orig = pred_x_sc * w_scale
                gt_onset = notes['onset_sec']
                gt_strip_x = notes['strip_x']
                frame_idx = np.clip(np.round(gt_onset * fps).astype(int), 0, T - 1)
                pred_at_onset = pred_x_orig[frame_idx]

                m = alignment_metrics(pred_at_onset, gt_strip_x, px_per_sec,
                                      gt_onset_sec=gt_onset, all_strip_x=gt_strip_x,
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

    keys = [k for k in good[0] if k.startswith(('mean_', 'median_', 'pct_', 'n'))]
    summ = {'n_pieces': len(good), 'n_errors': len(rows) - len(good), 'members': names, 'weights': weights}
    for k in keys:
        vals = np.asarray([r[k] for r in good if isinstance(r.get(k), (int, float))])
        if len(vals):
            summ[f'mean_{k}'] = float(vals.mean())
    summ['split'] = split

    with open(out_root / 'summary.json', 'w') as f:
        json.dump(summ, f, indent=2)

    print(f'\n=== F2 ensemble ({"+".join(names)}) on {split} ===')
    for thr in [0.05, 0.1, 0.5, 1.0, 5.0]:
        kk = f'mean_pct_within_{thr}s'
        if kk in summ:
            print(f'  pct@{thr}s  = {summ[kk]:.1f}%')
    print(f'  mean_err  = {summ.get("mean_mean_abs_err_sec", float("nan")):.3f}s')
    print(f'  median_err= {summ.get("mean_median_abs_err_sec", float("nan")):.3f}s')
    print(f'  n_pieces  = {summ["n_pieces"]}')
    return summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--processed', default='data/MSMD/processed')
    p.add_argument('--mert_emb_root', default='data/MSMD/mert_emb')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--device', default=None)
    a = p.parse_args()
    eval_split(DEFAULT_MODELS, a.processed, a.mert_emb_root, a.split,
              out_dir=a.out_dir, limit=a.limit, device=a.device)


if __name__ == '__main__':
    main()
