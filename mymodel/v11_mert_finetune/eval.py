"""v11-mert-finetune eval -- full strip inference with LSTM state, live MERT.

    python -m mymodel.v11_mert_finetune.eval \
        --checkpoint results/v11_mert_finetune/best_model.pt \
        --config     configs/v11_mert_finetune.yaml \
        --split      test \
        --processed  data/MSMD/processed
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.v10_mert_unet.mert_live import MERTLive, MERT_SR
from mymodel.v11_cpjku_fullstrip.data import load_strip_scaled
from mymodel.v11_mert_finetune.data import load_raw_audio
from mymodel.shared.metrics import alignment_metrics, henkel_metrics


def _predict_x_com(seg: torch.Tensor, threshold: float = 0.5) -> float:
    arr = seg.squeeze(0).cpu().numpy()
    arr = (arr >= threshold).astype(np.float32)
    col = arr.sum(axis=0)
    total = col.sum()
    if total < 1e-6:
        return float(arr.shape[1] // 2)
    xs = np.arange(arr.shape[1], dtype=np.float32)
    return float((xs * col).sum() / total)


@torch.no_grad()
def eval_split(checkpoint: str, cfg_path: str, processed_root: str, split: str,
               out_dir: str = None, limit: int = None, device: str = None,
               window_sec: float = None) -> dict | None:
    cfg    = OmegaConf.load(cfg_path)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    window_sec = window_sec or cfg.train.window_sec

    net_config = OmegaConf.to_container(cfg.net)
    network = ConditionalUNet(net_config)
    sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
    network.load_state_dict(sd['state_dict'], strict=True)
    network = network.to(device)
    network.eval()

    mert_live = MERTLive(mert_id=cfg.mert.mert_id,
                         unfreeze_last_n=cfg.mert.get('unfreeze_last_n', None))
    mert_live.load_state_dict(sd['mert_state_dict'], strict=True)
    mert_live = mert_live.to(device)
    mert_live.eval()

    fps      = cfg.data.fps
    h_strip  = cfg.data.h_strip
    w_scale  = cfg.data.w_scale

    proc = Path(processed_root)
    piece_ids = json.load(open(proc / 'splits.json'))[split]
    if limit:
        piece_ids = piece_ids[:limit]

    out_root = Path(out_dir or str(Path(checkpoint).parent / 'eval')) / split
    out_root.mkdir(parents=True, exist_ok=True)

    win_samples = int(window_sec * MERT_SR)

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

                audio_24k = torch.from_numpy(load_raw_audio(piece_dir / 'audio.wav')).to(device)
                score = load_strip_scaled(piece_dir / 'strip.png', h_strip, w_scale)
                H, W_sc = score.shape
                T = int(np.floor(audio_24k.shape[0] / MERT_SR * fps))

                score_t = torch.from_numpy(score[np.newaxis, np.newaxis]).to(device)
                score_1 = score_t.unsqueeze(0)

                hidden = None
                if network.use_lstm:
                    hidden = (torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
                              torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device))

                pred_x_sc = np.zeros(T, dtype=np.float64)

                for t in range(T):
                    end_sample = min(audio_24k.shape[0], int(round((t + 1) / fps * MERT_SR)))
                    start_sample = max(0, end_sample - win_samples)
                    window = audio_24k[start_sample:end_sample]
                    if window.shape[0] < win_samples:
                        window = F.pad(window, (win_samples - window.shape[0], 0))
                    emb = mert_live.embed_window(window, n_frames_20fps=1)  # (1, 768)
                    perf_t = emb.view(1, 1, 1, 768, 1)

                    out    = network(score=score_1, perf=perf_t, hidden=hidden)
                    seg    = out['segmentation']
                    hidden = out.get('hidden')
                    if hidden is not None:
                        hidden = (hidden[0].detach(), hidden[1].detach())

                    pred_x_sc[t] = _predict_x_com(seg.squeeze(0))

                pred_x_orig = pred_x_sc * w_scale

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
    p.add_argument('--config',     default='configs/v11_mert_finetune.yaml')
    p.add_argument('--processed',  default='data/MSMD/processed')
    p.add_argument('--out_dir',    default=None)
    p.add_argument('--limit',      type=int, default=None)
    p.add_argument('--device',     default=None)
    a = p.parse_args()
    eval_split(a.checkpoint, a.config, a.processed, a.split,
               out_dir=a.out_dir, limit=a.limit, device=a.device)


if __name__ == '__main__':
    main()
