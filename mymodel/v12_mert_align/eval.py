"""
Evaluate v12 MERT alignment on MSMD test split.

For each piece:
  1. Compute similarity matrix (MERT audio vs ResNet score columns)
  2. DTW decode -> path[t] = predicted column for each audio frame
  3. Convert path to predicted x_px per frame
  4. For each GT onset (onset_sec, strip_x):
       find predicted_x at onset frame
       use piecewise-linear interpolation of GT path to invert: predicted_x -> predicted_time
       error = |predicted_time - onset_sec|
  5. Aggregate pct_within_{0.05,0.1,0.25,0.5,1.0}s and median/mean error
"""
import argparse, json, os
import numpy as np
import torch
from scipy import interpolate

from mymodel.v12_mert_align.dataset import (
    MSMDPiece, MSMDDataset, MERT_HZ, AUDIO_SR, col_centers
)
from mymodel.v12_mert_align.model import MERTAlignModel
from mymodel.v12_mert_align.dtw import dtw_decode


THRESHOLDS = [0.05, 0.1, 0.25, 0.5, 1.0, 5.0]


def eval_piece(piece: MSMDPiece, model: MERTAlignModel,
               device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        wav   = piece.wav.to(device)
        cols  = piece.score_cols.to(device)
        sim   = model(wav, cols)                        # (T, N_cols)

    path = dtw_decode(sim)                              # (T,) int, col index
    centers = col_centers(piece.N_cols)                 # (N_cols,) x_px
    pred_x  = centers[path]                             # (T,) predicted x_px

    # Audio frames -> times
    frame_times = np.arange(len(path)) / MERT_HZ       # (T,)

    # Build predicted time -> x function (for forward eval)
    # Build GT x -> time function (for inverse)
    gt_times = piece.onset_sec
    gt_x     = piece.strip_x

    # Sort GT by onset time (should already be sorted)
    order = np.argsort(gt_times)
    gt_t_sorted = gt_times[order]
    gt_x_sorted = gt_x[order]

    # For each GT onset: find predicted_x at that time, then invert via GT to get error
    errors = []
    for gt_time, gt_xpos in zip(gt_t_sorted, gt_x_sorted):
        # Predicted position at GT time
        t_frame = int(round(gt_time * MERT_HZ))
        t_frame = min(t_frame, len(pred_x) - 1)
        p_x = pred_x[t_frame]

        # Invert: find when GT path crosses p_x
        # GT path: gt_t_sorted, gt_x_sorted (monotonically increasing x)
        if len(gt_x_sorted) < 2:
            continue
        # Find interval where gt_x crosses p_x
        inv_fn = interpolate.interp1d(
            gt_x_sorted, gt_t_sorted,
            kind='linear', bounds_error=False,
            fill_value=(gt_t_sorted[0], gt_t_sorted[-1])
        )
        predicted_time = float(inv_fn(p_x))
        errors.append(abs(predicted_time - gt_time))

    if not errors:
        return {}
    errors = np.array(errors)
    result = {
        'n_onsets': len(errors),
        'mean_error': float(errors.mean()),
        'median_error': float(np.median(errors)),
    }
    for thr in THRESHOLDS:
        result[f'pct_{thr}s'] = float((errors <= thr).mean() * 100)
    return result


def aggregate(results: list) -> dict:
    keys = [k for k in results[0] if k != 'n_onsets']
    n_total = sum(r['n_onsets'] for r in results)
    agg = {'n_pieces': len(results), 'n_onsets': n_total}
    for k in keys:
        vals = np.array([r[k] for r in results])
        agg[k] = float(vals.mean())
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split',      default='test')
    ap.add_argument('--data_root',  default='data/MSMD/processed')
    ap.add_argument('--device',     default='cuda')
    ap.add_argument('--out',        default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = MERTAlignModel().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    dataset = MSMDDataset(args.split, args.data_root)
    results = []
    for i, piece in enumerate(dataset.pieces):
        r = eval_piece(piece, model, device)
        if r:
            results.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(dataset)} pct@0.5s={r['pct_0.5s']:.1f}%")

    agg = aggregate(results)
    print("\n=== Results ===")
    print(f"Pieces: {agg['n_pieces']}  Onsets: {agg['n_onsets']}")
    for thr in THRESHOLDS:
        print(f"  pct within {thr}s : {agg[f'pct_{thr}s']:.1f}%")
    print(f"  mean error  : {agg['mean_error']:.3f}s")
    print(f"  median error: {agg['median_error']:.3f}s")

    if args.out:
        import json
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(agg, open(args.out, 'w'), indent=2)
        print(f"Saved to {args.out}")


if __name__ == '__main__':
    main()
