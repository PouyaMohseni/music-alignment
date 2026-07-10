"""D3 eval -- same decode/metric pipeline as D1/D2's eval.py (dtw_decode /
oltw_decode / particle_filter_decode, same interpol_c2o timing metric), only
difference is constructing D3Model instead of D1Model. MIDI never touches this
path (MidiEncoder is a train.py-only scaffold, same as D2) -- eval_piece and
main() below only ever import mymodel.d1_align_matrix.dtw's decoders and
mymodel.d1_align_matrix.data's loader, exactly like D1/D2's eval.py.

    python -m mymodel.d3_hybrid.eval --config configs/d3_hybrid.yaml \
        --checkpoint results/d3_hybrid/best_model.pt [--decoder particle_filter]
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mymodel.d3_hybrid.model import D3Model
from mymodel.d1_align_matrix import data as d1data
from mymodel.d1_align_matrix.dtw import dtw_decode, oltw_decode, particle_filter_decode

FPS = 20
THRESHOLDS = [0.05, 0.1, 0.5, 1.0, 5.0]


def eval_piece(model, piece, device, decoder='dtw', band_frac=0.05,
              pf_process_noise_std=3.0, pf_init_std=2.0):
    with torch.no_grad():
        S = model(piece.mert.to(device), piece.strip.to(device))
    S_np = S.float().cpu().numpy()
    if decoder == 'oltw':
        path_cols = oltw_decode(S_np)
    elif decoder == 'particle_filter':
        path_cols = particle_filter_decode(S_np, process_noise_std=pf_process_noise_std,
                                           init_std=pf_init_std)
    else:
        path_cols = dtw_decode(S_np, band_frac=band_frac)
    wd = piece.w_downsample
    diffs = []
    for f in piece.onset_frames:
        if f >= len(path_cols):
            continue
        x_pred = path_cols[f] * wd + wd / 2.0 + piece.add_per_staff
        x_gt = float(piece.gt_cols[f].item()) * wd + wd / 2.0 + piece.add_per_staff
        t_pred = float(piece.interpol_c2o(x_pred))
        t_gt = float(piece.interpol_c2o(x_gt))
        diffs.append(abs(t_pred - t_gt) / FPS)
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/d3_hybrid.yaml')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--decoder', default='dtw', choices=['dtw', 'oltw', 'particle_filter'])
    ap.add_argument('--band_frac', type=float, default=0.05)
    ap.add_argument('--pf_process_noise_std', type=float, default=3.0)
    ap.add_argument('--pf_init_std', type=float, default=2.0)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    band_frac = None if a.band_frac is not None and a.band_frac < 0 else a.band_frac

    cfg = yaml.safe_load(open(a.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dc = cfg['data']

    model_kwargs = {k: v for k, v in cfg['model'].items() if k != 'warm_start_ckpt'}
    model = D3Model(**model_kwargs).to(device)
    ck = torch.load(a.checkpoint, map_location=device)
    model.load_state_dict(ck['model'] if 'model' in ck else ck)
    model.eval()
    print(f'Loaded D3 checkpoint {a.checkpoint} '
          f'({sum(p.numel() for p in model.parameters()):,} params) on {device}', flush=True)

    pieces = d1data.load_split(a.split, dc['processed_root'], dc['cpjku_data'],
                               dc['mert_roots'], dc['scale_factor'],
                               cfg['model']['w_downsample'], limit=a.limit)
    decoder_label = {'dtw': f'DTW (offline, band_frac={band_frac})',
                     'oltw': 'OLTW (causal/online, greedy)',
                     'particle_filter': f'particle filter (causal/online, '
                                        f'process_noise_std={a.pf_process_noise_std}, '
                                        f'init_std={a.pf_init_std})'}[a.decoder]
    print(f'Decoding with {decoder_label} over {len(pieces)} pieces', flush=True)

    all_diffs = []
    for k, piece in enumerate(pieces):
        d = eval_piece(model, piece, device, decoder=a.decoder, band_frac=band_frac,
                       pf_process_noise_std=a.pf_process_noise_std, pf_init_std=a.pf_init_std)
        all_diffs.extend(d)
        if (k + 1) % 10 == 0:
            arr = np.array(all_diffs)
            print(f'  [{k+1}/{len(pieces)}] pct@0.5s so far = '
                  f'{100.0*(arr <= 0.5).mean():.1f}%', flush=True)

    arr = np.array(all_diffs)
    print(f'\n=== D3 ({decoder_label}) on MSMD {a.split} ({len(pieces)} pieces) ===', flush=True)
    for th in THRESHOLDS:
        print(f'  <= {th}s: {100.0*(arr <= th).mean():.1f}%', flush=True)
    print(f'  mean error:   {arr.mean():.3f}s', flush=True)
    print(f'  median error: {np.median(arr):.3f}s', flush=True)
    print(f'  total onsets: {len(arr)}', flush=True)


if __name__ == '__main__':
    main()
