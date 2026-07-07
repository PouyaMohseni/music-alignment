"""C5: test-time per-piece calibration eval. Same official CPJKU eval
pipeline as eval_official.py, but before scoring each piece, runs a few
gradient steps against that piece's OWN first `--calib_seconds` of
audio+score (extensions/decode/test_time_calibration.py), then scores
ONLY the remainder of the piece with the calibrated weights. Weights are
reset to their pre-calibration state before the next piece.

    python -m mymodel.cpjku_adapter.eval_test_time_calibration \
        --cpjku_root  third_party/cpjku_unet \
        --cpjku_data  data/MSMD/cpjku_fmt \
        --processed   data/MSMD/processed \
        --model       CB_TA \
        --split       test \
        --calib_seconds 8.0 --calib_steps 15 --calib_lr 1e-3
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from mymodel.cpjku_adapter.eval_official import (
    _patched_load_piece, build_split_file,
)


def _calibrated_eval(network, piece_names, cpjku_data, config, device,
                     seq_len, scale_factor, n_frames,
                     calib_seconds, calib_steps, calib_lr, threshold=0.5):
    from extensions.decode.test_time_calibration import calibrate_and_infer_piece

    pad = config['spectrogram_params']['pad']
    gt_width = config['gt_width']
    onset_diffs = []

    for piece_idx, piece_name in enumerate(piece_names):
        print(f'[{piece_idx+1}/{len(piece_names)}] Loading {piece_name}...', flush=True)
        params = {
            'i': piece_idx, 'path': cpjku_data, 'piece_name': piece_name,
            'spectrogram_params': config['spectrogram_params'],
            'scale_factor': scale_factor,
        }
        _, score, _, perf_dict = _patched_load_piece(params)
        perf = perf_dict[list(perf_dict.keys())[0]]

        onsets_set = set(perf['onsets'].tolist())
        # perf['add_per_staff'] is [staff_coords, add_per_staff_array] --
        # must unpack before use (see test_time_calibration.py's expected
        # signature / smoke_test_c5.py's comment for why).
        _staff_coords, add_per_staff = perf['add_per_staff']
        diffs, calib_init_loss, calib_final_loss = calibrate_and_infer_piece(
            network, score, perf['spec'], perf['interpol_fnc'],
            perf['interpol_c2o'], add_per_staff, onsets_set,
            pad=pad, gt_width=gt_width, n_frames=n_frames, device=device,
            calib_seconds=calib_seconds, fps=config['spectrogram_params']['fps'],
            num_steps=calib_steps, lr=calib_lr, seq_len=seq_len, threshold=threshold)

        print(f'  calib_loss: {calib_init_loss} -> {calib_final_loss}  '
              f'-> {len(diffs)} onset frames evaluated (post-calibration segment only)',
              flush=True)
        onset_diffs.extend(diffs)

    return {'frame_differences': {'onset_diffs': onset_diffs}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cpjku_root', default='third_party/cpjku_unet')
    p.add_argument('--cpjku_data', default='data/MSMD/cpjku_fmt')
    p.add_argument('--processed', default='data/MSMD/processed')
    p.add_argument('--model', default='CB_TA',
                   choices=['CB_TA', 'CB_noTA', 'FB_TA', 'FB_noTA', 'NTC_TA', 'NTC_noTA'])
    p.add_argument('--param_path', default=None)
    p.add_argument('--net_config', default=None)
    p.add_argument('--split', default='test')
    p.add_argument('--seq_len', type=int, default=128)
    p.add_argument('--scale_factor', type=int, default=1)
    p.add_argument('--calib_seconds', type=float, default=8.0)
    p.add_argument('--calib_steps', type=int, default=15)
    p.add_argument('--calib_lr', type=float, default=1e-3)
    a = p.parse_args()

    cpjku_root = Path(a.cpjku_root).resolve()
    if not cpjku_root.exists():
        raise FileNotFoundError(f'CPJKU repo not found: {cpjku_root}')
    if str(cpjku_root) not in sys.path:
        sys.path.insert(0, str(cpjku_root))

    from mymodel.cpjku_adapter import madmom_compat
    madmom_compat.patch()

    import torch
    from audio_conditioned_unet.network import ConditionalUNet

    config = {
        'spectrogram_params': {'sample_rate': 22050, 'frame_size': 2048, 'fps': 20, 'pad': 40},
        'gt_width': 10, 'real_perf': True, 'tempo_factors': [1000], 'sf_path': '',
    }

    if a.param_path is not None:
        param_path = Path(a.param_path)
        config_path = Path(a.net_config) if a.net_config else param_path.parent / 'net_config.json'
        model_label = f'trained:{param_path.parent.name}'
    else:
        param_path = cpjku_root / 'models' / a.model / 'best_model.pt'
        config_path = cpjku_root / 'models' / a.model / 'net_config.json'
        model_label = a.model

    if not param_path.exists():
        raise FileNotFoundError(f'Model not found: {param_path}')
    if not config_path.exists():
        raise FileNotFoundError(f'net_config.json not found: {config_path}')

    with open(config_path) as f:
        net_config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = ConditionalUNet(net_config)
    state_dict = torch.load(param_path, map_location='cpu')
    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f'Ignoring extension-only checkpoint keys: {unexpected}', flush=True)
    if missing:
        raise RuntimeError(f'Checkpoint is missing base-network keys: {missing}')
    network.to(device).eval()
    print(f'Loaded {model_label} ({sum(p.numel() for p in network.parameters()):,} params) '
          f'on {device}', flush=True)

    split_file, available = build_split_file(a.processed, a.cpjku_data, a.split)
    if not available:
        raise RuntimeError('No converted pieces found. Run: python -m mymodel.cpjku_adapter.convert')

    n_frames = network.perf_encoder.n_input_frames

    print(f'Running C5 calibrated eval on {len(available)} pieces '
          f'(calib_seconds={a.calib_seconds}, calib_steps={a.calib_steps}, '
          f'calib_lr={a.calib_lr}, seq_len={a.seq_len}, scale_factor={a.scale_factor}, '
          f'device={device})...', flush=True)
    stats = _calibrated_eval(
        network, available, a.cpjku_data, config, device=str(device),
        seq_len=a.seq_len, scale_factor=a.scale_factor, n_frames=n_frames,
        calib_seconds=a.calib_seconds, calib_steps=a.calib_steps, calib_lr=a.calib_lr)

    import numpy as np
    onset_diffs = np.array(stats['frame_differences']['onset_diffs']) / config['spectrogram_params']['fps']
    total_onsets = len(onset_diffs)

    print(f'\n=== CPJKU {model_label} + C5 test-time calibration on MSMD {a.split} '
          f'({len(available)} pieces, post-calibration segment only) ===')
    for th in [0.05, 0.1, 0.5, 1.0, 5.0]:
        pct = 100 * np.sum(onset_diffs <= th) / total_onsets
        print(f'  <= {th}s: {pct:.1f}%')
    print(f'  mean error:   {onset_diffs.mean():.3f}s')
    print(f'  median error: {np.median(onset_diffs):.3f}s')
    print(f'  total onsets: {total_onsets}')


if __name__ == '__main__':
    main()
