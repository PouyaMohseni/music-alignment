"""Run the official CPJKU CB_TA eval on our MSMD test split.

Uses their exact network, their exact eval loop, their exact metric
(% of onset frames within time threshold). Only difference: librosa
spectrogram instead of madmom (madmom incompatible with Python ≥3.11).

    python -m mymodel.cpjku_adapter.eval_official \
        --cpjku_root  third_party/cpjku_unet \
        --cpjku_data  data/MSMD/cpjku_fmt \
        --split_file  data/MSMD/cpjku_fmt/split_test.yaml \
        --model       CB_TA
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def build_split_file(processed_root: str, cpjku_data: str, split: str):
    """Write a YAML split file listing the piece IDs for the given split."""
    import yaml
    proc = Path(processed_root)
    splits = json.load(open(proc / 'splits.json'))
    piece_ids = splits.get(split, [])

    # Only include pieces that have been converted
    score_dir = Path(cpjku_data) / 'score'
    available = [p for p in piece_ids if (score_dir / f'{p}.npz').exists()]
    print(f'Split={split}: {len(available)}/{len(piece_ids)} pieces converted', flush=True)

    split_path = Path(cpjku_data) / f'split_{split}.yaml'
    with open(split_path, 'w') as f:
        yaml.dump({'files': available}, f)
    return str(split_path), available


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cpjku_root',  default='third_party/cpjku_unet')
    p.add_argument('--cpjku_data',  default='data/MSMD/cpjku_fmt')
    p.add_argument('--processed',   default='data/MSMD/processed')
    p.add_argument('--model',       default='CB_TA',
                   choices=['CB_TA', 'CB_noTA', 'FB_TA', 'FB_noTA', 'NTC_TA', 'NTC_noTA'])
    p.add_argument('--split',       default='test')
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--seq_len',     type=int, default=128)
    p.add_argument('--scale_factor', type=int, default=1,
                   help='Score downscale factor. Use 1 for our strips (already small).')
    a = p.parse_args()

    cpjku_root = Path(a.cpjku_root).resolve()
    if not cpjku_root.exists():
        raise FileNotFoundError(f'CPJKU repo not found: {cpjku_root}\n'
                                'Run: git submodule update --init third_party/cpjku_unet')

    # Add CPJKU repo to path BEFORE patching
    if str(cpjku_root) not in sys.path:
        sys.path.insert(0, str(cpjku_root))

    # Patch madmom → librosa BEFORE importing CPJKU modules
    from mymodel.cpjku_adapter import madmom_compat
    madmom_compat.patch()

    # Now import CPJKU modules
    import torch
    from audio_conditioned_unet.network import ConditionalUNet
    from audio_conditioned_unet.dataset import load_dataset, iterate_dataset, NonSequentialDatasetWrapper
    from audio_conditioned_unet.utils import load_game_config

    # Config: use their msmd.yaml but with real_perf=True (we have wav files, no MIDI synthesis)
    config = {
        'spectrogram_params': {
            'sample_rate': 22050,
            'frame_size': 2048,
            'fps': 20,
            'pad': 40,
        },
        'gt_width': 10,
        'real_perf': True,
        'tempo_factors': [1000],   # dummy — not used with real_perf
        'sf_path': '',             # not needed with real_perf
    }

    # Their model checkpoint
    param_path = cpjku_root / 'models' / a.model / 'best_model.pt'
    config_path = cpjku_root / 'models' / a.model / 'net_config.json'
    if not param_path.exists():
        raise FileNotFoundError(f'Model not found: {param_path}')

    with open(config_path) as f:
        net_config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = ConditionalUNet(net_config)
    network.load_state_dict(torch.load(param_path, map_location='cpu'))
    network.to(device).eval()
    print(f'Loaded {a.model} ({sum(p.numel() for p in network.parameters()):,} params) on {device}',
          flush=True)

    # Build split file
    split_file, available = build_split_file(a.processed, a.cpjku_data, a.split)
    if not available:
        raise RuntimeError('No converted pieces found. Run: python -m mymodel.cpjku_adapter.convert')

    # Patch their load_performance to handle our wav naming convention:
    # They expect: performance/<piece>_<tempo>.wav  but we have: performance/<piece>.wav
    import audio_conditioned_unet.utils as _utils
    _orig_load_perf = _utils.load_performance

    def _patched_load_performance(path, piece, spec_params, coords, coord2onset,
                                   sf_path, tempo_factor=1., real_perf=False, transpose=0):
        import os
        from scipy import interpolate
        import copy
        # Load pre-saved coords and onset frames from our NPZ
        npz = __import__('numpy').load(os.path.join(path, 'score', piece + '.npz'), allow_pickle=True)
        onset_frames = npz['onset_frames']          # (N,) int64
        coords_orig  = npz['coords']                # (N, 2) float32

        # Build spectrogram from our wav
        wav_path = os.path.join(path, 'performance', piece + '.wav')
        spec = _utils.wav_to_spec_otf(wav_path, spec_params)  # (n_bands, T+pad)

        onsets     = onset_frames
        coords_new = coords_orig
        interpol_fnc = interpolate.interp1d(
            onsets, coords_new.T, kind='previous', bounds_error=False,
            fill_value=(coords_new[0, :], coords_new[-1, :]))

        # coord2onset for interpol_c2o
        unrolled_x = coords_new[:, 1]
        interpol_c2o = interpolate.interp1d(
            unrolled_x, onsets, kind='previous', bounds_error=False,
            fill_value=(onsets[0], onsets[-1]))

        staff_coords = [0]
        add_per_staff = [0]

        return spec, onsets, coords_new, interpol_fnc

    _utils.load_performance = _patched_load_performance

    # Also patch load_piece in dataset to pass add_per_staff correctly
    import audio_conditioned_unet.dataset as _ds
    _orig_load_piece = _ds.load_piece

    def _patched_load_piece(params):
        import os, numpy as np
        from scipy import interpolate
        i, path, piece_name = params['i'], params['path'], params['piece_name']
        spec_params = params['spectrogram_params']

        npz = np.load(os.path.join(path, 'score', piece_name + '.npz'), allow_pickle=True)
        sheet     = npz['sheet']
        coords    = npz['coords'].astype(np.float32)
        onset_frames = npz['onset_frames']

        from audio_conditioned_unet.utils import load_score, wav_to_spec_otf
        scale_factor = params.get('scale_factor', 1)

        # Scale score image and coords
        import cv2
        score = 1 - sheet.astype(np.float32) / 255.
        if scale_factor != 1:
            score = cv2.resize(score, (score.shape[1] // scale_factor, score.shape[0] // scale_factor),
                               interpolation=cv2.INTER_AREA)
            coords = coords / scale_factor

        wav_path = os.path.join(path, 'performance', piece_name + '.wav')
        spec = wav_to_spec_otf(wav_path, spec_params)  # (n_bands, T+pad)

        onsets = onset_frames
        coords_new = coords

        interpol_fnc = interpolate.interp1d(
            onsets, coords_new.T, kind='previous', bounds_error=False,
            fill_value=(coords_new[0, :], coords_new[-1, :]))

        unrolled_x = coords_new[:, 1]
        interpol_c2o = interpolate.interp1d(
            unrolled_x, onsets, kind='previous', bounds_error=False,
            fill_value=(onsets[0], onsets[-1]))

        staff_coords = sorted(np.unique(coords_new[:, 0]))
        add_per_staff = np.array([0] * len(staff_coords))

        perf = {
            1000: {
                'interpol_fnc': interpol_fnc,
                'spec': spec,
                'onsets': onsets,
                'interpol_c2o': interpol_c2o,
                'add_per_staff': [staff_coords, add_per_staff],
            }
        }
        return i, score, piece_name, perf

    _ds.load_piece = _patched_load_piece

    # Load dataset
    n_frames = network.perf_encoder.n_input_frames
    print(f'Loading dataset ({a.split}, scale_factor={a.scale_factor})...', flush=True)

    # Patch config to use our tempo key
    config['tempo_factors'] = [1000]

    dataset = load_dataset(a.cpjku_data, config, n_frames=n_frames,
                           split_file=split_file, scale_factor=a.scale_factor)

    wrapped = NonSequentialDatasetWrapper(dataset)

    print(f'Running eval (batch_size={a.batch_size}, seq_len={a.seq_len})...', flush=True)
    stats = iterate_dataset(network, None, wrapped,
                            batch_size=a.batch_size, seq_len=a.seq_len,
                            device=str(device), train=False,
                            average_stats=False,
                            eval_center_of_mass=True,
                            eval_only_onsets=True)

    # Print results matching their eval_model.py --eval_onsets output
    frame_diffs = stats['frame_differences']
    thresholds  = [0.05, 0.1, 0.5, 1.0, 5.0]
    fps = config['spectrogram_params']['fps']

    onset_diffs  = np.array(frame_diffs['onset_diffs']) / fps
    total_onsets = len(onset_diffs)

    print(f'\n=== CPJKU {a.model} on MSMD {a.split} ({len(available)} pieces) ===')
    for th in thresholds:
        pct = 100 * np.sum(onset_diffs <= th) / total_onsets
        print(f'  <= {th}s: {pct:.1f}%')
    print(f'  mean error: {onset_diffs.mean():.3f}s')
    print(f'  median error: {np.median(onset_diffs):.3f}s')


if __name__ == '__main__':
    import numpy as np
    main()
