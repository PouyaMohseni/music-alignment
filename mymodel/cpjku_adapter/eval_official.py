"""Run the official CPJKU CB_TA eval on our MSMD test split.

Uses their exact network and metric (% of onset frames within time threshold).
Differences from their eval_model.py:
  - librosa spectrogram instead of madmom (madmom incompatible with Python >=3.11)
  - memory-efficient eval loop: score stored once as (1,1,1,H,W) and expanded
    zero-copy per chunk — avoids OOM from ScoreAudioDataset.__getitem__ which
    materialises (T, H, W) copies of the full score for every piece.

    python -m mymodel.cpjku_adapter.eval_official \
        --cpjku_root  third_party/cpjku_unet \
        --cpjku_data  data/MSMD/cpjku_fmt \
        --split_file  data/MSMD/cpjku_fmt/split_test.yaml \
        --model       CB_TA
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def _build_log_filterbank(sr: int, n_fft: int,
                          fmin: float = 60.0, fmax: float = 6000.0,
                          bands_per_octave: int = 12) -> 'np.ndarray':
    """Build triangular log-frequency filterbank matching madmom LogarithmicFilterbank.

    Parameters match their spectrogram_processor() call exactly:
        FilteredSpectrogramProcessor(LogarithmicFilterbank, num_bands=12,
                                     fmin=60, fmax=6000,
                                     norm_filters=True, unique_filters=False)

    Returns filterbank matrix (n_filters, n_fft//2+1) float32.
    """
    import numpy as np
    n_fft_bins = n_fft // 2 + 1

    # Center frequencies: fmin * 2^(k / bands_per_octave)
    # Include one extra on each side for triangular boundaries.
    num_octaves = int(np.ceil(np.log2(fmax / fmin)))
    n_total = bands_per_octave * num_octaves + 1
    all_centers = fmin * 2 ** (np.arange(-1, n_total + 1) / bands_per_octave)

    mask    = (all_centers >= fmin) & (all_centers <= fmax)
    centers = all_centers[mask]

    def freq_to_bin(f):
        return int(round(f * n_fft / sr))

    # Boundary bin below fmin and above fmax (for outermost filter slopes)
    lo_bin = freq_to_bin(all_centers[np.searchsorted(all_centers, fmin) - 1])
    hi_bin = freq_to_bin(all_centers[np.searchsorted(all_centers, fmax, side='right')])

    boundary_bins = (
        [np.clip(lo_bin, 0, n_fft_bins - 1)]
        + [np.clip(freq_to_bin(c), 0, n_fft_bins - 1) for c in centers]
        + [np.clip(hi_bin, 0, n_fft_bins - 1)]
    )

    # Build filters — skip degenerate ones (left == center == right in bin space).
    # madmom drops these automatically; they arise when adjacent center freqs
    # fall in the same STFT bin (happens at lowest frequencies with 10.8 Hz/bin).
    rows = []
    for k in range(len(centers)):
        left   = boundary_bins[k]
        center = boundary_bins[k + 1]
        right  = boundary_bins[k + 2]

        if left == center == right:
            continue  # degenerate — zero area after normalisation

        row = np.zeros(n_fft_bins, dtype=np.float64)

        if center > left:
            b = np.arange(left, center + 1)
            row[left:center + 1] = (b - left) / (center - left)
        else:
            row[center] = 1.0

        if right > center:
            b = np.arange(center, right + 1)
            row[center:right + 1] = (right - b) / (right - center)

        # norm_filters=True: divide by filter area
        s = row.sum()
        if s > 0:
            row /= s

        rows.append(row)

    return np.array(rows, dtype=np.float32)  # (n_valid_filters, n_fft_bins)


def _wav_to_spec_logfilter(wav_path: str, spec_params: dict) -> 'np.ndarray':
    """Approximate madmom's exact pipeline:
      SignalProcessor → FramedSignalProcessor → FilteredSpectrogramProcessor
        (LogarithmicFilterbank, num_bands=12, fmin=60, fmax=6000,
         norm_filters=True, unique_filters=False)
      → LogarithmicSpectrogramProcessor  (= log10(1 + spec))

    Uses librosa STFT + numpy triangular filterbank.
    Key differences vs our old _wav_to_spec_librosa:
      - Logarithmic (constant-Q-like) filterbank instead of mel
      - log10(1+x) instead of ln(1+x)  ← what their model trained on
      - center=False to match madmom's FramedSignalProcessor
    """
    import numpy as np, librosa
    sr    = spec_params['sample_rate']
    n_fft = spec_params['frame_size']
    fps   = spec_params['fps']
    pad   = spec_params['pad']
    hop   = int(sr / fps)

    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    # STFT magnitude — center=False matches madmom's non-centered frames
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop,
                            window='hann', center=False))  # (n_fft//2+1, T)

    fb      = _build_log_filterbank(sr, n_fft)           # (n_filters, n_fft//2+1)
    filt    = fb @ D                                       # (n_filters, T)
    log_s   = np.log10(1.0 + filt).astype(np.float32)    # log10 matches madmom

    return np.pad(log_s, ((0, 0), (pad, 0)), mode='constant')  # (n_filters, T+pad)


def _patched_load_piece(params):
    """Module-level so multiprocessing pool.map can pickle it.

    Stubs are re-applied here because pool.map workers are forked/spawned
    without the parent's sys.modules patches.
    """
    import os, sys, types
    import numpy as np
    from scipy import interpolate

    # Re-stub cv2 and madmom in each worker process
    if 'cv2' not in sys.modules:
        try:
            import cv2  # noqa: F401
        except ImportError:
            _fake = types.ModuleType('cv2')
            _fake.resize = None; _fake.INTER_AREA = 0
            sys.modules['cv2'] = _fake

    for _mod in ['madmom', 'madmom.io', 'madmom.io.midi', 'madmom.audio',
                 'madmom.audio.signal', 'madmom.audio.spectrogram', 'madmom.processors']:
        sys.modules.setdefault(_mod, types.ModuleType(_mod))

    i, path, piece_name = params['i'], params['path'], params['piece_name']
    spec_params  = params['spectrogram_params']
    scale_factor = params.get('scale_factor', 1)

    npz = np.load(os.path.join(path, 'score', piece_name + '.npz'), allow_pickle=True)
    sheet        = npz['sheet']
    coords       = npz['coords'].astype(np.float32)
    onset_frames = npz['onset_frames']

    score = 1 - sheet.astype(np.float32) / 255.
    if scale_factor != 1:
        new_h = sheet.shape[0] // scale_factor
        new_w = sheet.shape[1] // scale_factor
        try:
            import cv2 as _cv2
            if not callable(_cv2.resize):
                raise TypeError('cv2 stub')
            score = _cv2.resize(score, (new_w, new_h), interpolation=_cv2.INTER_AREA)
        except (ImportError, TypeError):
            # Fallback: PIL — available everywhere without module load
            from PIL import Image as _Image
            resized = np.array(
                _Image.fromarray(sheet).resize((new_w, new_h), _Image.LANCZOS)
            )
            score = 1 - resized.astype(np.float32) / 255.
        coords = coords / scale_factor

    wav_path = os.path.join(path, 'performance', piece_name + '.wav')
    spec     = _wav_to_spec_logfilter(wav_path, spec_params)

    onsets     = onset_frames
    coords_new = coords

    # Their interpol_fnc must return [y, x, height] (3 values).
    # They split it as: true_position, height = result[:-1], result[-1]
    # height = adaptive staff height; we scale H//2 by scale_factor so GT
    # rectangle covers the strip centre after downscaling.
    H_strip = sheet.shape[0] // scale_factor if scale_factor != 1 else sheet.shape[0]
    height_col = np.full((len(coords_new), 1), H_strip // 2, dtype=np.float32)
    coords_3 = np.concatenate([coords_new, height_col], axis=1)  # (N, 3)

    interpol_fnc = interpolate.interp1d(
        onsets, coords_3.T, kind='previous', bounds_error=False,
        fill_value=(coords_3[0, :], coords_3[-1, :]))

    unrolled_x = coords_new[:, 1]
    interpol_c2o = interpolate.interp1d(
        unrolled_x, onsets, kind='previous', bounds_error=False,
        fill_value=(onsets[0], onsets[-1]))

    staff_coords   = sorted(np.unique(coords_new[:, 0]))
    add_per_staff  = np.array([0] * len(staff_coords))

    perf = {
        1000: {
            'interpol_fnc':  interpol_fnc,
            'spec':          spec,
            'onsets':        onsets,
            'interpol_c2o':  interpol_c2o,
            'add_per_staff': [staff_coords, add_per_staff],
        }
    }
    return i, score, piece_name, perf


def _memory_efficient_eval(network, piece_names, cpjku_data, config,
                           device, seq_len=8, scale_factor=1,
                           n_frames=40, threshold=0.5):
    """Eval loop: loads each piece sequentially (no multiprocessing pool).

    Their load_dataset uses multiprocessing.Pool with fork; forked workers
    call librosa → numpy/BLAS and inherit locked BLAS thread pools, causing
    an indefinite deadlock on SLURM nodes.  We load each piece one at a time
    in the main process to avoid the deadlock entirely.

    Returns {'frame_differences': {'onset_diffs': [frame_diff, ...]}} matching
    the format produced by iterate_dataset with eval_only_onsets=True.
    """
    import torch
    import numpy as np
    from audio_conditioned_unet.utils import center_of_mass

    network.eval()
    onset_diffs = []
    pad = config['spectrogram_params']['pad']

    for piece_idx, piece_name in enumerate(piece_names):
        print(f'[{piece_idx+1}/{len(piece_names)}] Loading {piece_name}...', flush=True)
        params = {
            'i':                 piece_idx,
            'path':              cpjku_data,
            'piece_name':        piece_name,
            'spectrogram_params': config['spectrogram_params'],
            'scale_factor':      scale_factor,
        }
        _, score, _, perf_dict = _patched_load_piece(params)
        perf = perf_dict[list(perf_dict.keys())[0]]   # single tempo key (1000)

        spec             = perf['spec']              # (78, T_total)
        onsets_set       = set(perf['onsets'].tolist())
        interpol_fnc     = perf['interpol_fnc']
        interpol_c2o     = perf['interpol_c2o']
        staff_coords, add_per_staff = perf['add_per_staff']

        T_total = spec.shape[-1]

        # Single score tile — (1,1,1,H,W), expanded zero-copy per chunk
        score_t = torch.from_numpy(
            score[np.newaxis, np.newaxis, np.newaxis]
        ).to(device)

        hidden      = None
        t           = pad
        piece_diffs = 0

        while t < T_total:
            end         = min(t + seq_len, T_total)
            frame_range = list(range(t, end))
            sl          = len(frame_range)

            clips = []
            for i in frame_range:
                clip = spec[:, max(0, i - n_frames + 1):i + 1]
                if clip.shape[-1] < n_frames:
                    clip = np.pad(clip, ((0, 0), (n_frames - clip.shape[-1], 0)))
                clips.append(clip)
            perf_t = torch.from_numpy(
                np.array(clips)[:, np.newaxis, np.newaxis]
            ).to(device)

            score_batch = score_t.expand(sl, -1, -1, -1, -1)

            with torch.no_grad():
                out  = network(score=score_batch, perf=perf_t, hidden=hidden)
            pred   = out['segmentation']
            hidden = out.get('hidden')

            for j, i in enumerate(frame_range):
                frame_idx = i - pad
                if frame_idx not in onsets_set:
                    continue

                p        = pred[j, 0]
                p_thresh = (p >= threshold).float()
                com_pred = (center_of_mass(p_thresh)
                            if p_thresh.sum() > 0
                            else torch.zeros(2, device=device))
                com_np   = com_pred.cpu().numpy()

                gt_pos   = np.asarray(interpol_fnc(frame_idx))
                x_gt     = float(gt_pos[1])

                x_pred_g = float(com_np[1]) + float(add_per_staff[0])
                x_gt_g   = x_gt             + float(add_per_staff[0])

                frame_diff = abs(float(interpol_c2o(x_pred_g)) -
                                 float(interpol_c2o(x_gt_g)))
                onset_diffs.append(frame_diff)
                piece_diffs += 1

            t += sl

        print(f'  -> {piece_diffs} onset frames evaluated', flush=True)

    return {'frame_differences': {'onset_diffs': onset_diffs}}


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
    p.add_argument('--param_path',  default=None,
                   help='Path to a trained best_model.pt. If set, evaluates OUR trained '
                        'model instead of the bundled pretrained --model. '
                        'net_config.json must sit next to it (or pass --net_config).')
    p.add_argument('--net_config',  default=None,
                   help='Path to net_config.json for --param_path (default: alongside it).')
    p.add_argument('--split',       default='test')
    p.add_argument('--batch_size',  type=int, default=1,
                   help='Unused (kept for CLI compatibility). Always 1 in memory-efficient loop.')
    p.add_argument('--seq_len',     type=int, default=128)
    p.add_argument('--scale_factor', type=int, default=1,
                   help='Score downscale factor. Use 3 to match training; 1 for full-res strips.')
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

    import torch
    from audio_conditioned_unet.network import ConditionalUNet

    config = {
        'spectrogram_params': {
            'sample_rate': 22050,
            'frame_size': 2048,
            'fps': 20,
            'pad': 40,
        },
        'gt_width': 10,
        'real_perf': True,
        'tempo_factors': [1000],
        'sf_path': '',
    }

    if a.param_path is not None:
        # Evaluate OUR trained model.
        param_path  = Path(a.param_path)
        config_path = (Path(a.net_config) if a.net_config
                       else param_path.parent / 'net_config.json')
        model_label = f'trained:{param_path.parent.name}'
    else:
        # Evaluate their bundled pretrained model.
        param_path  = cpjku_root / 'models' / a.model / 'best_model.pt'
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
    network.load_state_dict(torch.load(param_path, map_location='cpu'))
    network.to(device).eval()
    print(f'Loaded {model_label} ({sum(p.numel() for p in network.parameters()):,} params) '
          f'on {device}', flush=True)

    split_file, available = build_split_file(a.processed, a.cpjku_data, a.split)
    if not available:
        raise RuntimeError('No converted pieces found. Run: python -m mymodel.cpjku_adapter.convert')

    n_frames = network.perf_encoder.n_input_frames

    print(f'Running eval on {len(available)} pieces '
          f'(seq_len={a.seq_len}, scale_factor={a.scale_factor}, device={device})...',
          flush=True)
    stats = _memory_efficient_eval(
        network, available, a.cpjku_data, config,
        device=str(device), seq_len=a.seq_len,
        scale_factor=a.scale_factor, n_frames=n_frames)

    import numpy as np
    frame_diffs  = stats['frame_differences']
    onset_diffs  = np.array(frame_diffs['onset_diffs']) / config['spectrogram_params']['fps']
    total_onsets = len(onset_diffs)

    print(f'\n=== CPJKU {model_label} on MSMD {a.split} ({len(available)} pieces) ===')
    for th in [0.05, 0.1, 0.5, 1.0, 5.0]:
        pct = 100 * np.sum(onset_diffs <= th) / total_onsets
        print(f'  <= {th}s: {pct:.1f}%')
    print(f'  mean error:   {onset_diffs.mean():.3f}s')
    print(f'  median error: {np.median(onset_diffs):.3f}s')
    print(f'  total onsets: {total_onsets}')


if __name__ == '__main__':
    main()
