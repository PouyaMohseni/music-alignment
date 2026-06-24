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


def _wav_to_spec_librosa(wav_path: str, spec_params: dict):
    """78-band log-mel spectrogram matching CB_TA's expected input shape.

    Their madmom LogarithmicFilterbank: 12 bands/octave, 60-6000 Hz → ~78 bins.
    We replicate shape with librosa mel (same n_mels=78, same fps/pad).
    Defined at module level so it works in multiprocessing workers.
    """
    import numpy as np
    import librosa
    sr    = spec_params['sample_rate']
    n_fft = spec_params['frame_size']
    fps   = spec_params['fps']
    pad   = spec_params['pad']
    hop   = int(sr / fps)
    y, _  = librosa.load(wav_path, sr=sr, mono=True)
    mel   = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=78, fmin=60.0, fmax=6000.0, power=1.0)
    log_mel = np.log1p(mel).astype(np.float32)
    return np.pad(log_mel, ((0, 0), (pad, 0)), mode='constant')  # (78, T+pad)


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
        import cv2
        score  = cv2.resize(score,
                            (score.shape[1] // scale_factor, score.shape[0] // scale_factor),
                            interpolation=cv2.INTER_AREA)
        coords = coords / scale_factor

    wav_path = os.path.join(path, 'performance', piece_name + '.wav')
    spec     = _wav_to_spec_librosa(wav_path, spec_params)

    onsets     = onset_frames
    coords_new = coords

    # Their interpol_fnc must return [y, x, height] (3 values).
    # They split it as: true_position, height = result[:-1], result[-1]
    # height = adaptive staff height; we use H//2 (our strip is single-line).
    H_strip = sheet.shape[0]
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


def _memory_efficient_eval(network, dataset, device, seq_len=8, threshold=0.5):
    """Eval loop that avoids T×H×W score copies (OOM workaround for large strips).

    ScoreAudioDataset.__getitem__ materialises (T, H, W) score arrays; for a
    3-min piece with a 128×3000 strip this is ~5 GB per piece — instant OOM.
    We access dataset.scores / dataset.performances directly and keep the score
    as a single (1,1,1,H,W) tensor, expanded zero-copy with torch.expand().

    Returns {'frame_differences': {'onset_diffs': [frame_diff, ...]}} which
    matches the format produced by iterate_dataset with eval_only_onsets=True.
    """
    import torch
    import numpy as np
    from audio_conditioned_unet.utils import center_of_mass

    network.eval()
    onset_diffs = []

    pad      = dataset.pad        # 40
    n_frames = dataset.n_frames   # 40
    n_pieces = len(dataset.scores)

    for score_id in range(n_pieces):
        score      = dataset.scores[score_id]        # (H, W) float32
        piece_name = dataset.piece_names[score_id]
        perfs      = dataset.performances[score_id]
        perf       = perfs[list(perfs.keys())[0]]    # our single tempo key (1000)

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
            end          = min(t + seq_len, T_total)
            frame_range  = list(range(t, end))
            sl           = len(frame_range)

            # perf: (sl, 1, 1, 78, n_frames)
            clips = []
            for i in frame_range:
                clip = spec[:, max(0, i - n_frames + 1):i + 1]
                if clip.shape[-1] < n_frames:
                    clip = np.pad(clip, ((0, 0), (n_frames - clip.shape[-1], 0)))
                clips.append(clip)
            perf_t = torch.from_numpy(
                np.array(clips)[:, np.newaxis, np.newaxis]   # (sl,1,1,78,n_frames)
            ).to(device)

            # score: (sl,1,1,H,W) — zero-copy broadcast
            score_batch = score_t.expand(sl, -1, -1, -1, -1)

            with torch.no_grad():
                out    = network(score=score_batch, perf=perf_t, hidden=hidden)
            pred   = out['segmentation']   # (sl, 1, H, W)
            hidden = out.get('hidden')

            for j, i in enumerate(frame_range):
                frame_idx = i - pad   # frame index relative to audio start
                if frame_idx not in onsets_set:
                    continue

                p        = pred[j, 0]                       # (H, W) on device
                p_thresh = (p >= threshold).float()

                if p_thresh.sum() == 0:
                    com_pred = torch.zeros(2, device=device)
                else:
                    com_pred = center_of_mass(p_thresh)

                com_np = com_pred.cpu().numpy()             # [y_pred, x_pred]

                # GT position from interpol_fnc → [y_gt, x_gt, height]
                gt_pos = np.asarray(interpol_fnc(frame_idx))
                x_gt   = float(gt_pos[1])

                # Unroll to global x (single staff: add_per_staff[0] == 0)
                x_pred_g = float(com_np[1]) + float(add_per_staff[0])
                x_gt_g   = x_gt             + float(add_per_staff[0])

                frame_diff = abs(
                    float(interpol_c2o(x_pred_g)) -
                    float(interpol_c2o(x_gt_g))
                )
                onset_diffs.append(frame_diff)
                piece_diffs += 1

            t += sl

        print(f'  [{score_id+1}/{n_pieces}] {piece_name}: {piece_diffs} onset frames evaluated',
              flush=True)

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
    p.add_argument('--split',       default='test')
    p.add_argument('--batch_size',  type=int, default=1,
                   help='Unused (kept for CLI compatibility). Always 1 in memory-efficient loop.')
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

    import torch
    from audio_conditioned_unet.network import ConditionalUNet
    from audio_conditioned_unet.dataset import load_dataset

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

    param_path  = cpjku_root / 'models' / a.model / 'best_model.pt'
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

    split_file, available = build_split_file(a.processed, a.cpjku_data, a.split)
    if not available:
        raise RuntimeError('No converted pieces found. Run: python -m mymodel.cpjku_adapter.convert')

    # Patch load_piece with our module-level function (picklable for pool.map)
    import audio_conditioned_unet.dataset as _ds
    _ds.load_piece = _patched_load_piece

    n_frames = network.perf_encoder.n_input_frames
    print(f'Loading dataset ({a.split}, scale_factor={a.scale_factor})...', flush=True)
    dataset = load_dataset(a.cpjku_data, config, n_frames=n_frames,
                           split_file=split_file, scale_factor=a.scale_factor)

    print(f'Running eval (seq_len={a.seq_len}, device={device})...', flush=True)
    stats = _memory_efficient_eval(network, dataset, device=str(device), seq_len=a.seq_len)

    import numpy as np
    frame_diffs  = stats['frame_differences']
    onset_diffs  = np.array(frame_diffs['onset_diffs']) / config['spectrogram_params']['fps']
    total_onsets = len(onset_diffs)

    print(f'\n=== CPJKU {a.model} on MSMD {a.split} ({len(available)} pieces) ===')
    for th in [0.05, 0.1, 0.5, 1.0, 5.0]:
        pct = 100 * np.sum(onset_diffs <= th) / total_onsets
        print(f'  <= {th}s: {pct:.1f}%')
    print(f'  mean error:   {onset_diffs.mean():.3f}s')
    print(f'  median error: {np.median(onset_diffs):.3f}s')
    print(f'  total onsets: {total_onsets}')


if __name__ == '__main__':
    main()
