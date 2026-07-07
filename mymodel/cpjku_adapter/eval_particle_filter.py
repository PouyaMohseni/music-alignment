"""C3 eval: identical to eval_official.py (same data loading, spectrogram,
network, checkpoint loading) except the decode step -- instead of
threshold-then-center-of-mass per frame, decode with a Bayesian particle
filter (extensions/decode/particle_filter.py) that tracks x-position across
frames using each frame's heatmap as an observation likelihood plus a
constant-velocity motion prior. No retraining: runs against CB_TA's bundled
pretrained model or any already-trained checkpoint unchanged.

    python -m mymodel.cpjku_adapter.eval_particle_filter \
        --cpjku_root  third_party/cpjku_unet \
        --cpjku_data  data/MSMD/cpjku_fmt \
        --processed   data/MSMD/processed \
        --model       CB_TA \
        --split       test
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mymodel.cpjku_adapter.eval_official import _patched_load_piece, build_split_file


def _particle_filter_eval(network, piece_names, cpjku_data, config,
                          device, seq_len=8, scale_factor=1, n_frames=40,
                          pf_kwargs=None):
    """Same structure as eval_official._memory_efficient_eval, but decodes
    with ParticleFilterXTracker instead of threshold+center-of-mass."""
    import torch
    import numpy as np
    from extensions.decode.particle_filter import ParticleFilterXTracker, heatmap_to_x_marginal

    pf_kwargs = pf_kwargs or {}
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

        score_t = torch.from_numpy(
            score[np.newaxis, np.newaxis, np.newaxis]
        ).to(device)

        hidden      = None
        t           = pad
        piece_diffs = 0
        tracker     = ParticleFilterXTracker(**pf_kwargs)   # fresh state per piece

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
                # Particle filter must step on EVERY frame (not just onset
                # frames) since it's a sequential tracker -- skipping frames
                # would break the constant-velocity motion model's timestep
                # assumption. Only record the error at actual onset frames,
                # same as eval_official.py.
                heatmap_np = pred[j, 0].detach().cpu().numpy()
                x_marginal = heatmap_to_x_marginal(heatmap_np)
                x_pred = tracker.step(x_marginal)

                frame_idx = i - pad
                if frame_idx not in onsets_set:
                    continue

                gt_pos   = np.asarray(interpol_fnc(frame_idx))
                x_gt     = float(gt_pos[1])

                x_pred_g = x_pred           + float(add_per_staff[0])
                x_gt_g   = x_gt             + float(add_per_staff[0])

                frame_diff = abs(float(interpol_c2o(x_pred_g)) -
                                 float(interpol_c2o(x_gt_g)))
                onset_diffs.append(frame_diff)
                piece_diffs += 1

            t += sl

        print(f'  -> {piece_diffs} onset frames evaluated', flush=True)

    return {'frame_differences': {'onset_diffs': onset_diffs}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cpjku_root',  default='third_party/cpjku_unet')
    p.add_argument('--cpjku_data',  default='data/MSMD/cpjku_fmt')
    p.add_argument('--processed',   default='data/MSMD/processed')
    p.add_argument('--model',       default='CB_TA',
                   choices=['CB_TA', 'CB_noTA', 'FB_TA', 'FB_noTA', 'NTC_TA', 'NTC_noTA'])
    p.add_argument('--param_path',  default=None,
                   help='Path to a trained best_model.pt. If set, evaluates OUR trained '
                        'model instead of the bundled pretrained --model.')
    p.add_argument('--net_config',  default=None)
    p.add_argument('--split',       default='test')
    p.add_argument('--seq_len',     type=int, default=128)
    p.add_argument('--scale_factor', type=int, default=1)
    p.add_argument('--pf_n_particles', type=int, default=200)
    p.add_argument('--pf_process_noise_std', type=float, default=3.0)
    p.add_argument('--pf_velocity_ema_alpha', type=float, default=0.3)
    a = p.parse_args()

    cpjku_root = Path(a.cpjku_root).resolve()
    if not cpjku_root.exists():
        raise FileNotFoundError(f'CPJKU repo not found: {cpjku_root}\n'
                                'Run: git submodule update --init third_party/cpjku_unet')

    if str(cpjku_root) not in sys.path:
        sys.path.insert(0, str(cpjku_root))

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
        param_path  = Path(a.param_path)
        config_path = (Path(a.net_config) if a.net_config
                       else param_path.parent / 'net_config.json')
        model_label = f'trained:{param_path.parent.name}'
    else:
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

    pf_kwargs = dict(n_particles=a.pf_n_particles,
                      process_noise_std=a.pf_process_noise_std,
                      velocity_ema_alpha=a.pf_velocity_ema_alpha)

    print(f'Running particle-filter eval on {len(available)} pieces '
          f'(seq_len={a.seq_len}, scale_factor={a.scale_factor}, device={device}, '
          f'pf_kwargs={pf_kwargs})...', flush=True)
    stats = _particle_filter_eval(
        network, available, a.cpjku_data, config,
        device=str(device), seq_len=a.seq_len,
        scale_factor=a.scale_factor, n_frames=n_frames, pf_kwargs=pf_kwargs)

    import numpy as np
    frame_diffs  = stats['frame_differences']
    onset_diffs  = np.array(frame_diffs['onset_diffs']) / config['spectrogram_params']['fps']
    total_onsets = len(onset_diffs)

    print(f'\n=== CPJKU {model_label} + particle-filter decode on MSMD {a.split} '
          f'({len(available)} pieces) ===')
    for th in [0.05, 0.1, 0.5, 1.0, 5.0]:
        pct = 100 * np.sum(onset_diffs <= th) / total_onsets
        print(f'  <= {th}s: {pct:.1f}%')
    print(f'  mean error:   {onset_diffs.mean():.3f}s')
    print(f'  median error: {np.median(onset_diffs):.3f}s')
    print(f'  total onsets: {total_onsets}')


if __name__ == '__main__':
    main()
