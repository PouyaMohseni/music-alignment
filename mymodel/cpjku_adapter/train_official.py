"""Train the CPJKU ConditionalUNet (Henkel et al. ISMIR 2020) on our MSMD strips.

This is a faithful fork of third_party/cpjku_unet/audio_conditioned_unet/train_model.py.
It reuses THEIR exact training machinery unchanged:
  - ConditionalUNet           (their network)
  - iterate_dataset           (their BPTT training loop)
  - dice_loss(smoothing=0.)   (their loss, via iterate_dataset)
  - ReduceLROnPlateau         (their scheduler, patience=5, factor=0.5)
  - 100 epochs, early stop at patience*2, save best_model.pt + latest_model.pt
  - spectrogram normalisation from the training set (means/stds per band)

Only two things differ, both forced by our data, neither touches their algorithm:

  1. Data loading is patched (madmom -> librosa log-filterbank spectrogram, and
     load_piece -> our NPZ reader).  Same patches the eval uses.

  2. ScoreAudioDataset is subclassed (_LazyStripDataset) so __getitem__ does NOT
     replicate the (H, W) score T times.  Their original builds a (T, 1, H, W)
     array per piece; for our wide strips (up to 7045 px at scale 3) that is tens
     of GB and OOMs.  The lazy version materialises only the seq_len frames that
     prepare_batch slices.  Output dict structure is byte-for-byte identical to
     theirs, so iterate_dataset / prepare_batch / calculate_batch_stats are
     unchanged and unaware.

  batch_size is forced to 1: prepare_batch concatenates scores along the batch
  axis (np.concatenate(..., axis=1)), which requires equal H and W across the
  batch.  Our strips have variable widths, so only batch_size=1 is valid (BPTT
  over a single piece, LSTM state carried across seq_len chunks — exactly their
  use_lstm path).

    python -m mymodel.cpjku_adapter.train_official \
        --cpjku_root  third_party/cpjku_unet \
        --cpjku_data  data/MSMD/cpjku_fmt \
        --processed   data/MSMD/processed \
        --dump_root   results/cpjku_official \
        --seq_len 16 --scale_factor 3
"""
from __future__ import annotations
import argparse, copy, json, os, random, sys
from pathlib import Path

import numpy as np

# Reuse the exact data-loading patches the eval uses.
from mymodel.cpjku_adapter.eval_official import (
    _patched_load_piece,
    _wav_to_spec_logfilter,
    _build_log_filterbank,
    build_split_file,
)


def _load_dataset_sequential(cpjku_data, config, n_frames, augment, scale_factor,
                              split_file, ScoreAudioDataset):
    """Sequential drop-in for load_dataset — bypasses Pool.map BLAS-fork deadlock on SLURM.

    Their load_dataset calls multiprocessing.Pool(8).map; forked workers inherit
    locked BLAS thread pools from the parent and deadlock indefinitely.  Loading
    each piece in the main process avoids the fork entirely.
    """
    import glob, yaml
    import tqdm as _tqdm

    if split_file is not None:
        with open(split_file, 'rb') as fp:
            split = yaml.load(fp, Loader=yaml.FullLoader)
        files = [os.path.join(cpjku_data, 'score', f'{f}.npz') for f in split['files']]
    else:
        files = glob.glob(os.path.join(cpjku_data, 'score', '*.npz'))

    scores, piece_names, performances = {}, {}, {}

    for i, score_path in enumerate(_tqdm.tqdm(files, desc='Loading')):
        params = dict(
            i=i,
            piece_name=os.path.basename(score_path)[:-4],
            path=cpjku_data,
            sf_path=config.get('sf_path', ''),
            scale_factor=scale_factor,
            spectrogram_params=config['spectrogram_params'],
            tempo_factors=config.get('tempo_factors', [1000]),
            real_perf=config.get('real_perf', True),
        )
        try:
            _, score, piece_name, perfs = _patched_load_piece(params)
            scores[i] = score
            piece_names[i] = piece_name
            performances[i] = perfs
        except Exception as e:
            print(f'  FAIL {os.path.basename(score_path)}: {e}', flush=True)

    return ScoreAudioDataset(scores, performances, piece_names, n_frames=n_frames,
                             config=config, augment=augment, all_tempi=False)


# ──────────────────────────────────────────────────────────────────────────────
# Lazy strip dataset — memory-efficient drop-in for ScoreAudioDataset.
# Subclass that overrides __getitem__ only; everything else (set_random_perfs,
# get_score_shape, __len__, attributes) is inherited unchanged.
# ──────────────────────────────────────────────────────────────────────────────
def _make_lazy_dataset_class(ScoreAudioDataset, MSMD_Y_OFFSET):
    """Build the subclass inside main() so we can reference their imported symbols."""

    class _LazySlice:
        """Indexable view that materialises (n, 1, H, W) score/y slices on demand.

        Mirrors what ScoreAudioDataset.__getitem__ stores as inputs['score'] /
        targets['y'] — element shape (1, H, W), full length T — but never holds
        more than the requested slice in memory.
        """
        def __init__(self, kind, base_score, true_positions, height, gt_width,
                     shifts, T):
            self.kind   = kind            # 'score' or 'y'
            self.base   = base_score      # (H, W) float32
            self.tp     = true_positions  # (T, 2) int  [y, x]
            self.height = height          # (T,) int  adaptive staff height
            self.gw     = gt_width
            self.shifts = shifts          # (T, 2) int  [yshift, xshift] or None
            self.T      = T
            self.H, self.W = base_score.shape

        def __len__(self):
            return self.T

        def __getitem__(self, sl):
            idx = list(range(*sl.indices(self.T)))
            n   = len(idx)

            if self.kind == 'score' and self.shifts is None:
                # Constant across frames -> zero-copy broadcast view.
                return np.broadcast_to(self.base[None, None], (n, 1, self.H, self.W))

            out = np.zeros((n, 1, self.H, self.W), dtype=np.float32)
            for j, i in enumerate(idx):
                if self.kind == 'score':
                    s = self.base
                    if self.shifts is not None:
                        s = np.roll(s, self.shifts[i, 0], axis=0)
                        s = np.roll(s, self.shifts[i, 1], axis=1)
                    out[j, 0] = s
                else:  # 'y'
                    y    = np.zeros((self.H, self.W), dtype=np.float32)
                    cy, cx = int(self.tp[i, 0]), int(self.tp[i, 1])
                    h    = int(self.height[i])
                    y[max(0, cy - h // 2):cy + h // 2,
                      max(0, cx - self.gw // 2):cx + self.gw // 2] = 1.0
                    if self.shifts is not None:
                        y = np.roll(y, self.shifts[i, 0], axis=0)
                        y = np.roll(y, self.shifts[i, 1], axis=1)
                    out[j, 0] = y
            return out

    class _LazyPerf:
        """Indexable view returning (n, 1, n_mels, n_frames) spectrogram clips."""
        def __init__(self, spec, pad, n_frames, T):
            self.spec     = spec        # (n_mels, T_total)
            self.pad      = pad
            self.n_frames = n_frames
            self.T        = T
            self.n_mels   = spec.shape[0]

        def __len__(self):
            return self.T

        def __getitem__(self, sl):
            idx = list(range(*sl.indices(self.T)))
            out = np.zeros((len(idx), 1, self.n_mels, self.n_frames), dtype=np.float32)
            for j, frame in enumerate(idx):
                i    = self.pad + frame                      # spec index
                clip = self.spec[:, i - self.n_frames + 1:i + 1]
                if clip.shape[-1] < self.n_frames:           # left-pad first frames
                    clip = np.pad(clip, ((0, 0), (self.n_frames - clip.shape[-1], 0)))
                out[j, 0] = clip
            return out

    class _LazyStripDataset(ScoreAudioDataset):
        """ScoreAudioDataset that builds frames lazily (no T-replication)."""

        def __getitem__(self, item):
            score_id = item
            score    = self.scores[item]                     # (H, W)
            perfs    = self.performances[item]
            perf     = perfs[np.random.choice(list(perfs.keys()))]

            spec    = perf['spec']                           # (n_mels, T_total)
            inp     = perf['interpol_fnc']
            onsets  = perf['onsets']

            T = spec.shape[-1] - self.pad                    # number of frames
            H, W = score.shape

            # Precompute true positions [y, x, height] for every frame.
            frames = np.arange(T)
            tp_all = np.asarray(inp(frames), dtype=np.int32)  # (3, T)
            tp_all = tp_all.T                                 # (T, 3)
            true_positions = tp_all[:, :2]                    # (T, 2) [y, x]
            height         = tp_all[:, 2]                     # (T,)

            # Per-frame augmentation shifts (matches their per-frame np.roll aug).
            shifts = None
            if self.augment:
                # max_y_shift uses the spec-length true position, as in their code.
                ymax = int(inp(spec.shape[-1])[0])
                max_y_shift = max(score.shape[0] - ymax - MSMD_Y_OFFSET, -8)
                ysh = np.random.randint(-9, max(max_y_shift, -8), size=T)
                xsh = np.random.randint(-9, 13, size=T)
                shifts = np.stack([ysh, xsh], axis=1)        # (T, 2)

            onset_set = set(int(o) for o in onsets)
            is_onset  = [(f in onset_set) for f in range(T)]

            score_view = _LazySlice('score', score, true_positions, height,
                                    self.gt_width, shifts, T)
            y_view     = _LazySlice('y', score, true_positions, height,
                                    self.gt_width, shifts, T)
            perf_view  = _LazyPerf(spec, self.pad, self.n_frames, T)

            return {
                'inputs':  {'perf': perf_view, 'score': score_view, 'length': T},
                'targets': {'y': y_view, 'true_positions': true_positions},
                'file_name':    self.piece_names[score_id],
                'interpol_c2o': perf['interpol_c2o'],
                'add_per_staff': perf['add_per_staff'],
                'is_onset':     is_onset,
            }

    return _LazyStripDataset


def main():
    p = argparse.ArgumentParser(description='Train CPJKU ConditionalUNet on our MSMD strips')
    p.add_argument('--cpjku_root',  default='third_party/cpjku_unet')
    p.add_argument('--cpjku_data',  default='data/MSMD/cpjku_fmt')
    p.add_argument('--processed',   default='data/MSMD/processed')
    p.add_argument('--dump_root',   default='results/cpjku_official')
    p.add_argument('--tag',         default='CB_TA_strips')

    # Net config — defaults match their CB_TA model exactly.
    p.add_argument('--film_layers', nargs='+', type=int, default=[2, 3, 4, 5, 6, 7, 8])
    p.add_argument('--n_encoder_layers', type=int, default=4)
    p.add_argument('--n_filters_start',  type=int, default=8)
    p.add_argument('--rnn_size',         type=int, default=128)
    p.add_argument('--rnn_layer',        type=int, default=1)
    p.add_argument('--spec_enc',         type=int, default=32)
    p.add_argument('--audio_encoder',    default='CBEncoder')
    p.add_argument('--use_lstm', action='store_true', default=True)

    # Train hyperparameters — their defaults.
    p.add_argument('--batch_size',   type=int, default=1,
                   help='Forced to 1: variable strip widths break batch concat.')
    p.add_argument('--seq_len',      type=int, default=16)
    p.add_argument('--learning_rate', '--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--patience',     type=int, default=5)
    p.add_argument('--num_epochs',   type=int, default=100)
    p.add_argument('--scale_factor', type=int, default=3)
    p.add_argument('--augment',      action='store_true', default=False)
    p.add_argument('--clip_grads',   type=float, default=None)
    p.add_argument('--seed',         type=int, default=4711)
    p.add_argument('--param_path',   default=None, help='resume from checkpoint')
    args = p.parse_args()

    # ── Reproducibility (their settings) ─────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)

    cpjku_root = Path(args.cpjku_root).resolve()
    if not cpjku_root.exists():
        raise FileNotFoundError(f'CPJKU repo not found: {cpjku_root}')
    if str(cpjku_root) not in sys.path:
        sys.path.insert(0, str(cpjku_root))

    # ── Patch madmom -> librosa BEFORE importing their modules ───────────────
    from mymodel.cpjku_adapter import madmom_compat
    madmom_compat.patch()

    import torch
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from audio_conditioned_unet.network import ConditionalUNet
    from audio_conditioned_unet.dataset import (
        iterate_dataset, ScoreAudioDataset,
    )
    import audio_conditioned_unet.dataset as _ds
    from audio_conditioned_unet.dataset import MSMD_Y_OFFSET

    torch.manual_seed(args.seed)

    # Patch their load_piece with our NPZ reader (picklable, module-level).
    _ds.load_piece = _patched_load_piece

    _LazyStripDataset = _make_lazy_dataset_class(ScoreAudioDataset, MSMD_Y_OFFSET)

    # ── Build split files for train / val ────────────────────────────────────
    train_split, train_ids = build_split_file(args.processed, args.cpjku_data, 'train')
    val_split,   val_ids   = build_split_file(args.processed, args.cpjku_data, 'val')
    if not train_ids:
        raise RuntimeError('No train pieces converted. Run mymodel.cpjku_adapter.convert.')
    if not val_ids:
        raise RuntimeError('No val pieces converted. Run mymodel.cpjku_adapter.convert.')

    config = {
        'spectrogram_params': {'sample_rate': 22050, 'frame_size': 2048, 'fps': 20, 'pad': 40},
        'gt_width': 10,
        'real_perf': True,
        'tempo_factors': [1000],
        'sf_path': '',
    }
    val_config = copy.deepcopy(config)
    val_config['tempo_factors'] = [1000]

    # ── Net config (their CB_TA) ─────────────────────────────────────────────
    net_config = {
        'film_layers':      args.film_layers,
        'n_encoder_layers': args.n_encoder_layers,
        'n_filters_start':  args.n_filters_start,
        'rnn_size':         args.rnn_size,
        'rnn_layer':        args.rnn_layer,
        'use_lstm':         args.use_lstm,
        'audio_encoder':    args.audio_encoder,
        'spec_enc':         args.spec_enc,
    }

    from time import gmtime, strftime
    time_stamp = strftime('%Y%m%d_%H%M%S', gmtime()) + f'_{args.tag}'
    dump_path  = Path(args.dump_root) / time_stamp
    dump_path.mkdir(parents=True, exist_ok=True)
    with open(dump_path / 'net_config.json', 'w') as f:
        json.dump(net_config, f)
    print(f'Dump path: {dump_path}', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = ConditionalUNet(net_config)
    if args.param_path is not None:
        print(f'Loading weights from {args.param_path}', flush=True)
        network.load_state_dict(torch.load(args.param_path, map_location='cpu'))

    n_frames = network.perf_encoder.n_input_frames

    # ── Load data sequentially (bypasses Pool.map BLAS-fork deadlock on SLURM) ──
    print('Loading train dataset...', flush=True)
    train_dataset = _load_dataset_sequential(
        args.cpjku_data, config, n_frames=n_frames, augment=args.augment,
        scale_factor=args.scale_factor, split_file=train_split,
        ScoreAudioDataset=ScoreAudioDataset)
    print('Loading val dataset...', flush=True)
    val_dataset = _load_dataset_sequential(
        args.cpjku_data, val_config, n_frames=n_frames, augment=False,
        scale_factor=args.scale_factor, split_file=val_split,
        ScoreAudioDataset=ScoreAudioDataset)

    # ── Spectrogram normalisation from the training set (their exact code) ───
    specs = [train_dataset.performances[elem][1000]['spec'] for elem in train_dataset.performances]
    means = np.mean(np.concatenate(specs, axis=-1), axis=1)
    stds  = np.std(np.concatenate(specs, axis=-1), axis=1)
    network.perf_encoder.set_stats(means, stds)
    print(f'Spec stats: means.mean={means.mean():.4f}  stds.mean={stds.mean():.4f}', flush=True)

    # ── Wrap with the lazy dataset (memory-efficient drop-in) ────────────────
    train_dataset.__class__ = _LazyStripDataset
    val_dataset.__class__   = _LazyStripDataset

    network.to(device)
    print(f'Model on {device} | params: '
          f'{sum(pm.numel() for pm in network.parameters() if pm.requires_grad):,}', flush=True)

    optim = torch.optim.Adam(network.parameters(), lr=args.learning_rate,
                             weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=args.patience, factor=0.5)

    max_patience = args.patience * 2
    patience     = max_patience
    min_loss     = np.inf

    for epoch in range(args.num_epochs):
        tr_stats = iterate_dataset(network, optim, train_dataset, args.batch_size,
                                   seq_len=args.seq_len, train=True, device=device,
                                   threshold=0.5, clip_grads=args.clip_grads)
        tr_loss, tr_prec, tr_rec = tr_stats['loss'], tr_stats['precision'], tr_stats['recall']

        val_stats = iterate_dataset(network, None, val_dataset, batch_size=args.batch_size,
                                    seq_len=args.seq_len, train=False, device=device,
                                    threshold=0.5)
        val_loss, val_prec, val_rec = val_stats['loss'], val_stats['precision'], val_stats['recall']

        scheduler.step(val_loss)

        improved = val_loss < min_loss
        if improved:
            min_loss = val_loss
            patience = max_patience
            torch.save(network.state_dict(), dump_path / 'best_model.pt')
        else:
            patience -= 1
        torch.save(network.state_dict(), dump_path / 'latest_model.pt')

        tr_f1  = 2 * (tr_prec * tr_rec) / (tr_prec + tr_rec) if tr_prec > 0 and tr_rec > 0 else 0
        val_f1 = 2 * (val_prec * val_rec) / (val_prec + val_rec) if val_prec > 0 and val_rec > 0 else 0

        flag = '*BEST*' if improved else f'(patience {patience}/{max_patience})'
        print(f'Epoch {epoch:3d} | train loss {tr_loss:.4f} P {tr_prec:.3f} R {tr_rec:.3f} '
              f'F1 {tr_f1:.3f} | val loss {val_loss:.4f} P {val_prec:.3f} R {val_rec:.3f} '
              f'F1 {val_f1:.3f} | lr {optim.param_groups[0]["lr"]:.2e} {flag}', flush=True)

        if patience <= 0:
            print(f'Early stopping at epoch {epoch} (val loss did not improve for '
                  f'{max_patience} epochs).', flush=True)
            break

    print(f'Training done. best val loss = {min_loss:.5f}', flush=True)
    print(f'Best model: {dump_path / "best_model.pt"}', flush=True)
    print(f'Net config: {dump_path / "net_config.json"}', flush=True)


if __name__ == '__main__':
    main()
