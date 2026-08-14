"""S1 dataset -- unrolled strip + audio window + 1-D target column.

WHY WAVEFORM AUGMENTATION IS THE POINT OF THIS FILE
----------------------------------------------------
Real-IR augmentation is the single largest measured lever in this task: CYOLO
goes 46.0 -> 71.2 on room with it (+25.2), and our own heatmap model went
45.6 -> 56.6 (+11.0). It is applied to the WAVEFORM, so it is only available if
training reads audio rather than precomputed features.

That is exactly what we lost every time we precomputed an embedding bank. The
MERT-in-detector run overfit 12.6x (train frame-diff 2.04, val 25.73) while the
native waveform model had val BETTER than train (4.48 vs 7.61) -- because
waveform augmentation makes training the harder task. An audit traced this
directly: the precomputed bank held a frozen 7-point tempo grid (945 stems x 7
tempi) against the native path's continuous per-batch phase vocoder plus a
fresh IR draw per sample.

So this dataset renders the spectrogram ON THE FLY from augmented audio. It is
slower per step and it is the reason the model can generalise at all.

TARGET
------
One number per frame: the x column of the note sounding at that frame, on the
unrolled strip. No y, no staff index -- the strip has a single row by
construction, so the staff-assignment step that made a correct-x prediction
score ~0 (P1: 10.6 on room) simply does not exist here.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

SR = 22050
FPS = 20.0
N_MELS = 78
FRAME_SIZE = 2048
HOP = int(round(SR / FPS))

_NO_STRIP = torch.zeros(0)      # placeholder when return_strip=False


def _log_mel(y: np.ndarray) -> np.ndarray:
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=FRAME_SIZE, hop_length=HOP,
                                       n_mels=N_MELS, fmin=60, fmax=6000)
    return np.log1p(100.0 * m).astype(np.float32).T      # (T, 78)


class StripFollowDataset(Dataset):
    """Yields (strip, mel_window, target_col, piece_idx).

    Frames are grouped by piece by the sampler so `encode_strip` can be cached:
    the trunk is audio-independent and the strip does not change within a piece.
    """

    def __init__(self, score_dir, perf_dir, pieces, strip_scale=2, win=40,
                 ir_bank=None, ir_prob=0.5, tempo_range=(0.7, 1.4),
                 augment=True, max_strip_w=24000, return_strip=True):
        self.score_dir, self.perf_dir = score_dir, perf_dir
        self.strip_scale, self.win = strip_scale, win
        self.augment = augment
        self.ir_prob, self.tempo_range = ir_prob, tempo_range
        self.max_strip_w = max_strip_w
        # S2 reads the strip once per piece in the MAIN process. Returning it
        # per item too would ship ~5 MB through the worker pickle for every
        # frame in the batch -- gigabytes of IPC to re-send an object that
        # never changes within a piece. S1 still needs it, so this is a flag.
        self.return_strip = return_strip

        self.irs = []
        if ir_bank and augment:
            from pathlib import Path
            self.irs = sorted(str(p) for p in Path(ir_bank).rglob('*.wav'))
            if not self.irs:
                raise RuntimeError(f'ir_bank {ir_bank!r} has no wavs')

        self.items = []          # (piece, frame_idx, target_col)
        self.pieces = []
        self._strip_cache, self._audio_cache = {}, {}

        for p in pieces:
            sp = os.path.join(score_dir, p + '.npz')
            if not os.path.exists(sp):
                continue
            # allow_pickle=False on purpose. The only pickled member is
            # `coord2onset`, an object-array dict that convert.py:91 builds as
            # the IDENTITY map `{i: i for i in range(N)}` after truncating
            # coords and onset_frames together -- so coords[k] is onset[k] by
            # construction and reading it back would tell us nothing. Loading
            # it anyway cost us a hard failure: the NPZs were written under
            # NumPy 2.x, whose pickles reference `numpy._core`, which does not
            # exist in the 1.x interpreter this trains under.
            z = np.load(sp, allow_pickle=False)
            coords = np.array(z['coords'], dtype=np.float32)
            onsets = np.array(z['onset_frames'], dtype=np.int64)
            # coords are [y, x, ...]; on a strip every y is the same row, so
            # only x carries information
            xs = coords[:, 1]
            n = min(len(onsets), len(xs))
            if n < 2:
                continue
            pi = len(self.pieces)
            self.pieces.append(p)
            # one training item per annotated onset frame
            for k in range(n):
                self.items.append((pi, int(onsets[k]), float(xs[k])))

    def __len__(self):
        return len(self.items)

    def _strip(self, pi):
        if pi in self._strip_cache:
            return self._strip_cache[pi]
        z = np.load(os.path.join(self.score_dir, self.pieces[pi] + '.npz'),
                    allow_pickle=False)
        s = np.array(z['sheet'], dtype=np.float32) / 255.0
        s = 1.0 - s                                    # ink = high, as CYOLO does
        if self.strip_scale > 1:
            s = s[::self.strip_scale, ::self.strip_scale]
        if s.shape[1] > self.max_strip_w:
            s = s[:, :self.max_strip_w]
        t = torch.from_numpy(np.ascontiguousarray(s))[None]
        if len(self._strip_cache) < 32:
            self._strip_cache[pi] = t
        return t

    def _audio(self, pi):
        if pi in self._audio_cache:
            return self._audio_cache[pi]
        import librosa
        wav = os.path.join(self.perf_dir, self.pieces[pi] + '.wav')
        y, _ = librosa.load(wav, sr=SR, mono=True)
        if len(self._audio_cache) < 16:
            self._audio_cache[pi] = y
        return y

    def _augment_audio(self, y, rng):
        from scipy.signal import fftconvolve
        if self.irs and rng.random() < self.ir_prob:
            import librosa
            ir, _ = librosa.load(self.irs[rng.integers(0, len(self.irs))], sr=SR, mono=True)
            pk = int(np.argmax(np.abs(ir)))
            ir = ir[pk:][:SR]                          # trim pre-delay: it would
            n = float(np.abs(ir).max())                # translate audio vs labels
            if n > 0:
                ir = ir / n
                # 'full' then truncate, NOT mode='same': 'same' advances the
                # signal by (len(ir)-1)//2 samples relative to its onset labels,
                # a 4-20 frame desync that made an entire IR experiment look
                # like a failure for a month (commit e8320ea).
                y = fftconvolve(y, ir, mode='full')[:len(y)].astype(np.float32)
        if rng.random() < 0.5:
            y = y + rng.normal(0, float(np.abs(y).std()) * 0.02, len(y)).astype(np.float32)
        return y

    def __getitem__(self, i):
        pi, frame, x_true = self.items[i]
        rng = np.random.default_rng(random.randrange(1 << 30))

        y = self._audio(pi)
        t_end = FRAME_SIZE + int(frame * HOP)
        t_start = max(0, t_end - (self.win + 2) * HOP - FRAME_SIZE)
        seg = y[t_start:t_end]
        if len(seg) < FRAME_SIZE + HOP:
            seg = np.pad(seg, (FRAME_SIZE + HOP - len(seg), 0))
        if self.augment:
            seg = self._augment_audio(seg, rng)

        mel = _log_mel(seg)                            # (T, 78)
        if mel.shape[0] < self.win:
            mel = np.pad(mel, ((self.win - mel.shape[0], 0), (0, 0)))
        mel = mel[-self.win:]

        strip = self._strip(pi) if self.return_strip else _NO_STRIP
        # target in DOWNSCALED strip pixels, matching the model's input
        return (strip, torch.from_numpy(mel),
                torch.tensor(x_true / self.strip_scale, dtype=torch.float32),
                pi)


def collate_by_piece(batch):
    """All items in a batch come from one piece (see the sampler), so the strip
    is shared and `encode_strip` runs once."""
    strips, mels, tgts, pis = zip(*batch)
    return strips[0][None], torch.stack(mels), torch.stack(tgts), pis[0]


class PieceBatchSampler(torch.utils.data.Sampler):
    """Batches of frames drawn from ONE piece, so the trunk can be cached."""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.by_piece = {}
        for idx, (pi, _, _) in enumerate(dataset.items):
            self.by_piece.setdefault(pi, []).append(idx)
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        batches = []
        for pi, idxs in self.by_piece.items():
            idxs = list(idxs)
            if self.shuffle:
                random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                b = idxs[i:i + self.batch_size]
                if len(b) > 1:
                    batches.append(b)
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(max(0, len(v) // self.batch_size) for v in self.by_piece.values())
