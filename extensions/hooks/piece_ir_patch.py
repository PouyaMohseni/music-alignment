"""Degrade the training audio with a room, ONE room per piece.

WHY DEGRADE AT ALL
------------------
The selector is fit on synthetic training audio where the detector's own argmax
is already 92.7% correct, and then asked to run on real room recordings where it
is 80%. The confusions it exists to resolve are therefore nearly absent from its
training set. Convolving the training audio with real room impulse responses
manufactures those confusions from data we already have, without touching the
labels: reverberation changes what the detector sees, not where the notes are.

ONE ROOM PER PIECE
------------------
Upstream draws an IR independently per __getitem__ with p=0.5, so consecutive
frames of one performance would be recorded in different rooms. That is
unphysical, and it corrupts exactly the features this selector depends on: the
displacement from the previous position is computed across two frames, and if
those frames came from different acoustics the candidate dynamics are ones no
real recording produces. (The same mistake cost us a 200x slowdown and a wasted
run on the S2 data path, drawn per NOTE instead of per piece.)

So the IR is chosen deterministically from the piece name: every frame of a
performance is heard in one room, a different room per piece, and the whole
assignment is reproducible from the seed.

Upstream's convolution is already correct -- convolve(x, ir, 'full') truncated
by len(ir)-1 keeps the onset alignment. It is our own copy in
extensions/augmentation/impulse_response.py that used mode='same' and shifted
the audio half an IR length against its labels, invalidating the B6 experiment.
This uses theirs.
"""
from __future__ import annotations

import os
import random
import zlib

import numpy as np


def patch_piece_ir(seed: int = 0, prob: float = 1.0, snr_db: float = 0.0):
    """snr_db > 0 adds noise after the room, to make a HARDER validation set.

    The reverberant validation split has a detector argmax of 90.7 where room
    has 80.0, so it is a much easier problem, and a higher-capacity selector can
    win there by fitting slack real audio does not have. Measured over seven
    variants, validation rank is ANTI-correlated with room rank (Spearman
    -0.39): it picks vel_feat, worth 90.0, over feat_wide, worth 94.0.

    A validation set is only a proxy if it is as hard as the target. Noise at a
    fixed SNR, deterministic per piece like the room, is the cheapest knob that
    moves the argmax toward 80 without touching the labels.
    """
    from scipy.signal import convolve

    from cyolo_score_following.augmentations.impulse_response import ImpulseResponse

    def __call__(self, sample):
        piece = sample['file_name'].rsplit('_page_', 1)[0]
        rng = random.Random(zlib.crc32(f'{piece}|{seed}'.encode()))
        if rng.random() >= prob or not self.irs:
            return sample
        ir = self.irs[rng.randrange(len(self.irs))]
        if ir.shape[0] < 2:
            return sample
        p = sample['performance']
        out = convolve(p, ir, 'full')[:-(ir.shape[0] - 1)]
        if snr_db > 0 and out.size:
            rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
            if rms > 0:
                # `rng` is a stdlib random.Random, which has no .normal --
                # that is numpy's API. Seed a numpy generator from the SAME
                # per-piece key so the noise stays deterministic per piece,
                # like the room choice above.
                nrng = np.random.default_rng(rng.getrandbits(64))
                n = nrng.normal(0.0, rms * 10 ** (-snr_db / 20.0), out.shape)
                out = out + n.astype(out.dtype)
        sample['performance'] = out
        return sample

    ImpulseResponse.__call__ = __call__
    ImpulseResponse._piece_ir = True
    print(f'[IR] one room per piece (seed={seed}, prob={prob}, '
          f'snr={snr_db or "off"} dB)', flush=True)


def patch_loader_ir(ir_path):
    """eval.py never exposes --ir_path; the dataset builder accepts it."""
    import cyolo_score_following.dataset as ds

    _orig = ds.load_dataset

    def load_dataset(paths, **kw):
        kw['ir_path'] = ir_path
        return _orig(paths, **kw)

    ds.load_dataset = load_dataset
    ds._ir_loader_patched = True
    print(f'[IR] loader will build IR augmentation from {ir_path}', flush=True)
