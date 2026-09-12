"""Splice score discontinuities into a performance, CODA's jump benchmark.

WHY
---
CODA (ISMIR 2026) contributes a repeat-aware jump benchmark and reports that
CYOLO-SB recovers the correct system for 8-12% of jumps within one second,
against 64-78% for CODA with its break mode. Our decoder inherits CYOLO-SB's
weakness by construction: the transition prior is a floored Gaussian, so a jump
costs a fixed `jump` nats and the tracker resists it. That is exactly the
"CODA w/o break" row, which recovers 23-29%.

We cannot reproduce their REPEAT subset: it rests on repeat structure they
annotated by hand and have not released. We can reproduce their RANDOM subset
exactly as specified -- three randomly placed jumps per piece -- which their own
Table 2 calls the harder of the two, because a random destination resumes at an
arbitrary onset rather than a musically defined bar line.

HOW THE SPLICE WORKS
--------------------
The timeline is rewritten, not the annotations. `load_sequences` yields one
sequence entry per spectrogram frame, each carrying the frame index and the
ground-truth position at that frame. We build a permutation of frame indices,

    new frame t  ->  old frame order[t]     (-1 marks an inserted silence)

concatenate the audio in that order one hop at a time, and re-emit the sequences
with `frame` renumbered to the new timeline and `true_position` carried over
from the old one. Everything downstream -- the detector, the harness metric, the
candidate dump -- is untouched and sees an ordinary performance that happens to
contain jumps.

Splicing the AUDIO rather than the candidate dump matters. The detector
conditions on a 40-frame buffer, so after a jump its conditioning is genuinely
wrong for two seconds. Reordering a finished dump would hand the decoder perfect
post-jump proposals and measure a system nobody can build.

SILENCE IS A PARAMETER, NOT AN ASSUMPTION
-----------------------------------------
CODA's break mode fires on low waveform energy, resting on Nakamura et al.'s
observation that 59 of 63 real repeats are preceded by a break. `gap` frames of
digital silence are therefore inserted by default (3-12, their augmentation
range). Setting `gap=0` removes the cue and leaves the jump instantaneous, which
is the case their mechanism cannot see and a page-turn or a performer error
actually produces. Both are run.

The per-frame RMS written to the sidecar is measured off the spliced signal the
same way an online system would measure it. Nothing here tells the decoder where
the jumps are; that is what the sidecar's `jumps` array is for, and it is read
only by the scorer that grades the result.
"""
from __future__ import annotations

import os
import zlib

import numpy as np

HOP = 1102
FPS = 22050 / HOP

SIDECAR: dict = {}


def _plan(n_frames, n_jumps, gap, rng, min_seg):
    """Old-frame index per new frame, -1 for an inserted silent frame.

    Every segment is at least `min_seg` frames so the 5 s post-jump window CODA
    scores always lies inside one segment, and the segments sum to the original
    length so the spliced take is as long as the take it came from. A benchmark
    that also doubled the duration would confound jump recovery with drift.
    """
    n_jumps = min(n_jumps, n_frames // min_seg - 1)
    if n_jumps < 1:
        return np.arange(n_frames, dtype=np.int64), []
    seg = n_frames // (n_jumps + 1)
    order, jumps = [], []
    cur = 0
    for i in range(n_jumps + 1):
        jit = int(rng.integers(-seg // 4, seg // 4 + 1)) if seg > 4 * min_seg else 0
        length = max(min_seg, min(seg + jit, n_frames - cur))
        order.extend(range(cur, cur + length))
        if i == n_jumps:
            break
        src = cur + length - 1
        # a destination has to be somewhere the tracker is not already going,
        # otherwise the "jump" is a step the prior would have taken anyway, and
        # it needs min_seg frames of material left after it
        for _ in range(64):
            dst = int(rng.integers(0, n_frames - min_seg))
            if abs(dst - src) > min_seg:
                break
        order.extend([-1] * gap)
        jumps.append({'at': len(order), 'src_old': src, 'dst_old': dst, 'gap': gap})
        cur = dst
    return np.asarray(order, np.int64), jumps


def patch_jump(n_jumps=3, gap=8, seed=0, min_seg=100):
    """Rewrite every loaded piece into a jump-spliced version of itself."""
    import cyolo_score_following.utils.data_utils as du

    prev = du.load_sequences

    def load_sequences(params):
        out = prev(params)
        (piece_idx, scores, signal, piece_name, seqs,
         interpol_c2o, staff_coords, add_per_staff) = out
        if isinstance(signal, str) or not seqs:
            raise RuntimeError('jump splice needs the audio loaded in memory')
        n = len(seqs)
        if n < 2 * min_seg:
            print(f'[JUMP] {piece_name}: {n} frames, too short to splice', flush=True)
            return out
        rng = np.random.default_rng(zlib.crc32(f'{piece_name}|{seed}'.encode()))
        order, jumps = _plan(n, n_jumps, gap, rng, min_seg)

        sig = np.asarray(signal)
        chunks = np.zeros((len(order), HOP), sig.dtype)
        for t, o in enumerate(order):
            if o < 0:
                continue
            s = int(o) * HOP
            c = sig[s:s + HOP]
            chunks[t, :c.shape[0]] = c
        new_sig = chunks.reshape(-1)

        new_seqs = []
        for t, o in enumerate(order):
            base = seqs[int(o)] if o >= 0 else seqs[0]
            e = dict(base)
            e['frame'] = t
            e['is_onset'] = bool(base['is_onset']) if o >= 0 else False
            new_seqs.append(e)

        side = {
            'order': order,
            'rms': np.sqrt((chunks.astype(np.float64) ** 2).mean(1)).astype(np.float32),
            'is_onset': np.array([bool(s['is_onset']) for s in new_seqs]),
            'jumps': np.array([[j['at'], j['src_old'], j['dst_old'], j['gap']]
                               for j in jumps], np.int64).reshape(-1, 4),
        }
        SIDECAR[piece_name] = side
        # load_sequences runs in a fork pool, so anything left in SIDECAR dies
        # with the worker. One file per piece, merged by the parent at exit.
        d = os.environ.get('JUMP_SIDECAR_DIR', '')
        if d:
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, piece_name + '.npz'), **side)
        print(f'[JUMP] {piece_name}: {n} -> {len(order)} frames, '
              f'{len(jumps)} jumps, gap {gap}', flush=True)
        return (piece_idx, scores, new_sig, piece_name, new_seqs,
                interpol_c2o, staff_coords, add_per_staff)

    du.load_sequences = load_sequences
    import cyolo_score_following.dataset as ds
    if hasattr(ds, 'load_sequences'):
        ds.load_sequences = load_sequences
    ds._jump_patched = True
    print(f'[JUMP] {n_jumps} jumps per piece, gap {gap} frames, seed {seed}',
          flush=True)


def write_sidecar(path, from_dir=''):
    """Merge the per-piece files the fork workers wrote into one npz."""
    import glob
    flat = {}
    src = dict(SIDECAR)
    for f in sorted(glob.glob(os.path.join(from_dir or '.', '*.npz'))) if from_dir else []:
        piece = os.path.basename(f)[:-4]
        with np.load(f, allow_pickle=True) as z:
            src[piece] = {k: z[k] for k in z.files}
    for piece, d in src.items():
        for k, v in d.items():
            flat[f'{piece}||{k}'] = v
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **flat)
    print(f'[JUMP] sidecar -> {path} ({len(src)} pieces)', flush=True)
