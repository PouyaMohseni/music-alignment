"""Would pitch agreement help, if we could read it perfectly?

The failure analysis says errors are pitch-DISTINGUISHABLE: median pitch overlap
between the chosen and the true position is 0.000, and 59.2% of errors land
somewhere with no notes in common. Score-following's recent SOTA reports a large
gain from feeding raw onset probabilities from a transcription model. We use no
pitch at all -- audio enters only as FiLM conditioning on the detector.

This measures the CEILING of that idea before building anything, by scoring
candidates with pitch agreement taken from the annotations.

THIS IS A FEASIBILITY PROBE, NOT A RESULT. It reads MIDI at inference, which
the deployed system may never do -- the method has to work on music with no
symbolic reference. If the probe pays, the real build is two heads trained with
MIDI as supervision and run on image and audio features alone at inference:

  score head   backbone feat (128) at a candidate -> its pitch, read off the
               staff from the image
  audio head   z (128) at a frame -> the pitch classes currently sounding

and the FEATURE is their agreement, which varies per candidate. A raw audio
pitch estimate is constant across a frame and could rank nothing, the same
reason the z vector failed twice; only the interaction carries information.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.offline_decode import TH, load

FPS = 22050.0 / 1102.0
MSMD = '/scratch/pmohseni/datasets/cyolo_data/msmd'


def piece_notes(name):
    """Annotated noteheads: (x, y, pitch, onset_seconds, page)."""
    stem = name.split('_page_')[0]
    for d in ('msmd_rp', 'msmd_train', 'msmd_valid', 'msmd_test'):
        for f in glob.glob(f'{MSMD}/{d}/{stem}.npz'):
            z = np.load(f, allow_pickle=True)
            out = []
            for c in z['coords']:
                out.append((float(c['note_x']), float(c['note_y']),
                            int(c['pitch']), float(c['onset']),
                            int(c['page_nr'])))
            return np.array(out, np.float64) if out else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/scratch/pmohseni/omr/candf/room.npz')
    ap.add_argument('--tol', type=float, default=0.05,
                    help='seconds; a note counts as sounding this close to the onset')
    a = ap.parse_args()
    pages = load(a.dump)
    tot = agree_true = agree_wrong = 0
    disc = []
    miss = 0
    for p in pages:
        nt = piece_notes(p['name'])
        if nt is None:
            miss += 1
            continue
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            t_now = float(p['t_gt'][i]) / FPS
            sounding = set(nt[np.abs(nt[:, 3] - t_now) <= a.tol, 2].astype(int))
            if not sounding:
                continue
            tot += 1
            # Each candidate's implied pitch, joined in the ONSET domain.
            #
            # cand[:,0] is UNROLLED x (add_per_staff already folded in) while
            # the annotations carry raw per-page note_x, so matching on x
            # compares two different coordinate systems -- the first version of
            # this probe did exactly that and reported a meaningless 1.14x.
            #
            # cand[:,5] is the candidate's own position mapped through the
            # piece's x->onset interpolator, so it identifies WHICH NOTEHEAD the
            # box sits on, by position. That is precisely what a pitch head
            # reading the image would recover, and it sidesteps the coordinate
            # problem. It is a CEILING: a real head would be noisier.
            ct = c[:, 5] / FPS
            near = nt[np.argmin(np.abs(nt[:, 3][None, :] - ct[:, None]), 1),
                      2].astype(int)
            ok = np.array([pp in sounding for pp in near])
            corr = np.abs(c[:, 5] - p['t_gt'][i]) <= TH
            if corr.any():
                agree_true += ok[corr].mean()
            if (~corr).any():
                agree_wrong += ok[~corr].mean()
            # how much does agreement shrink the candidate set?
            disc.append(ok.mean())
    if not tot:
        print('no usable frames -- annotations did not join to the dump')
        return
    print(f'{tot} frames joined ({miss} pages had no annotation file)\n')
    print(f'  pitch agrees, CORRECT candidates    {agree_true / tot:6.3f}')
    print(f'  pitch agrees, WRONG candidates      {agree_wrong / tot:6.3f}')
    print(f'  fraction of all candidates agreeing {np.mean(disc):6.3f}')
    lift = (agree_true / tot) / max(agree_wrong / tot, 1e-9)
    print(f'\n  correct-vs-wrong agreement ratio    {lift:6.2f}x')
    print('  a ratio near 1 means pitch cannot separate them and the whole')
    print('  direction is dead; well above 1 means the two heads are worth it.')


if __name__ == '__main__':
    main()
