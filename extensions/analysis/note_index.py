"""Turn candidate positions into a NOTE INDEX, and decode in that coordinate.

WHY
---
The transition prior models displacement in pixels. Measured on the room set,
the true step from one onset to the next has a coefficient of variation of

    0.425 in pixels        0.256 in note index

so it is 40% more predictable counted in noteheads, and that holds in 14 of 16
pieces. The cause is engraving: a run of sixteenths packs many noteheads into
the width one whole note occupies, so a fixed sigma_px is simultaneously too
tight in sparse passages and too loose in dense ones. Drift errors sit at a
median of 97.6 px, which on an 835 px page is the wrong NOTEHEAD in the same or
an adjacent bar -- exactly the discrimination a pixel prior is worst at.

HOW THE INDEX IS BUILT
----------------------
Without symbolic input, the only notehead evidence is the detector's own
candidates. They cluster: many anchors fire on one notehead. So per page, pool
class-0 candidates over the WHOLE piece into position bins, and a candidate's
index is the rank of its bin in reading order. Two properties matter:

  * it is PER-CANDIDATE -- each candidate gets its own rank -- which is the
    property that made backbone features help and made z useless;
  * it uses nothing the tracker is not given: no annotation, no pitches, only
    boxes the frozen detector already emitted.

The bins are built from candidates pooled across the whole recording, which is
not causal. That is a deliberate, stated limitation of this experiment: it asks
whether the COORDINATE is better before paying to estimate it online. A
deployed version would accumulate bins as it goes, and the score page is
available from the first frame anyway, so a one-pass pre-scan over the image is
legitimate in a real system.
"""
from __future__ import annotations

import numpy as np


def build_bins(pages, bin_px=12.0, min_obj=0.10):
    """-> per page name, sorted array of notehead-bin centres in unrolled x."""
    bins = {}
    for p in pages:
        xs = []
        for c in p['cand']:
            if c.shape[0] == 0:
                continue
            keep = c[:, 4] >= min_obj
            xs.append(c[keep, 0])
        if not xs:
            bins[p['name']] = np.zeros(0)
            continue
        x = np.sort(np.concatenate(xs))
        if x.size == 0:
            bins[p['name']] = np.zeros(0)
            continue
        # single-link clustering at bin_px, then the objectness-free centroid;
        # a notehead is one cluster no matter how many anchors fired on it
        cut = np.nonzero(np.diff(x) > bin_px)[0]
        groups = np.split(x, cut + 1)
        bins[p['name']] = np.array([g.mean() for g in groups])
    return bins


def to_index(x, centres):
    """Fractional rank of x among the bin centres, so nearby candidates inside
    one notehead do not collapse to the same integer."""
    if centres.size == 0:
        return np.asarray(x, np.float64) / 12.0
    return np.interp(np.asarray(x, np.float64), centres,
                     np.arange(centres.size, dtype=np.float64))
