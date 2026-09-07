"""DAgger for the candidate selector: train on the states it actually visits.

WHY THIS IS THE BIGGEST REMAINING IDEA
--------------------------------------
The selector's input depends on its OWN previous position, so teacher forcing
trains it on a history it will never see. The measured cost of that mismatch is
enormous: removing the previous-position noise -- the crude stand-in for it --
takes validation rollout from 95.33 to 92.38, below the detector's own argmax.
Exposure bias is not a detail here, it is the dominant term.

Laplace noise on the previous position is a guess at the distribution of the
model's own mistakes. DAgger measures it instead: roll the current policy out,
collect the states it actually reaches, relabel each with the oracle choice, and
refit on the union. Repeat. This is what Peter (2024) calls a simplified offline
RL problem for symbolic alignment, and the reason it is not RL proper is that we
know the optimal action at every step -- it is the candidate closest to ground
truth, which the dumps already contain.

Everything stays on the training split; room is never touched.
"""
from __future__ import annotations

import numpy as np


def rollout_states(model, pieces, build, torch, use_abs_obj=True, featdim=0,
                   nf=None):
    """Run the policy greedily and return the (piece, frame, x_prev, y_prev,
    x_prev2, dfr, dfr_prev) states it actually visits, with the oracle's choice
    at each. These are the states teacher forcing never generates."""
    out = []
    for pi, p in enumerate(pieces):
        xp = yp = xp2 = fp = fp2 = None
        for fi, c in enumerate(p['cand']):
            if len(c) == 0:
                continue
            fr = int(p['frame'][fi])
            dfr = fr - fp if fp is not None and fr > fp else None
            dfp = fp - fp2 if fp is not None and fp2 is not None and fp > fp2 else None
            f = build(c, p['bar'][fi], p['sys'][fi], xp, yp, dfr,
                      ntot=int(p['ntot'][fi]), use_abs_obj=use_abs_obj,
                      x_prev2=xp2, dframes_prev=dfp)
            if nf is not None:
                f = f[:, :nf]
            ff = None
            if featdim and p.get('feat') is not None:
                fv = p['feat'][fi].astype(np.float32)
                if fv.shape[0] < c.shape[0]:
                    fv = np.vstack([fv, np.zeros((c.shape[0] - fv.shape[0],
                                                  featdim), np.float32)])
                ff = torch.from_numpy(fv[:c.shape[0]]).unsqueeze(0)
            with torch.no_grad():
                s = model(torch.from_numpy(f).unsqueeze(0), feat=ff)[0].numpy()
            j = int(np.argmax(s))
            # the state the policy REACHED, plus what it should have done
            out.append((pi, fi, xp, yp, xp2, dfr, dfp))
            xp2, fp2 = xp, fp
            xp, yp, fp = float(c[j, 0]), float(c[j, 1]), fr
    return out
