"""Take the frozen cyolo_sb apart and see which parts carry the signal.

Everything here is a TEST-TIME intervention on the released checkpoint -- no
training, no parameters. The point is to learn how the model works, not only to
chase a number, so each probe is designed to have an informative answer whether
it helps or hurts.

  P-FPN    Suppress one of the three detection scales (P3/8, P4/16, P5/32).
           Which resolution actually finds the note? If a scale contributes only
           distractors, removing it should HELP -- and if removing any scale is
           harmless, the model is carrying two thirds of a detection head for
           nothing.

  P-FILM   Scale the audio conditioning: gamma' = 1 + s(gamma-1), beta' = s*beta.
           s=0 makes the network audio-BLIND while leaving every weight intact,
           so it measures how much of the score is actually driven by listening
           rather than by score-image priors. s>1 asks whether the trained
           conditioning is under-confident. This is the cleanest available test
           of "does it really use the audio", and it costs one eval.

  P-SYS    Constrain the note to the predicted SYSTEM. The system readout scores
           0.917 and the bar 0.829, while note tracking sits at 84.7 -- the
           model localises the staff far more reliably than the note, and the
           decoder currently throws that away completely. Knowing the system
           pins the note to one row of the unrolled score, which is exactly the
           ambiguity that makes repeated passages hard.
"""
from __future__ import annotations

import os

import torch

_CFG = {
    'drop_scales': set(),      # subset of {0,1,2} = P3/8, P4/16, P5/32
    'film_scale': 1.0,
    'sys_constrain': 0.0,      # 0 = off; else IoU-ish slack in pixels
}


def configure(drop_scales=(), film_scale=1.0, sys_constrain=0.0):
    _CFG['drop_scales'] = {int(s) for s in drop_scales}
    _CFG['film_scale'] = float(film_scale)
    _CFG['sys_constrain'] = float(sys_constrain)
    print(f'[PROBE] drop_scales={sorted(_CFG["drop_scales"])} '
          f'film_scale={_CFG["film_scale"]} sys_constrain={_CFG["sys_constrain"]}',
          flush=True)


def patch_probes():
    from cyolo_score_following.models.custom_modules import FiLMConv
    from cyolo_score_following.models.yolo import Detect

    # ---- P-FPN: suppress a detection scale by zeroing its objectness.
    # Detect.forward already tags every candidate with its layer index in the
    # last channel, so this is a clean cut with no reindexing.
    _orig_detect = Detect.forward

    def detect_forward(self, x):
        z, xs = _orig_detect(self, x)
        drop = _CFG['drop_scales']
        if drop and isinstance(z, torch.Tensor) and z.numel():
            layer = z[..., -1]
            for s in drop:
                z[..., 4] = torch.where(layer == s, torch.zeros_like(z[..., 4]), z[..., 4])
        return z, xs

    Detect.forward = detect_forward

    # ---- P-FILM: dial the conditioning between "audio-blind" and "amplified".
    _orig_film = FiLMConv.forward

    def film_forward(self, x, z):
        s = _CFG['film_scale']
        if s == 1.0:
            return _orig_film(self, x, z)
        x = self.norm(self.conv(x))
        gamma = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(z).unsqueeze(-1).unsqueeze(-1)
        # interpolate towards the IDENTITY transform, not towards zero: scaling
        # gamma itself to 0 would kill the activation entirely rather than
        # removing the audio's influence on it.
        gamma = 1.0 + s * (gamma - 1.0)
        beta = s * beta
        return self.act(gamma * x + beta)

    FiLMConv.forward = film_forward
    print('[PROBE] Detect.forward and FiLMConv.forward patched', flush=True)


def system_filter(sel_note, prediction_row, slack):
    """Keep only note candidates lying inside the predicted system box.

    sel_note        (K, >=6) class-0 candidates for ONE sample
    prediction_row  (N, >=6) every candidate for that sample, all classes
    Returns the filtered candidates, or the originals if the constraint would
    empty the set -- a probe must never manufacture a no-detection frame.
    """
    sysrows = prediction_row[prediction_row[:, -1] == 2]
    if sysrows.shape[0] == 0 or sel_note.shape[0] == 0:
        return sel_note
    best = sysrows[int(sysrows[:, 4].argmax())]
    cy, ch = float(best[1]), float(best[3])
    lo, hi = cy - ch / 2 - slack, cy + ch / 2 + slack
    keep = (sel_note[:, 1] >= lo) & (sel_note[:, 1] <= hi)
    return sel_note[keep] if int(keep.sum()) > 0 else sel_note
