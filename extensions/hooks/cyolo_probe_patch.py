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


def patch_int_scale_width():
    """Make --scale_width usable at all.

    eval.py:23 declares `--scale_width type=float, default=416` -- an INT
    default with a float parser. Omit the flag and cv2 gets ints and is happy;
    pass it, even at its own default value, and data_utils.py:104 calls
    cv2.resize(score, (416.0, 416.0)) which raises cv2.error. So the option has
    never been usable, which is very likely why input resolution has never been
    varied in this codebase.

    eval.py does `from ...dataset import load_dataset`, and that import runs
    when runpy executes it -- after this patch -- so rebinding the module
    attribute here is enough.
    """
    import cyolo_score_following.dataset as ds

    _orig = ds.load_dataset

    def load_dataset(paths, augment=False, scale_width=416, **kw):
        return _orig(paths, augment=augment, scale_width=int(scale_width), **kw)

    ds.load_dataset = load_dataset
    print('[PROBE] --scale_width coerced to int (upstream passes a float to cv2)',
          flush=True)


# ---------------------------------------------------------------- P-HIST/P-NOW
# The conditioning vector is z = z_enc(concat(LSTM_hidden, current_window)):
# 64 dims of AUDIO HISTORY and 32 dims of the LAST 2 SECONDS
# (conditioning_networks.py:96 and :149). Zeroing one half at a time separates
# "does it remember?" from "does it listen right now?".
#
# This matters for interpreting our own result. The network already contains an
# LSTM, yet bolting an external memory onto the DECODE was worth +4.38. If
# zeroing the LSTM half costs nothing, the model is essentially reactive and the
# recurrent path is not carrying position -- which would explain why an external
# memory had so much to add. Note also that the LSTM state only refreshes every
# kw=40 frames, so its contribution is stale by up to two seconds by design.
_ZCFG = {'mask': 'none'}


def set_z_mask(mask):
    _ZCFG['mask'] = mask
    print(f'[PROBE] z mask = {mask}', flush=True)


def _wrap_z(self):
    if getattr(self, '_z_wrapped', False):
        return
    orig, H = self.z_enc, self.seq_model.hidden_size

    def masked(v):
        m = _ZCFG['mask']
        if m == 'hist':          # drop the LSTM history, keep the present
            v = torch.cat([torch.zeros_like(v[..., :H]), v[..., H:]], -1)
        elif m == 'now':         # drop the present, keep only history
            v = torch.cat([v[..., :H], torch.zeros_like(v[..., H:])], -1)
        return orig(v)

    self.z_enc = masked
    self._z_wrapped = True


def patch_z_mask():
    from cyolo_score_following.models.conditioning_networks import ContextConditioning

    _orig_get = ContextConditioning.get_conditioning
    _orig_seq = ContextConditioning.encode_sequence

    def get_conditioning(self, x, hidden=None):
        _wrap_z(self)
        return _orig_get(self, x, hidden)

    def encode_sequence(self, x, hidden=None):
        _wrap_z(self)
        return _orig_seq(self, x, hidden)

    ContextConditioning.get_conditioning = get_conditioning
    ContextConditioning.encode_sequence = encode_sequence
    print('[PROBE] z_enc maskable (hist / now)', flush=True)


def bar_filter(sel_note, prediction_row, slack):
    """Pin the note to the predicted BAR box -- finer than the system (0.829 vs
    0.917 accuracy, but a bar is a far tighter region, so a correct one
    constrains x much harder)."""
    rows = prediction_row[prediction_row[:, -1] == 1]
    if rows.shape[0] == 0 or sel_note.shape[0] == 0:
        return sel_note
    best = rows[int(rows[:, 4].argmax())]
    cx, cw = float(best[0]), float(best[2])
    cy, ch = float(best[1]), float(best[3])
    keep = ((sel_note[:, 0] >= cx - cw / 2 - slack) & (sel_note[:, 0] <= cx + cw / 2 + slack)
            & (sel_note[:, 1] >= cy - ch / 2 - slack) & (sel_note[:, 1] <= cy + ch / 2 + slack))
    return sel_note[keep] if int(keep.sum()) > 0 else sel_note
