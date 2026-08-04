"""R1 -- test-time adaptive input normalisation (CMN / CMVN) for the audio tower.

WHY. Both baselines freeze GLOBAL TRAINING statistics and never look at the
test recording:

  * CB_TA (audio_encoder.py:36) does `(x - self.means)/self.stds` where
    means/stds are `nn.Parameter(..., requires_grad=False)` set exactly once,
    before epoch 1, from ONE synthetic performance per training piece
    (train_model.py:143-146, `train_dataset.performances[elem][1000]['spec']`).
    They are then saved into the checkpoint and reused verbatim at eval.
  * CYOLO's `TemporalBatchNorm` (models/custom_modules.py:91) looks adaptive
    but is a stock `nn.BatchNorm1d(78, affine=False)` with the default
    `track_running_stats=True`. Under `model.eval()` that uses
    running_mean/running_var -- i.e. also global training statistics. It
    adapts during training only.

So neither model compensates for the recording channel, and this is not a
technique we are porting from CYOLO -- it is untested in this line of work.

WHY IT SHOULD MATTER ON MSMD-Rec. The models are trained on fluidsynth output
and tested on a Yamaha hybrid piano captured by a room mic. To first order a
room + mic + distance is a linear time-invariant filter: a per-frequency-band
multiplicative gain on the magnitude spectrum. The network's input is a
78-band LOG-mel spectrogram, and in the log domain a multiplicative per-band
gain is an ADDITIVE PER-BAND OFFSET:

    log(g_b * X[b,t]) = log X[b,t] + log g_b

Subtracting a per-band mean estimated from the test signal itself cancels
`log g_b` exactly, whatever it is. That is cepstral mean normalisation, the
standard channel-compensation step in robust ASR, and it is why `mean` is the
default mode here rather than `meanvar`.

WHY VARIANCE NORMALISATION IS OPTIONAL AND OFF BY DEFAULT. A static channel
shifts the mean but does not change the per-band variance, so dividing by a
locally-estimated std corrects nothing the mean has not already corrected --
while introducing a real risk: the training stds were estimated over whole
concatenated performances (full dynamic range), whereas a short window sees
much less variation. Dividing by too small a local std inflates the input and
pushes it off the manifold the network was fit on. `meanvar` is available so
that can be measured rather than assumed, and it shrinks the local std toward
the training std by `--var_shrink`.

CAUSALITY. `window` frames of history are used, never the future, so this
stays a legal online score follower. At eval the caller passes seq_len=128
(6.4 s at 20 fps), which is long enough to average over several bars but
short enough to track slow drift.

`alpha` interpolates between the frozen training mean (0.0, exact baseline
behaviour) and the fully adaptive estimate (1.0), so the probe can show a
monotone trend instead of a single on/off number.
"""
from __future__ import annotations

import torch


def adapt_stats(x_flat: torch.Tensor,
                mean_global: torch.Tensor,
                std_global: torch.Tensor,
                mode: str = 'mean',
                alpha: float = 1.0,
                var_shrink: float = 0.5,
                eps: float = 1e-5):
    """Return (mean, std) to normalise with, blending frozen train statistics
    with statistics estimated from `x_flat` itself.

    x_flat      (N, D) -- N observations of a D-dim input (D = 78 mel bands,
                or 768 MERT dims). Statistics are taken over N, i.e. over time.
    mean_global (D,)   -- the checkpoint's frozen training mean.
    std_global  (D,)   -- the checkpoint's frozen training std.
    """
    if alpha <= 0.0:
        return mean_global, std_global

    # N == 1 gives a zero-variance, self-cancelling estimate (x - x = 0), which
    # would erase the input entirely. Refuse to adapt on a single frame.
    if x_flat.shape[0] < 2:
        return mean_global, std_global

    mean_local = x_flat.mean(dim=0)
    mean = (1.0 - alpha) * mean_global + alpha * mean_local

    if mode == 'mean':
        return mean, std_global

    if mode == 'meanvar':
        std_local = x_flat.std(dim=0, unbiased=False).clamp_min(eps)
        # shrink toward the training std: a short window under-estimates
        # dynamic range, and dividing by too small a value inflates the input.
        std_local = var_shrink * std_global + (1.0 - var_shrink) * std_local
        std = (1.0 - alpha) * std_global + alpha * std_local
        return mean, std.clamp_min(eps)

    raise ValueError(f'unknown adaptive-norm mode {mode!r} (want mean|meanvar)')
