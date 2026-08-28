"""Capture the 128-dim backbone feature each candidate is scored from.

THE FACT THAT MOTIVATES THIS
----------------------------
cyolo_sb has 1,464,613 parameters. The function that produces the objectness we
rank candidates by is `Detect.m[0]`, a single 1x1 convolution:

    Detect_17.m.0.weight   (15, 128, 1, 1)   1,920 parameters
    Detect_17.m.0.bias     (15,)                15

1,935 parameters, 0.13% of the model, mapping a 128-dim feature vector per grid
cell to three anchors x (box, objectness). Everything our decoder fights with --
the ranking that puts the right box below thirty wrong ones -- comes out of
that one conv.

So a learned re-ranker of comparable size is not "adding capacity" in the sense
that failed twenty-two times on this project. It is re-fitting the smallest
component in the network, the one directly responsible for the measured failure,
on an objective the original was never trained for: the original conv was fitted
to detect, not to rank against a temporal prior.

WHY DUMP FEATURES RATHER THAN FINE-TUNE THE CONV
------------------------------------------------
Fine-tuning it in place needs backprop through the frozen backbone, three hours
per epoch over 353 pieces. The conv's INPUT is 128 numbers per grid cell, and we
only care about cells that produced a candidate. Dumping those turns the problem
into offline supervised learning over 165k (frame, candidate) pairs that trains
in minutes -- and it lets the re-ranker see the features AND the temporal context
at once, which the conv structurally cannot.

Detect.forward does `x[i] = self.m[i](x[i])`, so the features are the tensor on
the way in. Candidate j at scale i sits at anchor a, cell (gy, gx) with
j = a*ny*nx + gy*nx + gx, which is the flattening view(bs, -1, no+1) applies
after permute(0, 1, 3, 4, 2).
"""
from __future__ import annotations

import numpy as np

LAST_FEAT = {'f': None, 'shape': None}


def patch_capture_feat(scale: int = 0):
    """scale 0 = P3, which is where every class-0 (note) candidate comes from:
    the probe suite measured only_P3 = 84.7 against a full-model 84.7."""
    import torch

    from cyolo_score_following.models.yolo import Detect

    if getattr(Detect, '_feat_captured', False):
        return
    _orig = Detect.forward

    def forward(self, x):
        f = x[scale]
        LAST_FEAT['f'] = f.detach()                     # (bs, C, ny, nx)
        LAST_FEAT['shape'] = (self.na, f.shape[2], f.shape[3])
        return _orig(self, x)

    Detect.forward = forward
    Detect._feat_captured = True
    print(f'[FEAT] capturing backbone features at scale {scale}', flush=True)


def gather(batch_idx, cand_idx):
    """Feature vectors for the given flat candidate indices of one batch item."""
    f = LAST_FEAT['f']
    if f is None or LAST_FEAT['shape'] is None:
        return None
    na, ny, nx = LAST_FEAT['shape']
    j = np.asarray(cand_idx, np.int64)
    gy, gx = (j % (ny * nx)) // nx, j % nx
    v = f[batch_idx, :, gy, gx]                          # (C, n) -> transpose
    return v.t().cpu().numpy().astype(np.float16)
