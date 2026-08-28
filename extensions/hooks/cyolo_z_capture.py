"""Capture z, the audio conditioning vector the detector is actually steered by.

The selector currently ranks candidates on box geometry and a scalar objectness.
It therefore cannot express anything audio-dependent: it knows a candidate is 40
px to the right of the last position and that the detector gave it 0.31, but not
what the music is doing. Objectness does carry the audio, but compressed to one
number per box, and by the time the selector sees it the information that would
separate two similarly-confident boxes is gone.

z is the 128-dim vector every FiLM layer in the network is modulated by --
zeroing it collapses the model to 2.6 pct@0.5s, so it is the whole audio side of
the model. `Model.predict(score, z)` receives it directly, which makes it one
patch to capture and 512 bytes per frame to store.
"""
from __future__ import annotations

import numpy as np

LAST_Z = {'z': None}


def patch_capture_z():
    from cyolo_score_following.models.yolo import Model

    if getattr(Model, '_z_captured', False):
        return
    _orig = Model.predict

    def predict(self, x, z):
        LAST_Z['z'] = z.detach().cpu().numpy().astype(np.float32)
        return _orig(self, x, z)

    Model.predict = predict
    Model._z_captured = True
    print('[Z] capturing the conditioning vector', flush=True)
