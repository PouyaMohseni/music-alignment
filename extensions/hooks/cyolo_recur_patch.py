"""Change WHEN the recurrent state is allowed to see the recent past.

THE STRUCTURAL FACT
-------------------
z = z_enc([ LSTM hidden (64) , encoding of the last 40 frames (32) ]).

The present half is fresh: `last_steps = x[-40:]` always ends at the current
frame. The history half is not. `encode_samples` chops the performance so far
into non-overlapping 40-frame chunks ANCHORED AT FRAME 0:

    x[: kw * (T // kw)]        kw = 40, i.e. 2 s at 20 fps

so the newest chunk the LSTM ingests ends at 40*floor(T/40) -- up to 39 frames,
almost two seconds, behind the audio the detector is being asked about. The
64-dim half of the conditioning vector is stale by construction, and it is
stale by exactly as much as the phase of T against a fixed grid, which is an
artefact of where the recording happens to start.

THE FIX, WITH NO NEW PARAMETERS
-------------------------------
Anchor the chunks at the END instead:

    x[T - kw * (T // kw) :]

Same chunk length, same chunk count, same LSTM, same weights -- the newest chunk
now ends exactly at T. This is not off-distribution: training also feeds
non-overlapping 40-frame chunks, and the LSTM has no way to observe absolute
position, so only the phase changes. What the model loses is the oldest partial
chunk, which the start-anchored version was discarding at the other end anyway.

`window` tests the other half: keep the 40-frame shape the Linear demands but
zero everything before the last W frames, so we can ask whether the present half
needs its full 2 s or is carried by the tail.

NOTE: `get_conditioning` -- the obvious place to look, with its explicit
`step_cnt == kw` gate -- is ONLY used by test.py, the real-time demo. eval.py
runs Model.forward -> encode_sequence. Patching the demo path would have
produced a sweep of identical numbers.
"""
from __future__ import annotations

_CFG = {'anchor': 'start', 'window': 0}


def configure(anchor='start', window=0):
    _CFG['anchor'] = anchor
    _CFG['window'] = int(window)


def patch_encode_samples():
    import torch
    import torch.nn.functional as F

    from cyolo_score_following.models.conditioning_networks import ContextConditioning

    def encode_samples(self, x):
        # verbatim upstream except for the two marked lines
        last_steps = []
        zero_lengths = []
        for i in range(len(x)):

            if x[i].shape[0] < self.kw:
                padding = self.kw - x[i].shape[0]
                x[i] = F.pad(x[i], (0, 0, padding, 0), mode='constant')

            ls = x[i][-self.kw:]
            w = _CFG['window']
            if 0 < w < self.kw:                                   # (1) present half
                ls = ls.clone()
                ls[:self.kw - w] = 0
            last_steps.append(ls.unsqueeze(0))

            T = x[i].shape[0]
            n = T // self.kw
            seq = (x[i][T - self.kw * n:] if _CFG['anchor'] == 'end'
                   else x[i][:self.kw * n])                       # (2) history half
            stacked = torch.stack(seq.split(self.kw)).unsqueeze(1)

            if stacked.shape[0] == 1:
                zero_lengths.append(i)
                x[i] = torch.zeros_like(stacked)
            else:
                x[i] = stacked

        lengths = [spec.shape[0] for spec in x]
        last_steps = self.enc(torch.stack(last_steps))

        x = self.enc(torch.cat(x))

        x = list(torch.split(x, lengths))

        for idx in zero_lengths:
            x[idx] = torch.zeros(1, x[idx].shape[-1], device=x[idx].device)
            lengths[idx] = 1

        return x, last_steps, lengths

    ContextConditioning.encode_samples = encode_samples
    ContextConditioning._recur_patched = True
    print(f"[RECUR] history anchor={_CFG['anchor']} "
          f"present window={_CFG['window'] or 40}/40 frames", flush=True)
