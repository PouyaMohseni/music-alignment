# Hybrid models: combining what CB_TA-Ext and CYOLO each demonstrably do well

Target: beat `cyolo_sb` = **63.0** pct@0.5s on MSMD-Rec `room`. Our best is 44.7.

Every model here is reported on **all three tiers** via `eval_all_tiers.sh`,
because the tiers disagree strongly enough that any one of them alone misleads
(B1a: #1 synth, #8 room. B6: 4th di-left, last room).

---

## What each side actually brings, with the number that proves it

| ingredient | owner | evidence |
|---|---|---|
| **Detection formulation** (box + objectness, not dense Dice heatmap) | CYOLO | synth→real degradation **−4.3** vs our **−45** |
| **Multi-granularity** (note + bar + system) | `cyolo_sb` | **+4.9** real audio over `cyolo` (58.1 → 63.0) |
| **`z = concat(LSTM hidden, instantaneous spec)`** | CYOLO | direct non-recurrent path alongside the recurrent one |
| **Multi-scale FPN** P3/P4/P5 | CYOLO | position is resolved at several receptive fields |
| **MERT frozen audio tower** | ours | **+22** on room (37–44 vs 15–28 CBEncoder) |
| **Spectrogram tower's clean precision** | ours (CBEncoder) | **wins di-left**: 62–69 vs MERT's 51–59 |
| **Gated belief filter** (temporal prior) | ours (N3) | **+6.2** room over B1a base |
| **Pitch auxiliary loss** | ours (B2) | **+5.2** room over B1a base |

The two largest single effects — MERT's +22 and detection's −4.3-vs-−45 — live
on **opposite sides** and have never been combined. That is H1.

---

## H1 — MERT-conditioned CYOLO  ★ highest ceiling

Replace CYOLO's 78-band conv audio tower with MERT; keep the detector, FPN,
anchors and `sb` multi-class head untouched.

**Mechanism.** CYOLO's robustness comes from its *output* parameterisation and
its multi-scale head, neither of which cares what produced `z`. Its *input*
side is a plain mel-spectrogram CNN — precisely the component our sweep shows
is worth −22 points on room. So the two contributions are architecturally
independent and should compose.

**Integration points** (both required):
1. `cyolo_score_following/models/conditioning_networks.py:26-66` —
   `ContextConditioning.enc` maps a `(N,1,40,78)` window → `spec_out=32`.
   Swap for a MERT projector over `(N,1,40,768)`, pooling the 40 frames.
   Keep `kw=40` windowing, `seq_model` LSTM and `z_enc` **unchanged**, so the
   `concat(hidden, last_steps)` trick is preserved.
2. `cyolo_score_following/dataset.py:283-308` — CYOLO stores raw `signals`
   per piece and computes the spectrogram *inside* the model, so the dataset
   must carry MERT embeddings instead. Piece keys are
   `os.path.basename(score_path)[:-4]` → `{piece}_page_{n}`, which already
   matches our bank's `{piece}_tempo_{tf}.npy` naming after the tempo suffix
   is applied.

**Cost.** Highest of the four. Needs a MERT loader for CYOLO's data layout.
**Gate on job 66979946** (`cyolo_sb` reproduction) actually running — it is
still PENDING and has never executed, so there is no verified base to modify.

**Prediction.** room ≥ 60. If MERT's +22 composes even at half strength with
the detection formulation's robustness, this clears `cyolo_sb`.
**Falsifier:** if room lands near plain `cyolo_sb` (63) with synth unchanged,
MERT and the detector are redundant rather than complementary — meaning both
were fixing the same failure and the ceiling is lower than assumed.

---

## H2 — Detection head on OUR network

Keep MERT + FiLM + belief filter; replace the dense Dice heatmap output with
box regression + objectness.

**Why separately from H1.** H1 changes the audio tower *and* the whole
detector at once. H2 changes **only the output parameterisation**, so it
isolates whether that is what buys the −4.3 degradation. If H2 alone recovers
most of the gap, the FPN and anchors are incidental and H1 can be simplified;
if it recovers none, the robustness lives in the multi-scale head instead.
Either answer redirects the remaining effort, which is why this is worth a run
even though its own ceiling is lower than H1's.

**Prediction.** room +8–15. Dice on a dense heatmap is calibration-sensitive:
degraded audio flattens the map and the argmax jumps. Objectness is a *ranked*
decision and degrades more gracefully.

---

## H3 — Multi-granularity (the `sb` mechanism) in our network

Predict **note + bar + system** simultaneously; let the coarse heads gate the
fine one.

**Mechanism.** Under degraded audio, exact position is hard but *which system
am I in* stays recoverable. `cyolo`→`cyolo_sb` is +4.9 on real audio from
exactly this and nothing else.

**Cheap because the targets need no new data:** derive them by dilating the
existing `y_batch` at `dataset.py:376` — wide horizontal dilation → bar, full
staff-row band → system. Add 2 channels off the penultimate decoder stage;
Dice on each, reusing the `iterate_dataset_ext` aux-loss machinery B2/B5
already use. At inference bias fine logits by `λ·log σ(coarse)`.

**Prediction.** room +4–6, mirroring CYOLO's own ablation. Should **stack**
with H1/H2 since it is a supervision change, not an architecture change.

---

## H4 — Dual audio tower: MERT ⊕ spectrogram

Feed **both** towers and fuse, using CYOLO's `concat` idea for `z`.

**Mechanism — this one is forced by our own data.** The sweep shows a clean
double dissociation that no current model exploits:

|  | room | di-left |
|---|---|---|
| MERT models | **37–44** | 51–59 |
| CBEncoder models | 15–28 | **62–69** |

MERT buys room-robustness and *pays for it* in clean accuracy. Each tower is
better exactly where the other is worse, so a model with both inputs should be
able to dominate both columns rather than trade between them.

**Integration is unusually cheap.** Stack the two features into one array of
78+768 = 846 rows in `_patched_load_performance`; nothing in `prepare_batch`
or the network changes because it just looks like a taller spectrogram. The
dual encoder then splits its input:
- `x[:, :, :78, :]` → CBEncoder conv stack → 32-d
- `x[:, :, 78:, -1]` → last frame → MERT projector → 32-d
- concat → `Linear(64, spec_enc)`

**The one real risk:** frame alignment. MERT is resampled to 20 fps and the
spectrogram is computed at 20 fps, but they will differ by a few frames; both
must be truncated to `min(len)` or every onset label is silently offset.

**Prediction — and it is a strict one.** room ≥ 44 **and** di-left ≥ 62. It
must match the better tower on *both* columns. Matching only one means the
fusion collapsed to a single tower and the gate should be inspected.

---

## Ranking

| | ceiling | cost | blocked on |
|---|---|---|---|
| **H1** MERT-in-CYOLO | **room ≥ 60** | high | `cyolo_sb` baseline (66979946) must run |
| **H4** dual tower | room ≥44 **and** di-left ≥62 | low | nothing — buildable now |
| **H2** detection head on ours | room +8–15 | medium | nothing |
| **H3** multi-granularity | room +4–6 | medium | nothing; stacks with H1/H2 |

H1 is the only one with a credible path to 63 on its own. H3 stacks with
everything. H4 is the cheapest real test of a complementarity we have already
measured but never used.

## Protocol

Every model above is submitted through:

```bash
bash eval_all_tiers.sh <EXPERIMENT> <WRAPPER> [--after <TRAIN_JOBID>]
```

which chains synth + di-left + room off the training job, so the full grid
appears without anyone having to remember to collect it. Already wired for
`R3_mert_pitch_belief` (64849–64851) and `R2a_channel_aug` (64852–64854).
