# Real-audio tracks — target: beat `cyolo_sb` (63.0 pct@0.5s on MSMD-Rec `room`)

Objective set 2026-08-03: optimise for REAL audio, not synthetic MSMD.

## Where we stand

| model | room | di-left | synth |
|---|---|---|---|
| **`cyolo_sb`** (Henkel & Widmer 2021) | **63.0** | — | 90.8 |
| `cyolo` | 58.1 | — | — |
| N3_belief_propagation (ours, best) | 44.7 | 58.5 | 89.3 |
| MERT_B2_pitch_aux | 43.7 | 57.1 | 87.9 |
| B1a_mert_swap (clean MERT base) | 38.5 | 55.2 | **90.0** |
| B3_inr_subpixel (best CBEncoder) | 27.9 | 64.4 | 86.8 |
| B6_impulse_response | 15.6 | 68.0 | 87.7 |

`cyolo_sb_a` (70.6) is **not reproducible** — "+A" is extra scanned-score data
not in the Zenodo release. `cyolo_sb` at 63.0 is the real bar.

Two facts drive every track below:

1. **Synthetic rank does not predict real rank.** B1a is #1 on synth (90.0) and
   #8 on room (38.5). B6 is 4th on `di-left` (68.0) and *last* on `room` (15.6).
2. **All the real-audio headroom is on the MERT path.** Every MERT model scores
   37–44 on `room`; every CBEncoder model scores 15–28. But CBEncoder *wins* on
   clean `di-left` (62–69 vs 51–59). MERT buys room-robustness and pays for it
   in clean accuracy.

## A claim that did not survive inspection

I initially proposed that CYOLO's robustness came from `TemporalBatchNorm`
adapting normalisation to the test recording, unlike CB_TA's frozen train-set
stats. **This is wrong.** `TemporalBatchNorm` (`models/custom_modules.py:91`) is
a stock `nn.BatchNorm1d(78, affine=False)` with the default
`track_running_stats=True`, so under `model.eval()` it uses running statistics
accumulated during training — also global. Both models freeze global training
statistics at test time.

That kills it as an *explanation*, but reframes it as an *opportunity*: nobody
in this line of work adapts normalisation to the test recording. See R1.

---

## R1 — Channel-invariant normalisation

**Claim.** The input is a 78-band **log**-mel spectrogram. A room + mic +
distance is, to first order, a per-band multiplicative gain, and in the log
domain that is an **additive per-band offset**:
`log(g_b · X[b,t]) = log X[b,t] + log g_b`. Subtracting a per-band mean
estimated from the test signal itself cancels `log g_b` exactly — this is
cepstral mean normalisation, the standard channel compensation in robust ASR.
CB_TA instead divides by statistics measured on fluidsynth output
(`train_model.py:143-146`), so the offset passes straight through.

**Why variance normalisation is off by default.** A static channel shifts the
mean but not the per-band variance, so `meanvar` corrects nothing extra while
risking a real harm: training stds were measured over whole performances (full
dynamic range) and a 6.4 s window sees far less, so dividing by too small a
local std inflates the input off the manifold. `meanvar` is available with
shrinkage so this is measured rather than assumed.

- Code: `extensions/audio_encoders/adaptive_norm.py`,
  `extensions/hooks/adaptive_norm_patch.py`, `eval_adaptnorm_cpu.sh`
- **Zero retrain** — only the normalisation constants change, no weights.
- `alpha=0` reproduces the frozen-stats baseline bit-for-bit and is the
  built-in control that the patch is a no-op when disabled.
- Status: 8 CPU jobs (50350–50357) on `room` across MERT_B2_pitch_aux,
  B3_inr_subpixel, B6_impulse_response × {alpha 0, alpha 1 mean, meanvar}.
- Trained variant is **deliberately gated** on the probe. Train-time CMN needs
  a normalisation window matched between train (BPTT chunks) and eval
  (seq_len 128); that is only worth building if the operator helps at all.

## R2 — Acoustic domain randomisation on the MERT path

**Claim.** B6 already tried this and came last on the tier it was built for.
Two things were wrong, both fixed:

1. **Wrong branch.** It augmented CBEncoder, which loses to MERT by ~20 points
   on `room` before any augmentation.
2. **Wrong nuisance.** Reverb only. Reverb is *temporal smearing*; the dominant
   synth→real difference is the *static per-band gain* of R1's analysis.
   `random_tilt()` adds exactly that, ±12 dB, smooth in log-frequency.

MERT is frozen and read from `.npy`, so waveform augmentation is invisible
unless re-encoded — hence a second 6615-piece embedding bank.

- Code: `scripts/precompute_mert_augmented.py`, `precompute_mert_augmented.sh`
- DSP validated: overlap-add round-trips to 3e-16; tilt is smooth,
  frequency-dependent, within its prior. `_frame()` needed a **ceil** — floor
  dropped up to `hop-1` trailing samples, truncating every performance's tail.
- No test leakage: degradation priors are fixed and seeded from the piece key;
  nothing is measured from MSMD-Rec.
- Status: pilot shard 50660 queued. Full `--array=1-39` follows on success.

### R2a — the zero-precompute counterpart (running)

Time-constant per-dimension affine on the MERT embedding, drawn **once per
forward call** and held across the chunk. Per-frame resampling would be
ordinary noise the model averages away; held constant, the only way to be
invariant is to stop relying on absolute per-dimension levels.

It is a *proxy* — it perturbs MERT's output where a real room perturbs its
input, and a frozen nonlinear encoder does not commute with the two. Running it
alongside R2 answers whether 6615 renders buy anything, which is a result
either way.

- Code: `extensions/hooks/channel_aug_patch.py`,
  `extensions/hooks/run_train_r2a_channel_aug.py`, `train_r2a_channel_aug.sh`
- Warm-starts from `B1a_mert_swap` (38.5), the **clean** base, so the delta is
  attributable to augmentation alone rather than confounded with pitch aux.
- Status: **GPU job 50907**.

## R3 — Stack the two best real-audio ingredients

`N3_belief_propagation` (44.7) and `MERT_B2_pitch_aux` (43.7) are the two
largest gains over the shared B1a base (38.5) and have never been combined.
They attack different failure modes and share no parameters:

- the belief filter is a **temporal** prior — it bounds inter-frame motion and
  keeps an explicit escape probability, so one bad audio frame cannot teleport
  the estimate. Degraded audio produces exactly that kind of frame.
- the pitch loss is a **representational** constraint — forcing FiLM features
  to stay pitch-predictive stops the tower leaning on timbre/channel cues that
  do not survive a change of piano.

- Code: `extensions/hooks/run_train_r3_mert_pitch_belief.py`,
  `train_r3_mert_pitch_belief.sh`
- Warm-starts from MERT_B2_pitch_aux with the belief gate zero-initialised, so
  step zero reproduces that model exactly.
- Eval reuses `run_eval_native_n3_belief_propagation.py` — identical
  architecture, only the training loss differs.
- Status: **GPU job 50856**.

## R4 — Multi-granularity coarse-to-fine supervision  *(specified, not built)*

**Claim.** `cyolo` → `cyolo_sb` is +4.9 on real audio, and the only difference
is that `sb` predicts **bar and system** boxes alongside note position
(`nc: 3`, plus anchor sets `[61,32 …]` and `[249,33 …]`). Under degraded audio
exact position is hard but *which system am I in* stays recoverable, so the
coarse head survives and can gate the fine one.

Port to CB_TA:
- coarse targets need **no new data** — derive them by dilating the existing
  `y_batch` in `dataset.py:376` (wide horizontal dilation → bar; full staff-row
  band → system).
- add 2 output channels off the penultimate decoder feature map; Dice loss on
  each, reusing the `iterate_dataset_ext` aux-loss machinery B2/B5 already use.
- at inference, bias the fine logits by `λ·log σ(coarse)`.

Not built: the head + targets + inference gating + smoke test is more than I
could validate this session, and shipping it unsmoke-tested has bitten this
project repeatedly (zero-gate dead branch, `initialize_weights` clobbering
zero-init, three separate `sed` truncations).

### R4-alt — MERT inside the CYOLO detector

Combine the two things that demonstrably work: MERT (+22 on `room` over
CBEncoder) and CYOLO's detection formulation (−4.3 synth→real vs our −45).
Integration point is `ContextConditioning.enc`
(`models/conditioning_networks.py:26-66`): replace the 78-band conv stack with
a MERT projector over `(N, 1, 40, 768)`, keeping the `kw=40` windowing, LSTM
and `z_enc` untouched. Also needs a MERT loader for CYOLO's dataset layout.

Sensible to gate on **job 66979946** (`cyolo_sb` reproduction) actually
running — it is still PENDING and has never executed, so there is no verified
base to modify yet.

---

## Honest assessment of the 63.0 target

Best current is 44.7. Nothing incremental closes +18. Of these tracks only
**R2** (sim2real domain randomisation) and **R4-alt** (architecture change)
have that order of magnitude; R1, R2a and R3 are each plausibly worth a few
points and are cheap. R2 is the one to push hardest.
