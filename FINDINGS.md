# Where the remaining error is, and what does not reach it

Measured 2026-09-09 on MSMD-Rec `room`, shipped model `vel_p8` at **93.42**.

## The decomposition that governs everything

| | | |
|---|---|---|
| **A** | deployed, conditions on its own history | **93.42** |
| **B** | teacher-forced, conditions on the true history | **98.07** |
| **C** | real-time oracle, also knows the answer | **99.57** |
| **B − A** | **error propagation** | **4.65** (76%) |
| **C − B** | **ranking quality** | **1.50** (24%) |

Ranking is nearly solved. Given a correct history the scorer sits 1.50 from an
oracle. Three quarters of the remaining gap is the tracker inheriting its own
mistakes. Effort on features or capacity is bounded by ~1.5 points.

How fast error accumulates (snap the history to truth every *k* onsets):

| k | 1 | 2 | 3 | 5 | 10 | 20 | 50 | never |
|---|---|---|---|---|---|---|---|---|
| acc | 98.07 | 97.01 | 96.46 | 96.31 | 95.90 | 95.13 | 94.48 | 93.42 |

## Ceilings

| ceiling | value |
|---|---|
| offline, unconstrained | 100.00 |
| offline, monotone | 99.90 |
| **real-time, unbounded reach** | **99.57** |
| real-time, ≤200 px forward | 98.41 |
| real-time, ≤100 px forward | 96.63 |
| real-time, ≤50 px forward | 95.30 |

* **Being real-time costs 0.33 points.** Latency is not a lever; this is why
  fixed-lag lookahead lost.
* **Monotonicity costs 0.10.** Backward tolerance is not a lever.
* **Zero frames on room lack a correct candidate.** The detector is not the
  bottleneck and retraining it cannot help room.
* The "96.0 causal ceiling" quoted for months is none of these. It matches
  real-time with a ~100 px forward window — an artefact of our prior's
  tightness, not a property of causality.

## Twelve mechanisms that failed

Tempo tracked inside the prior (93.42 → 90.62) · forward reach widened
(σ 30/60/120 all worse) · forward-asymmetric σ · a vertical prior term
(λ_y=2 → 71.32) · system-box constraint (slack 10 → 84.72) · jump floor −8
(+0.46 on `do`, **−2.34 on room**) · committee tracking, independent and
resampling, seed- and prior-diversified (best −0.31) · robust state by least
squares, Theil–Sen and shrinkage (monotone in smoothing; k=12 → −9.64) · hard
gated recovery · soft belief-weighted recovery · score ensembling · DAgger.

## Why they all failed — one reason, two halves

At 93.4% the last position is exact ~95% of the time.

* Any **unconditional** intervention corrupts those 95% to rescue the 5%, and
  loses on the arithmetic.
* Any **conditional** intervention must know it is lost. Lostness *is*
  detectable — `obj_chosen` separates at **AUC 0.805**, while position is
  uninformative at 0.558 — but precision never exceeds **0.33** at a 5% base
  rate, so acting destroys two good locks per save.

Corollary: local smoothing cannot work, because step sizes vary **6× within a
piece** (p10 1.59, p90 9.49 px/frame), so the trajectory is not locally linear.
The same fact explains why a conservative `fwd_px=6.0` beats a correctly
centred tracked tempo.

## The methodological lesson

**A ceiling says what is achievable, not what is actionable.** Four separate
inferences drawn from oracle ceilings on 2026-09-09 each produced a mechanism
that lost on room. Measure the mechanism; never infer it from the bound.
