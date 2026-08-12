# Provenance for every number in `main.tex`

Rule for this paper: **no number goes in the draft unless it traces to a job log
or a file in this repo.** This project has already been burned twice by figures
that turned out to be the wrong column or an agent's unverified claim, so the
audit trail is part of the deliverable.

Status legend: **[V]** I verified this myself · **[A]** from an agent report,
not independently re-checked · **[P]** published, quoted from a paper.

---

## Section 3 — the two-microphone measurement (Table 1)

Source: `results/amt_bridge_eval-437566.log`, produced by
`scripts/amt_bridge_eval.py` via `amt_bridge_eval_cpu.sh`.
Transcriptions from job 402243 (`scripts/amt_transcribe_real.py`), stored under
`/scratch/pmohseni/amt_out/{kong_stock,edwards_robust}/{room,di-left}/` —
25 pages per cell, 100 files total.

| cell | value | status |
|---|---|---|
| kong room F1@50ms | 0.9116 | **[V]** |
| kong di-left F1@50ms | 0.9124 | **[V]** |
| kong room F1@100ms | 0.9546 | **[V]** |
| kong di-left F1@100ms | 0.9903 | **[V]** |
| edwards room F1@50ms | 0.9441 | **[V]** |
| edwards di-left F1@50ms | 0.9259 | **[V]** |
| edwards room F1@100ms | 0.9864 | **[V]** |
| edwards di-left F1@100ms | 0.9909 | **[V]** |

Round-trip validation, same log:
- oracle at latency $0$: **100.00** pct@0.5s **[V]** — proves the
  score-note → pixel → onset-frame path is lossless.
- oracle at latency $-0.050$: **94.79** **[V]** — independently reproduces an
  earlier in-repo figure of 93.4 under the same convention.

Non-causal / online figures:
- offline DTW, room: **98.06** **[V]**
- online greedy, room: **10.73**; di-left **26.69**; median error 8–13 s **[V]**

Measured systematic offset: constant wav-vs-MIDI shift of **−0.03 to −0.04 s**
across all four conditions **[V]** — under one frame at 20 fps.

**Caveat that must stay in the paper:** `score/{piece}.mid` is byte-identical to
`performance/{piece}.mid` — these are reproducing-piano playback, so there is
**no tempo deviation**. This is why the DTW figure is near-tautological and
must not be presented as a score-following result.

---

## Section 4 — the intervention asymmetry

| claim | value | status |
|---|---|---|
| detection tracker, no IR → IR, room | 46.0 → 71.2 (+25.2) | **[P]** Henkel & Widmer, EUSIPCO 2021 |
| our heatmap tracker, fixed real IR | 45.6 → 56.6 (+11.0) | **[V]** |
| IR bank size | 693 real measured IRs (`/scratch/pmohseni/ir_bank`) | **[V]** |
| CYOLO loads after its own filter | 663 | **[V]** `results/cyolo_repro-547365.log` |

Our 56.6: `results/eval_any-497726.log` (R2r_realir round 2) **[V]**.
Round 1 = 56.2, round 3 = 54.4 — **the run peaked at round 2 and then
regressed**, so 56.6 is final, not a waypoint. Companion tiers: di-left 64.5,
synth 87.4 **[V]**.

The bug disclosure:
- `extensions/augmentation/impulse_response.py` used `fftconvolve(mode='same')`,
  advancing audio by `(len(ir)-1)//2` samples — a **4.1–20.0 frame** label
  desync on ~50% of samples. Fixed in commit `e8320ea`, **2026-08-06**. **[V]**
- The discarded IR experiment (B6) trained **2026-07-05 – 07-08**, a month
  before the fix. **[V]** So "IR augmentation does not help us" was reading a
  bug.
- CYOLO's own convolution is `convolve(..., 'full')[:-(len(ir)-1)]`, which is
  mathematically identical to our *fixed* version **[V]** — good evidence the
  fix is right rather than merely different.

Output-parameterisation comparison (same paper, same data, same audio tower):
MM-Loc **58.5** vs CUNet **22.4** on room **[P]**, Frontiers Table 4.

---

## Section 5 — measurement problems

**Aggregation** — `AGGREGATION_FINDING.md`, recomputed from
`results/eval_any-111639.log` (25 pages, 4415 onsets):

| estimator | value | status |
|---|---|---|
| micro (pooled onsets) | 45.4 | **[V]** |
| macro over 25 pages | 46.8 | **[V]** |
| macro over 16 pieces | 52.7 | **[V]** |

Chopin Op.9 = 29.8% of the micro metric, 6.2% of piece-macro; scores 14.6 where
the rest scores 57.0 **[V]**.

**Power** — `scripts/benchmark_power_analysis.py`,
`results/analysis/benchmark_power.json`: design effect ≈ 180, effective
N ≈ 25 pieces, MDE ≈ 10 points **[V]**.
Retrain control: same recipe, two runs → 22.7 vs 16.4 **[V]**.

**Oracle inputs** — `CODA_COMPARABILITY.md`, audited at commit `dba9829`:
`system_boxes`/`bar_boxes` are required positional args of `forward()`
(`coda_model.py:314-317`), passed at the training call site
(`train.py:173-177`); loss has only system CE + bar CE + note MSE, no
localisation term **[V]**. No detection path exists in the release **[V]**.

---

## Baseline figures

| system | room @0.5s | status |
|---|---|---|
| cyolo_sb (reproduced by us) | **79.9** | **[V]** `--only_onsets`, matches published row exactly |
| cyolo_sb @0.1s | 63.0 | **[V]** |
| CYOLO | 71.2 | **[P]** |
| cyolo_sb_a (not reproducible — needs "+A" data) | 86.5 | **[P]** |
| CUNet | 22.4 | **[P]** |
| MM-Loc | 58.5 | **[P]** |

The 17-point column error (63.0/70.6 are the **0.1 s** column, not 0.5 s) is
documented in `BASELINE_CORRECTION.md` **[V]**.

---

## Deliberately NOT in the paper

- **P1 (bucketed-softmax output head).** Three failed runs, three different
  causes (module-level import; centre-of-mass over an above-half-max plateau;
  height marginalisation on multi-staff pages). It has **never been measured**
  — neither supported nor refuted. Nothing about it goes in until a clean
  number exists.
- **OMR results** (oemer F1 0.985 native / 0.9925 hires, median |Δx| 0.33 px).
  Verified **[V]**, but off-thesis for a 4-page paper. Belongs in the longer
  version.
- **The full CODA audit.** Too big to defend in 4 pages; Section 5 states only
  the part that bears on measurement. Full version → TISMIR/ISMIR.
- **cyolo+IR reproduction.** Still training (epoch ~10/50). If it converges near
  71.2 it strengthens Section 4 considerably; if it undershoots, Section 4's
  asymmetry claim weakens and must be softened.

## Open risks to the draft

1. Section 4 leans on one published delta (+25.2) against one measured delta
   (+11.0). Our own cyolo+IR run is the control that makes this a like-for-like
   comparison; until it converges, the comparison is across papers.
2. The two-mic result is on reproducing-piano playback. A human performance
   would stress transcription harder. This bounds the claim and is stated.
3. No ICASSP precedent found for an evaluation-audit paper, which is why the
   thesis is the positive measurement (Section 3) and not the audit.
