# Master plan — score following on sheet-music images

**Written 2026-08-08.** Supersedes `PLAN_2026-08-06.md` and the track docs
(`REAL_AUDIO_TRACKS.md`, `HYBRID_MODELS.md`), which remain valid as detail but
were written against numbers since corrected.

Every claim below is tagged:
**[V]** verified by me directly in code, data, or a job log ·
**[A]** reported by a research agent, not independently re-checked ·
**[E]** estimate or inference, explicitly not a measurement.

---

## 0. One-paragraph summary

We build image-conditioned score followers: given live audio and a sheet-music
image, predict the pixel position frame by frame. Our real-audio number is
**45.6** pct@0.5s on MSMD-Rec `room`; the reproducible bar is **cyolo_sb =
79.9**. Roughly two months of work assumed that gap was *acoustic* — that room
reverberation destroys the audio signal. **That assumption is now falsified.**
A stock automatic music transcriber recovers 91–95% of onsets from the identical
recordings, and the room costs it essentially nothing. The information is
present; our architecture cannot hold onto it. Separately, an audit of the
current SOTA claim (CODA, ISMIR 2026) found it is measured under an undisclosed
oracle and a different aggregator. So the work now has **two deliverables**: an
evaluation/measurement paper that is essentially finished, and a model track
with a newly-localised target.

---

## 1. Where we actually stand

### 1.1 The scoreboard (pct@0.5s, MSMD-Rec `room`, micro-aggregated)

| system | room | note |
|---|---|---|
| CODA (ISMIR 2026) | 88.3 **claimed** | macro + oracle layout boxes — see §3.1 |
| cyolo_sb_a | 86.5 | needs "+A" data we do not have; not reproducible |
| **cyolo_sb — the real bar** | **79.9** | **[V]** we reproduced it exactly |
| CYOLO | 71.2 | |
| MM-Loc (Dorfer-style bucketed softmax) | 58.5 | same lab, same data, same audio tower |
| **ours (R3: MERT + pitch-aux + belief filter)** | **45.6** | |
| **CUNet — published, our own architecture family** | **22.4** | |

**[V]** Our reproduction passes `--only_onsets` (`eval_cyolo_sota_cpu.sh:43,49`)
and lands on `63.0 @0.1s / 79.9 @0.5s`, matching CODA's copied baseline row
`.630 / .799` exactly. So the bar is self-reproduced, not trusted — and the
onset-only denominator is confirmed identical on both sides.

Two things this table says that were invisible for months:

- **We are extending a weak baseline, not failing at a strong one.** Published
  CUNet — the exact dense-heatmap + soft-Dice family we build on — scores
  **22.4**. R3 is 45.6. We have already roughly *doubled* the published version
  of our own architecture.
- **The target was misread by 17 points.** 63.0/70.6 are the **≤0.1 s** column
  of Frontiers Table 4; every number we report is ≤0.5 s. See
  `BASELINE_CORRECTION.md`.

### 1.2 The strongest architectural signal we have

**[V]** Within one paper, one lab, one dataset, one audio tower:
**MM-Loc 58.5 vs CUNet 22.4** on room. A **36-point swing attributable to the
output parameterisation alone** — bucketed softmax classification over position
versus dense soft-Dice heatmap. Corroborated externally by the OOD-segmentation
literature (Galdran et al.): soft-Dice wins in-distribution and loses
out-of-distribution, which is exactly the synth→real transition.

This is stronger evidence than anything in `HYBRID_MODELS.md`, and it points at
the output layer, not the audio front end.

---

## 2. The decisive new result: the room is not the problem

**[V]** `results/amt_bridge_eval-437566.log`, 25 real recordings, two AMT models,
two microphones on the **same take** — so the room is isolated with the
performance held exactly fixed.

| | room | di-left | Δ (room − di-left) |
|---|---|---|---|
| kong_stock onset-F1 @50 ms | 0.9116 | 0.9124 | **−0.001** |
| kong_stock onset-F1 @100 ms | 0.9546 | 0.9903 | −0.036 |
| edwards_robust @50 ms | 0.9441 | 0.9259 | **+0.018** |
| edwards_robust @100 ms | 0.9864 | 0.9909 | −0.005 |

**The room costs a transcriber ~0 points at 50 ms**, and the reverb-augmented
model is *better* on room than on the direct pickup. Our image models lose ~35
points on those same files.

**Consequence.** The prior story — "82% of the 34-point gap is acoustic" — is
**wrong**. Note-level information survives the room nearly intact. The failure
is representational: our front end plus dense-heatmap output is brittle to the
domain shift, not starved of signal. This is a far more actionable diagnosis,
and it aligns with §1.2 pointing at the output parameterisation.

### 2.1 Guards that make these numbers trustworthy

- **[V]** Oracle ceiling recomputed in-harness: **100.00** pct@0.5s at latency 0
  — the score-note → pixel → onset-frame round-trip is lossless, so no AMT
  number is confounded with a coordinate bug. (94.79 at latency −0.050,
  independently matching an earlier pass's 93.4.)
- **[V]** Constant wav-vs-MIDI offset measured at −0.03 to −0.04 s across all
  four conditions — under one frame at 20 fps. Systematic but harmless at 0.5 s.

### 2.2 What these numbers are NOT

- **[V]** Offline DTW over the transcriptions scores **98.06** on room. **This
  does not beat cyolo_sb.** It is non-causal (score following must be online),
  and near-tautological here: score MIDI is byte-identical to performance MIDI
  (Disklavier playback), so there is no tempo deviation for DTW to fight.
- **[V]** The **online** matcher is where it collapses: greedy scores **10.7**
  (room) / 26.7 (di-left) with median error of **8–13 seconds**. It loses the
  pointer and never recovers. *That*, not transcription, is the hard part of a
  causal decomposed tracker — and it is now precisely located.

---

## 3. Findings register — what ~10 agent investigations established

### 3.1 CODA's SOTA claim is not like-for-like **[V, in full]**

`CODA_COMPARABILITY.md`. Audited at commit `dba9829`.

- **CODA is handed the page layout — at training time too.** `system_boxes` and
  `bar_boxes` are **required positional arguments** of `forward()`
  (`coda_model.py:314-317`), passed straight from the MSMD npz to the training
  call site (`train.py:173-177`). The loss has exactly three terms — system CE
  over given candidates, bar CE over given candidates, note MSE — with **no box
  regression, no objectness, no IoU**. No detection path exists in the release
  at all; the inherited YOLO builder is dead code. Its own docstring: *"No
  anchors, no NMS, no objectness — pure classification over known candidates."*
  **CODA picks 1 of N given regions; CYOLO must produce the region.** The paper
  does not disclose this.
- **Aggregation swap.** CODA reports **macro** (`evaluate_batch.py:179-183`);
  CYOLO reports **micro** (`eval.py:76-83`). Its own Table 1 (macro) and Table 2
  (micro) disagree with each other.
- **Not reproducible.** No released weights; the real-audio Setting II ships no
  data, no split file, and no code path (`finalize_run.py` hard-codes the
  synthetic 94/66/28 counts). Baselines are copied from Henkel & Widmer, not
  re-run.
- **Not a defect:** thresholds, units, and the onset-only denominator are
  identical on both sides.

### 3.2 Aggregation alone is worth +7.3 points **[V, on our own data]**

`AGGREGATION_FINDING.md`. Same R3 checkpoint, three aggregators:

| aggregation | R3 room |
|---|---|
| MICRO (what we and every baseline report) | **45.4** |
| MACRO over 25 pages | 46.8 |
| **MACRO over 16 pieces — CODA's aggregator** | **52.7** |

Cause: Chopin Op.9 is **29.8% of the micro metric** but **6.2% macro-by-piece**,
and scores 14.6 where the rest scores 57.0. Macro quietly down-weights the
hardest 30% of the benchmark by ~5×. **[E]** If CODA's 88.3 is macro and it
degrades on Chopin like every model we have, its micro equivalent is plausibly
~80–84 — i.e. around cyolo_sb_a, not clear of it.

### 3.3 The benchmark cannot support small claims **[V]**

Piece-level cluster bootstrap: **design effect ≈ 180×**. Effective N is **25
pieces, not 4415 onsets**. MDE ≈ 10 points; 99 pieces would give 5, 619 would
give 2.

**[V]** Corroborated by a retrain control: 3 of 6 identical-recipe retrains
significantly beat *themselves*. B4 scored 22.7 in one run and 16.4 in another,
d = −6.29 — **larger than a typical published ablation effect**.

### 3.4 OMR on clean engraving is close to solved **[V]**

`OMR_FEASIBILITY.md` + job 452178. 20 MSMD pages, two render resolutions,
scored against MUNG ground truth at 0.5-notehead-width tolerance:

| variant | precision | recall | F1 | median \|Δx\| | p90 \|Δx\| |
|---|---|---|---|---|---|
| native | 0.9902 | 0.9807 | **0.9854** | 0.33 px (0.033 nhw) | 0.79 px |
| hires | 0.9970 | 0.9881 | **0.9925** | 0.31 px | 0.78 px |

Per-page F1 spread: min 0.949, median 0.985, max 0.996 — **no catastrophic
page**. Spurious detections 0.30–0.98%. Cost ~5 min/page.

**This substantially beats the [E] estimate** (~97–99% detection, which held;
but the coordinate precision — a third of a pixel — is far better than assumed).
Failure taxonomy is orderly: 4-ledger-line notes miss at 6.67% (5.6× base rate),
dense neighbourhoods and chord clusters next; horizontal position on the page is
essentially neutral. The domain match is exact — **[V]** DeepScores was
"generated by rendering existing MusicXML files with **Lilypond**", the same
engraver MSMD uses, so detectors trained on it are in-domain here.

### 3.5 External real-audio benchmarks barely exist **[V + A]**

`EXTERNAL_BENCHMARKS.md`. This corrected two claims I had previously relayed:

- **[V] SMR's released audio is synthetic.** The public release ships only
  `data/midi/` and `data/pdf/`; `01_prepData.ipynb` cell 3 renders it with
  `fluidsynth -F p{i}.wav default.sf2 .../midi/p{i}.mid`. The real-audio version
  (200 YouTube recordings + manual line timestamps) is **unreleased**. Drop it.
- **[V] MAcc is not our metric.** `mus_align/eval.py` compares fractional
  **measure indices** with `error_boundary=0.5` on a 100 Hz time grid —
  **[A]** ≈ ±1.10 s at MeSA-13's 2.20 s median measure, ~2× looser than
  pct@0.5s, and time-weighted rather than onset-weighted.
- **[A] MeSA-13 is the only dataset pairing real scans + real audio + fine time
  alignment that exists** — and its 13 pieces power only a ~14-point claim,
  *worse* than our current 25. It is a validation set, not a benchmark. Its
  fully-automatic baseline is **0.72**, not the 0.82 headline (which is
  human-in-the-loop).
- **[A]** Shelving ASAP/Magaloff/Zeilinger was correct: Matchmaker benchmarks
  them on symbolic scores with **no sheet images**, so they cannot score a
  pixel-position model.
- **[A]** YTSV/U-MusT has **762 h of real solo-piano audio** — useless for
  evaluation (slide-level alignment) but the most relevant corpus found for
  real-audio *pretraining*.

### 3.6 Bugs found and fixed along the way **[V]**

| bug | consequence if unfixed |
|---|---|
| `impulse_response.py` used `fftconvolve(mode='same')` | 4.1–20.0 frame label desync on 50% of augmented samples |
| `_frame()` floored instead of ceiling | dropped up to hop−1 trailing samples per file |
| `eval_all_tiers.sh` used `dependency=afterok` | cancelled all 6 chained evals (training TIMEOUTs by design) |
| cyolo `world_size` AttributeError; fork-pool deadlock | training would not start (OpenMP is not fork-safe with `Pool(8)`) |
| onnxruntime ignores `OMP_NUM_THREADS` | oemer workers spawned 64 threads each, OOM-killed at rc=137 |
| `shift_diag` fed negative onsets to mir_eval | an *optional diagnostic* killed the primary measurement |

---

## 4. What this adds up to: two deliverables

### 4.1 Deliverable A — the evaluation paper (analysis complete)

This is the contribution that is **already earned**. Nothing in it depends on
beating anyone.

Claims, all backed above:
1. The SOTA claim is measured under an **undisclosed oracle**: CODA consumes
   ground-truth system and bar boxes at train and test, which CYOLO must predict
   (§3.1).
2. It is also measured with a **different aggregator**, worth **+7.3 points** on
   our data — comparable to CODA's entire claimed margin (§3.2).
3. The benchmark's effective N is **25, not 4415**; MDE ≈ 10 points, and
   identical-recipe retrains differ by more than a published ablation (§3.3).
4. **Constructive half:** report micro *and* macro; report piece-level bootstrap
   CIs; report the acoustic tier grid (`synth` / `rp_synth` / `di-left` / `room`)
   rather than one number; and disclose which structures a model is *given*.

**Risk to be honest about:** this is a critique paper. It needs the constructive
half (4) and ideally a model result to avoid reading as sour grapes.

### 4.2 Deliverable B — the model track

Target: **79.9**. Current: **45.6**. The §2 result says the audio side is
healthy, so effort goes to the output parameterisation and the temporal decoder.

---

## 5. Priorities, in order

### P1 — Output parameterisation (highest expected value)

**Why first.** §1.2 gives a 36-point same-lab swing from exactly this change
(MM-Loc 58.5 vs CUNet 22.4). §2 independently says the failure is not the audio.
Two unrelated lines of evidence converging on the output layer is the strongest
signal in this document.

**Do:** replace the dense soft-Dice heatmap with a ranked/classification output
— bucketed softmax over x position (MM-Loc style), or box + objectness (CYOLO
style, = H2 in `HYBRID_MODELS.md`).

**Prediction:** room +10–20.
**Falsifier:** if it moves room by <5 with synth unchanged, the output layer is
not the bottleneck and P2 becomes primary.

### P2 — MERT audio tower inside CYOLO's detector (H1)

The two largest effects we have measured live on opposite sides and have never
been combined: MERT is worth +22 on room; the detection formulation degrades
−9.4 synth→real where our dense heatmap degrades ~−40.

**Blocked on one thing.** **[V]** `train_cyolo_repro_gpu.sh:98-106` passes
`--train_sets --val_sets --config --augment --dump_root --log_root --tag
--num_workers` and **no IR flag** — so our reproduction reproduces the published
**no-IR** row. Henkel & Widmer (EUSIPCO 2021) report CYOLO **without** IR = 46.0
room and **with** IR = 71.2 (+25.2). We must confirm whether their `train.py`
exposes an IR path and relaunch with it, or we are modifying the wrong base.
**Zero API cost — pure cluster work.**

**Prediction:** room ≥ 60.
**Falsifier:** if room lands near plain cyolo_sb with synth unchanged, MERT and
the detector are redundant rather than complementary.

### P3 — The online matcher (newly opened by §2.2)

Greedy loses the pointer and sits 8–13 s away from truth. A causal monotonic
matcher with re-entry — this is what `M1.md` was designed for — is now a
*measured* gap rather than a speculative one. Note this is the same failure mode
as the repeat-ambiguity diagnostic.

### P4 — Multi-condition / IR augmentation, **de-prioritised**

§2 removes most of its rationale: if the room does not destroy the signal, there
is less to be robust *to*. Keep the fixed `impulse_response.py` and the degraded
MERT bank, but stop treating acoustic augmentation as the main lever.

---

## 6. Explicitly ruled out

| direction | why |
|---|---|
| **SMR as a benchmark** | **[V]** its released audio is fluidsynth-synthetic |
| **ASAP / Magaloff / Zeilinger / Batik / Vienna4x22** | **[A]** symbolic scores, no sheet images — cannot score a pixel model |
| **CollabScore** | **[A]** contains no audio at all; it is an OMR ground-truth set |
| **SMT / Zeus / homr for OMR** | **[A]** better transcribers, but structurally discard pixel coordinates |
| **Chasing CODA's 88.3 head-on** | it is macro + oracle-boxed; the honest target is 79.9 |
| **More CB_TA-Ext variants** | no non-detection model in the family has exceeded 58.5; the ceiling is structural |

---

## 7. The decomposed pipeline: what it is and is not

A standing scope question, recorded so it does not drift again.

**It is a diagnostic.** Audio → AMT notes ⟷ score → OMR notes → position is
**offline alignment**, a different task from online score following. §3.4 shows
OMR is not the blocker (F1 0.985–0.993, sub-pixel Δx) and §2 shows AMT is not
either — so the decomposition *would* work, and that is precisely what makes it
useful as a measurement: **it proves the information is present and localises
the failure to our architecture.**

**It is not the paper.** Pivoting to it would abandon online score following.
The OMR and AMT results earn **one table row each** as diagnostics, and the
effort goes back to §5.

---

## 8. Open questions

1. **[V, unresolved]** Does CYOLO's `train.py` expose an IR augmentation path?
   Gates P2. Cheap to answer.
2. **[E]** Magnitude of CODA's oracle-layout advantage. Unbounded from code —
   CODA cannot run without boxes and none of its five ablations removes them.
3. **[A]** Whether CYOLO's *published* table used `--only_onsets`. Our
   reproduction does and matches exactly, which is strong evidence it did.
4. **[A]** MeSA-13 redistribution terms — the repo has **no LICENSE file**, and
   audio provenance is mixed. Fine to evaluate and cite; not to redistribute.
5. Does the AMT result hold on a *human* performance? The MSMD-Rec recordings
   are Disklavier playback with no tempo deviation, which is an easier setting.

---

## 9. Immediate next actions

| # | action | cost | blocks |
|---|---|---|---|
| 1 | Check CYOLO `train.py` for an IR flag; relaunch `cyolo_sb` with it | cluster only | P2 |
| 2 | Build the P1 output head (bucketed softmax / objectness) on our best MERT base | cluster only | — |
| 3 | Draft the evaluation paper — §4.1 is complete and pushed | writing | — |
| 4 | Fold the OMR + AMT diagnostics in as one table row each | writing | — |

**No further research agents are queued.** The remaining work is building and
writing.

---

## 10. Document map

| file | holds |
|---|---|
| `BASELINE_CORRECTION.md` | the 17-point column error; corrected scoreboard |
| `AGGREGATION_FINDING.md` | micro vs macro, measured on our data |
| `CODA_COMPARABILITY.md` | full CODA audit with file:line evidence |
| `OMR_FEASIBILITY.md` | OMR survey + the measured oemer result |
| `EXTERNAL_BENCHMARKS.md` | MeSA-13 / SMR / everything ruled out |
| `HYBRID_MODELS.md` | H1–H4 designs (numbers superseded; mechanisms stand) |
| `M1.md` | monotonic alignment — now motivated by §2.2 |
| `results/analysis/` | power, calibration, OMR scored JSON |
