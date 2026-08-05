# Baseline correction (2026-08-04) — the target was wrong by 17 points

**Verified directly, not taken from an agent's word.**

## What was wrong

Throughout the real-audio work the bar was quoted as `cyolo_sb = 63.0` and
`cyolo_sb_a = 70.6` on MSMD-Rec `room`. **Those are the ≤0.1 s column of
Frontiers Table 4, not ≤0.5 s** — which is the metric every one of our own
numbers uses.

Proof, both independent:

1. `cyolo_score_following/eval.py:65` hardcodes
   `thresholds = [0.05, 0.1, 0.5, 1.0, 5.0]` — so column 3, not column 2, is 0.5 s.
2. Our own reproduction, `results/eval_cyolo_sota-66914671.log`, prints for
   cyolo_sb_a on room: `<=0.05: 68.2 / <=0.1: 70.6 / <=0.5: 86.5 / <=1.0: 89.1 /
   <=5.0: 98.1`, matching published `0.682/0.706/0.865/0.891/0.981` exactly.
   70.6 is unambiguously the 0.1 s cell.

## Corrected table (pct@0.5s, MSMD-Rec room)

| model | room @0.5s | previously quoted |
|---|---|---|
| CYOLO | **71.2** | 58.1 (was the 0.1s cell) |
| **CYOLO-SB — the reproducible bar** | **79.9** | 63.0 (was the 0.1s cell) |
| CYOLO-SB+A (not reproducible, needs "+A" data) | 86.5 | 70.6 (was the 0.1s cell) |
| **CUNet — published, our own architecture family** | **22.4** | never quoted |
| MM-Loc (Dorfer-style bucketed softmax) | 58.5 | never quoted |
| **our R3 (MERT + pitch-aux + belief filter)** | **45.6** | — |

The gap to beat is therefore **34 points (45.6 → 79.9)**, not 17.

## Two things this reframes

**We are not failing — we are extending a weak baseline.** Published CUNet, the
exact dense-heatmap + Dice family we build on, scores **22.4** on room. R3 is
45.6. We have already roughly **doubled** the published version of our own
architecture. That was never visible while the comparison was against the wrong
column.

**CYOLO's "4.3-point synth→real drop" is only the non-reproducible +A model.**
Reproducibly, at 0.5 s: CYOLO 88.5→71.2 = −17.3; CYOLO-SB 89.3→79.9 = −9.4.
Ours is ~−40. Still a large gap, but not the 45-vs-4.3 chasm that framed the
earlier track design.

## The strongest lead this surfaced

A **within-paper, same-lab, same-data** controlled comparison in Frontiers
Table 4: **MM-Loc = 58.5** on room vs **CUNet = 22.4**, with essentially the
same audio tower. A 36-point swing attributable to the OUTPUT PARAMETERISATION
(bucketed softmax classification over position vs dense soft-Dice heatmap).
This is much stronger evidence than anything in HYBRID_MODELS.md for attacking
the loss/output formulation first. It is corroborated by the OOD-segmentation
literature (Galdran et al.) finding soft-Dice wins in-distribution and loses
out-of-distribution.

## Bug found in passing

`extensions/augmentation/impulse_response.py:85` uses
`fftconvolve(waveform, ir, mode='same')`, which shifts the signal by ~half the
IR length **relative to its onset labels**. B6's catastrophic 15.6 on room is
therefore not evidence that IR augmentation fails — the experiment is invalid.

Confined to that one file: `scripts/precompute_mert_augmented.py` (the R2 bank)
uses `mode='full'` then truncates, which is correct. R2's 6615-piece degraded
bank is unaffected.

## Claim that did NOT survive checking

An agent reported that all CB_TA-Ext models were trained without tempo
augmentation, citing `mymodel/cpjku_adapter/train_official.py:305`
(`'tempo_factors': [1000]`). That is an older, separate code path. Our runs go
through `third_party/cpjku_unet/audio_conditioned_unet/train_model.py` with
`--config configs/msmd_aug.yaml`, which sets
`tempo_factors: [500, 750, 950, 1000, 1050, 1250, 1500]`. Tempo augmentation is
ON for the CB_TA-Ext line.
