# Real-audio (MSMD-Rec) results — 2026-08-01

Tier: `third_party/cpjku_unet/data/msmd/msmd_real_performances` — 25 score
pages, real piano recorded on a Yamaha hybrid, **same typeset MSMD scores**
(so sheet/coords/coord2onset are the standard ones; no rendering needed).
Zero-shot: every model was trained on synthetic MSMD only.
Conditions: `room` = room microphone, `di-left` = direct pickup.
Metric: pct@0.5s.

| model | room | di-left | synthetic MSMD |
|---|---|---|---|
| MERT_B3_inr_subpixel | **41.8** | 55.2 | 89.8 |
| B1a_mert_swap        | **41.0** | 57.5 | 89.2 |
| MERT_B2_pitch_aux    | 39.7 | 53.2 | 86.7 |
| B3_inr_subpixel      | 27.1 | 66.2 | 87.1 |
| B4_temporal_consistency | 22.7 | **70.9** | 86.3 |
| **CB_TA (paper's own checkpoint)** | 21.5 | 63.7 | — |
| B2_pitch_aux         | 19.4 | 66.1 | 87.6 |
| B5_dense_contrastive | 17.8 | 69.0 | 87.7 |
| C2_soft_dtw          | 17.5 | 64.3 | 87.4 |
| B6_impulse_response  | 16.4 | 67.4 | 87.5 |

## 1. The harness is validated

CPJKU's **own released CB_TA checkpoint** scores 21.5 / 63.7 — inside the
range of our CBEncoder reimplementations (16.4-27.1 / 64.3-70.9). Our pipeline
reproduces their model's behaviour, so the numbers below are not an artefact.

Independently, Henkel & Widmer 2021 (Frontiers, Table 4, pct@0.5s) report for
the conditional U-Net family: **0.750 synthetic -> 0.125 real (-83.3%)**, while
CYOLO loses only 18-30%. Our CBEncoder results sit in that same regime.

## 2. MERT roughly DOUBLES real-audio robustness

On `room`, MERT models reach ~40-42% against 21.5% for the paper's own
checkpoint and 16-27% for every CBEncoder variant. Pretraining on real music
does not merely raise in-domain accuracy; it substantially narrows the
synthetic->real gap. This is the strongest result the project has.

## 3. ...but it is a CROSSOVER, not a clean win

On the clean `di-left` condition MERT is WORSE (53-57) than the CBEncoder
models (64-71). MERT trades clean-condition ceiling for degradation
robustness. Reporting only `room` would overstate the case.

## 4. B6 does not do what it was built for

B6 is impulse-response augmentation, and CB_TA-Ext.md says it is "judged on
real-audio tiers only". On the only such tier it ranks LAST on `room` (16.4).
Caveat before calling it refuted: B6 trains on a SYNTHESISED IR bank
(`synthesize_ir`), not measured room responses, so the honest reading is
"synthetic IR augmentation did not transfer", not "IR augmentation is
useless". Henkel's augmentation is the same recipe (on-the-fly IR, tempo
0.5-2.0, image shifts) yet ALL their CYOLO variants got it, so augmentation is
not what separates CUNet from CYOLO -- architecture is.

## 5. Synthetic ranking does not predict real ranking

On synthetic all six CBEncoder models sit within 1.4 points (86.3-87.7). On
`room` they span 10.7 points in a completely different order. The 86-90%
synthetic leaderboard is not evidence about real-world behaviour.

## Caveats

- 25 pages only. Per-model gaps of a few points are within noise; the
  MERT-vs-CBEncoder gap (~20 points) and the di-left-vs-room gap (~40) are not.
- CB_TA-Ext.md:630 warns against running many ablations on this tier precisely
  to avoid overfitting design choices to it. Treat per-model ordering as
  indicative, not decisive.
- The synthetic column is the 125-page msmd_test, i.e. DIFFERENT pieces. A
  matched control (same 25 pages, synthetic audio -- CPJKU's own `rp_synth`
  condition) has been built at /scratch/pmohseni/acoustic_tiers/rp_synth and
  should replace that column.
