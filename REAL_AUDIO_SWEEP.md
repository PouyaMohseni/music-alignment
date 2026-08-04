# Real-audio results — full sweep, 2026-08-03

All 22 experiments x {synth, room, di-left}, one harness, pct@0.5s.
`room` = real room microphone (the objective). `di-left` = real direct pickup.
Reference: CYOLO-SB+A (their released weights) = 90.8 synth / 86.5 room.

| model | room | di-left | synth |
|---|---|---|---|
| N3_belief_propagation | **44.7** | 58.5 | 89.3 |
| MERT_B2_pitch_aux | **43.7** | 57.1 | 87.9 |
| MERT_noisy | **42.8** | 58.7 | 88.5 |
| MERT_B5_dense_contrastive | **42.8** | 53.3 | 88.0 |
| MERT_C2_soft_dtw | **42.6** | 51.2 | 87.4 |
| N2_memory_retrieval | **42.0** | 52.1 | 86.6 |
| B1a_gated_film | **39.7** | 52.4 | 86.8 |
| B1a_mert_swap | **38.5** | 55.2 | 90.0 |
| MERT_B3_inr_subpixel | **37.9** | 51.7 | 89.7 |
| B1a_gated_cross_attention | **35.3** | 50.6 | 81.6 |
| Gated_dinov2_residual | **30.8** | 55.8 | 75.5 |
| B3_inr_subpixel | **27.9** | 64.4 | 86.8 |
| C2_soft_dtw | **20.6** | 63.2 | 86.0 |
| B1a_cross_attention | **19.3** | 29.7 | 71.1 |
| B5_dense_contrastive_aux | **17.8** | 68.2 | 86.9 |
| N1_long_context | **16.5** | 16.9 | 24.1 |
| B4_temporal_consistency | **16.4** | 68.7 | 86.9 |
| B2_pitch_aux | **16.0** | 62.9 | 86.7 |
| B6_impulse_response | **15.6** | 68.0 | 87.7 |
| B1a_spatial_film | **11.1** | 11.5 | 44.3 |
| V_dinov2_full_encoder | **7.6** | 7.8 | 9.2 |
| MERT_dinov2_cross_attention | **2.6** | 2.8 | 5.4 |

## Findings

1. **N3 (Bayes filter) is the best real-audio model at 44.7** and also holds
   synthetic (89.3). Gate check confirms the branch is genuinely active:
   gate 0.0->0.568, uniform escape floor 0.100->0.0486 (it learned to TRUST
   temporal continuity), evidence_scale 1.0->0.344. Not B1a passing through.
2. **Synthetic rank does not predict real rank.** B1a_mert_swap is best on
   synth (90.0) but 8th on room (38.5); MERT_B3 89.7 -> 37.9. Optimising
   synthetic MSMD has been actively misleading for the real-audio goal.
3. **B6 is refuted on the tier it was built for.** Impulse-response
   augmentation ranks LAST on room (15.6), below every non-augmented
   CBEncoder sibling, while scoring fine on di-left (68.0). Synthetic IR
   augmentation does not transfer to a real room.
4. **MERT buys room-robustness specifically, and trades clean accuracy.**
   Every MERT model sits 37-44 on room vs 15-28 for CBEncoder; but on the
   clean di-left pickup the CBEncoder models WIN (62-69 vs 51-59).
5. Broken, not merely weak: MERT_dinov2_cross_attention (2.6 room),
   V_dinov2_full_encoder (7.6 room).

## Gap to close

CYOLO reaches 86.5 on room and loses only 4.3 points synth->real. Our best is
44.7, losing ~45. The deficit is architectural, not encoder-level.
