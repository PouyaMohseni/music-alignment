# The SOTA claim compares macro-aggregated to micro-aggregated numbers

**Verified independently on our own data, 2026-08-06.**

## The two aggregators

| | code | what it computes |
|---|---|---|
| **CPJKU / Henkel & Widmer / us** | `eval_model.py:102` `np.sum(onset_diffs <= th) / total_onsets` | **MICRO** — every onset weighted equally |
| **CODA (ISMIR 2026)** | `scripts/evaluate_batch.py:183` `float(np.mean(ratios))` | **MACRO** — every *piece* weighted equally |

## What that is worth on this benchmark

Recomputed from our R3 room run's own per-page output
(`results/eval_any-111639.log`, 25 pages, 4415 onsets):

| aggregation | R3 room pct@0.5s |
|---|---|
| MICRO (pooled onsets) — what we and every baseline report | **45.4** *(log: 45.6; reconstruction sound)* |
| MACRO over 25 pages | 46.8 |
| **MACRO over 16 pieces — CODA's aggregator** | **52.7** |

**The aggregation choice alone is worth +7.3 points**, on an unchanged model.

## Why it is so large

One recording dominates, and it is the one every model fails:

- Chopin Nocturne Op.9: **1315 / 4415 onsets = 29.8% of the micro metric**
- but only **6/25 pages (24.0%)** macro-by-page, and **1/16 (6.2%)** macro-by-piece
- it scores **14.6** where the rest of the set scores **57.0**

So macro aggregation quietly down-weights the hardest 30% of the benchmark by ~5x.

## Why this matters for the published claim

CODA reports **88.3** on MSMD-Rec real audio and compares it against CYOLO 71.2,
cyolo_sb 79.9 and cyolo_sb_a 86.5 — numbers it **copied from Henkel & Widmer
without re-running** (its own §4 says so). Those copied numbers are micro.

If CODA's 88.3 is macro and it degrades on the Chopin Nocturne the way every
model in our table does, its micro equivalent is plausibly **~80-84** — i.e.
around cyolo_sb_a, not 1.8 points clear of it.

**The aggregation difference (+7.3 measured here) is comparable to CODA's entire
claimed margin** over cyolo_sb (+8.4) and larger than its margin over
cyolo_sb_a (+1.8).

## What is verified vs inferred

**VERIFIED (high confidence):**
- Both aggregator implementations, read from source.
- The +7.3 gap, recomputed from our own per-page log.
- CODA's repo contains **no real-audio evaluation code at all** — a
  case-insensitive grep for `real_perf` / `msmd_rec` / `msmd_rp` / `setting ii`
  across every `.py`/`.yaml`/`.sh` returns nothing, and `run_full_pipeline.sh`
  hard-asserts exactly 94/66/28 synthetic pieces. The row that made it SOTA is
  the one row the release cannot reproduce.
- No pretrained weights: empty `git tag`, empty `git lfs ls-files`, GitHub
  releases API `[]`, HF models API for the author `[]`.
- "16-piece subset" and "25 pages" ARE the same set: collapsing `_page_N` on our
  25 gives exactly the 16 names in `room_split.yaml`, page counts sum to 25, and
  total duration of the 16 room wavs is 0.3115 h = Henkel's published "0.31 h".
  **So 88.3 is on our exact audio.**

**INFERRED (not proven):**
- That CODA's *Setting II* number specifically is macro. `evaluate_batch.py` is
  the only aggregator in the repo and it is macro — but since no Setting II code
  was released, this cannot be confirmed from the artifact.

## Consequences

1. **Our own standings table is internally consistent** — all micro — so
   R3 45.6 vs cyolo_sb 79.9 remains apples-to-apples. The gap is real.
2. **Report both aggregations from now on.** R3 is 45.6 micro / 52.7 macro-16.
3. **This is the evaluation paper's opening result**, and it is far stronger
   than the one we had (cyolo_sb vs cyolo_sb_a being inside noise). It is also
   not an attack on one group: three consecutive papers on this benchmark report
   bare point estimates with no CIs, no multi-seed runs, and now a silent
   aggregator change.
