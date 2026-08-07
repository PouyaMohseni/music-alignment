# Is CODA's SOTA claim comparable to CYOLO's? — a code-level audit

CODA (arXiv:2607.21899, ISMIR 2026, github.com/ValleyC/CODA @ `dba9829`, MIT)
reports .743 vs CYOLO-SB's .630 at ≤0.10 s on real audio, and .893 vs .885 at
≤0.50 s on synthetic. This file records what the released code says about
whether those two columns measure the same thing.

Audited against a clone at `/scratch/pmohseni/coda_audit/CODA`. Every claim
below carries a `file:line` that was read directly. Claims marked **[inferred]**
were not verifiable from code.

---

## Headline: CODA is given the page layout; CYOLO has to find it

`coda/models/coda_model.py:314-317` — the boxes are **required positional
arguments** of `forward()`, not an optional evaluation convenience:

```python
    def forward(self, score, perf, system_boxes, bar_boxes, bars_per_system,
                gt_system_idx=None, gt_bar_in_sys=None,
                prev_gt_system_idx=None, prev_gt_bar_page_idx=None,
                tempo_aug=False, p_pred=0.0):
```

Chain of custody on the **training** path — this is the part that was not
previously checked, and it is the part that matters:

| step | location |
|---|---|
| read from MSMD npz | `coda/utils/data_utils.py:32` — `systems`, `bars` |
| into piece metadata | `coda/utils/data_utils.py:239-243` |
| into the sample | `coda/dataset.py:703-705` |
| collated | `coda/dataset.py:622-624` |
| **into the train forward** | **`scripts/train.py:173-177`** |
| consumed as ROIs | `coda_model.py:402-407` (system), `:482-489` (bar) |

Boxes are never perturbed: `coda/dataset.py:711-728` applies only the same x/y
roll already applied to the score image, so they stay pixel-registered.

The loss confirms there is nothing to learn about layout — `coda/utils/loss.py`
`selection_loss` has exactly three terms: system CE over the **given**
candidates, bar CE over the **given** candidates, and note MSE in bar-local
coordinates. No box regression, no objectness, no IoU. The module says so
itself (`coda_model.py:10`):

> No anchors, no NMS, no objectness -- pure classification over known candidates.

and so does `configs/coda.yaml:2`:

> System -> Bar -> Note cascade via classification over known MSMD annotations.

**No detection path exists in the release at all.** `coda/models/backbone.py:60-62`
explicitly skips any layer named `Detect` when building from YAML, and
`configs/coda.yaml` contains none to skip. The inherited YOLO builder
`coda/models/builder.py:29-33` (with its `no = na * 5  # bbox + objectness`) is
**dead code** — `backbone.py:43` uses its own `_build_layers` instead. The
`predict_sb=True` targets built at `dataset.py:576-578` never reach a loss;
`data.targets` survives only so `selection_getitem` can recover the
augmentation shift (`dataset.py:716`) and for video overlay.

All three consumers of boxes are oracle: training (`train.py:175-177`),
streaming validation (`coda/utils/streaming_eval.py:130-137`), and reported
evaluation (`scripts/evaluate.py:514-516, 522-525`).

**So CODA's task is "pick 1 of N given regions." CYOLO's is "produce the
region."** The reported system accuracy .991 vs .963 and bar .975 vs .890 are
measured across that asymmetry. This is not a tuning-level defect — the two
systems solve different problems. The paper does not state it in §4.1.2 or
§4.1.3; it is visible only in the code and in the config comment.

**[inferred]** The *magnitude* of the advantage cannot be bounded from code.
CODA cannot be run without layout boxes, and none of Table 3's five ablations
removes them.

---

## Aggregation: CODA reports macro, CYOLO reports micro

CODA, per piece — `scripts/evaluate.py:719`:
```python
onset_ratios[t] = sum(1 for e in onset_errors if e <= t) / len(onset_errors)
```
CODA, across pieces — `scripts/evaluate_batch.py:179-183`, unweighted mean of
the 94 per-piece ratios, and the Table 1 LaTeX row consumes exactly these
(`evaluate_batch.py:357`):
```python
    for t in ONSET_THRESHOLDS:
        key = f'onset_ratio_{t:.2f}s'
        ratios = [m[key] for m in all_metrics if m.get(key) is not None]
        if ratios:
            summary[f'mean_{key}'] = float(np.mean(ratios))
```

CYOLO — `eval.py:76-83`, pool first, then divide:
```python
    frame_diffs = np.concatenate([piece_stats for piece_stats in stats['piece_stats'].values()]) / FPS
    total_frames = len(frame_diffs)
    ...
        ratio = np.sum(frame_diffs <= th) / total_frames
```
The per-piece print at `eval.py:59-73` is display-only and never averaged.

The two LaTeX rows are therefore produced by **different estimators**. Our own
measurement of one checkpoint across both — 45.4 micro vs 46.8 macro on 25
pieces (`AGGREGATION_FINDING.md`) — puts the pure estimator effect near +1.4,
which does not explain CODA's 7.7-point Table 1 gap but is uncorrected and
directionally favorable.

**It matters most in Setting II**, which is macro over just 16 pieces of very
unequal length against a micro baseline. That is exactly the regime where our
own macro-16 vs micro spread reached **+7.3** points. Treat `.743 vs .630` as
the least trustworthy number in the paper.

Internal inconsistency worth noting: CODA computes both macro and micro for the
jump table (`evaluate_batch.py:195, 198, 214, 218`) and Table 2 uses **micro**
(`evaluate_batch.py:307-310`), while Table 1 uses macro. Two tables in one
paper, two estimators, neither disclosed — §4.1.2 says only "the cumulative
ratio of onsets tracked," which reads as micro.

---

## Onset protocol: clean. Settled by our own reproduction.

Both sides use `[0.05, 0.10, 0.50, 1.00, 5.00]` **seconds**
(`evaluate.py:691`, `eval.py:57`).

There was an open question about the denominator: CYOLO's `--only_onsets`
(`eval.py:18`) defaults to `False`, and its output is labelled "Average frame
tracking ratios" — so if the published table were produced by the default
invocation, it would be all-frame while CODA's is onset-only.

**Resolved.** Our reproduction passes `--only_onsets`
(`eval_cyolo_sota_cpu.sh:43,49`) and yields `cyolo_sb` = **63.0 @0.1 s /
79.9 @0.5 s**, matching CODA's copied baseline row `.630 / .799` exactly. The
published CYOLO numbers are onset-only. The denominators agree and this is
**not** a defect.

Useful side effect: it means our 79.9 bar is self-reproduced, not trusted.

---

## Reproducibility of the release

- **No pretrained weights.** `README.md:231-233`: *"Model weights are not
  included in this source repository."* No `.pt`/`.pth`/`.ckpt`, no Git LFS.
  Even synthetic Table 1 needs a 50-epoch retrain to re-derive.
- **No real-audio path.** Setting II pairs synthetic images with real piano
  recordings on a 16-piece subset, but the release ships no such data, no split
  file, and no code path. `scripts/finalize_run.py:12-14` hard-codes
  `{"standard": 94, "repeat": 66, "random": 28}` and refuses to emit
  `status: complete` otherwise. The 16 pieces are never identified.
  **Setting II is not reproducible from this release.**
- **Baselines copied, not re-run.** §4.1.3: *"All baseline numbers are taken
  directly from [10] under the same evaluation protocol."* Table 1 caption:
  *"† Proprietary training data; baseline numbers from [10]."* [10] = Henkel &
  Widmer. No CYOLO submodule, checkpoint, or invocation in the repo; the only
  CYOLO figure in code is a hard-coded latency comment
  (`evaluate_batch.py:464`).

---

## Two smaller one-sided effects

- **Truncated-tail onsets dropped from the denominator** (favorable, likely <1
  point). `evaluate.py:421-423` breaks out of the frame loop when
  `to_ > signal_np.shape[-1]`; `evaluate.py:699-700` then `continue`s past those
  onsets rather than counting them as errors.
- **Time-conversion mechanics differ** (mixed sign, not quantifiable). CODA
  builds an x→time map **per system** and indexes it by the *predicted* system
  (`evaluate.py:342-350, 707-710`), assigning `time_err = 100.0` on a
  wrong-system prediction (`evaluate.py:713`) — a hard miss at every threshold,
  **harsher** than CYOLO's single unrolled page map (`dataset.py:194`), which
  can still land inside 5 s after a system error. Conversely `np.interp` clamps
  to the system's x-range, bounding within-system error by that system's
  duration rather than the page's — mildly **lenient**.

---

## Ranked summary

| # | issue | direction | size |
|---|---|---|---|
| 1 | Oracle system/bar boxes at **train and test** | favors CODA | large, **[inferred]** — unbounded from code |
| 2 | Macro (CODA) vs micro (CYOLO) aggregation | favors CODA | ~1.4 pts on 94 pieces; **up to ~7 on Setting II's 16** |
| 3 | Setting II unreproducible; no weights released | unverifiable | n/a |
| 4 | Truncated-tail onsets dropped | favors CODA | <1 pt |
| 5 | Per-system interpolation + 100 s wrong-system penalty | mixed | small |
| — | Thresholds, units, onset-only denominator | **no defect** | — |

## What this means for us

The bar we must clear is **our own reproduced `cyolo_sb` = 79.9 @0.5 s**, not
CODA's 88.3. CODA's margin over CYOLO is measured under an oracle-layout
asymmetry that the paper does not disclose and a macro/micro estimator swap
that it also does not disclose — and its real-audio claim, the one closest to
our objective, sits in the regime where the estimator swap is worth the most.

Nothing here says CODA is a weak system. It says the published margin is not a
like-for-like measurement, which is a result for the evaluation paper rather
than an excuse for our numbers.
