# External real-audio benchmarks: what actually exists

Motivation. Our real-audio result rests on **25 pages**, and the piece-level
cluster bootstrap gives a design effect of ~180x — effective N is 25, not 4415
onsets. That powers roughly a **10-point** claim (99 pieces → 5 points, 619 →
2). Our set also pairs *synthetic score images* with recordings of MIDI
playback, so it is partly oracle-flavoured. An external benchmark with real
scans, real audio and human annotations would fix both.

Claims are **[verified]** where I read the code/data myself, **[agent]** where
they come from the survey and I did not re-check.

---

## Correction to an earlier claim in `OMR_FEASIBILITY.md`

That file listed **SMR v1.0** as "200 solo piano IMSLP scans, one **real
YouTube recording** each." **That is wrong. SMR's released audio is synthetic.**

**[verified]** The public release `HMC-MIR/YoutubeScoreFollowing` contains only
`data/midi/` and `data/pdf/` (GitHub contents API) — no audio, no `timeAnnot/`,
no `lineAnnot/`. Its own prep notebook renders the audio, `01_prepData.ipynb`
cell 3:

```python
audio_cmd = "fluidsynth -F p" + str(i) + ".wav default.sf2 " \
            "/home/mshan/ttemp/spring2020/data-v2.1/midi/p" + str(i) + ".mid -R=1"
```

**[agent]** The real-audio asset — 200 YouTube recordings with manual per-line
timestamps — is a separate Shan & Tsai TISMIR 2021 addition and **is not
published**; the notebooks reference `/home/mshan/ttemp/data/timeAnnot/` and
`lineAnnot/`, which exist nowhere public.

So SMR is a *synthetic* benchmark, strictly worse for our purposes than the
synthetic set we already own and understand. **Drop it.**

---

## Correction: MAcc is not comparable to pct@0.5s

`MAcc 0.82` was quoted as though it sat on our axis. It does not.

**[verified]** `mus_align/eval.py`:
```python
acc = np.sum(np.abs(ref_eval_measures - pred_eval_measures) <= error_boundary) / len(eval_times)
```
`error_boundary` defaults to **0.5**, and both sides are **fractional measure
indices**, not seconds. The grid is uniform in *time*:
```python
eval_times = np.linspace(0, pred.alignment.times[-1], int(pred.alignment.times[-1]/frame_rate))
```
at `frame_rate = 0.01` (100 Hz), and both interpolators use
`fill_value='extrapolate'`.

Two consequences:
- **[agent, measured from the 13 annotation files]** median measure = 2.20 s, so
  MAcc ≈ a **±1.10 s** criterion (per-piece 0.63–1.77 s) — roughly **2× looser**
  than our pct@0.5s.
- MAcc is **time-weighted**; our pct@0.5s is **onset-weighted**. Silences, held
  notes and slow passages get duration weight instead of note count.

**[agent]** MAcc is also **macro** (unweighted mean over 13 pieces), whereas
Shan & Tsai's line accuracy is **micro** (`total_acc / total_times`). Do not mix
them — the same distinction was worth up to 7 points in `AGGREGATION_FINDING.md`.

---

## MeSA-13 is the only real candidate that exists

**[agent]** 13 sheet-music scans + real performance audio + expert measure
annotations. Public repo `https://github.com/mfeffer/mesa-13` (124 MB incl. 80 MB
of demo videos; the 13 piece folders are ~62 MB) — easier than the CMU Box
mirror. Three files per piece: score PDF, MP3/OGG of a real performance, and
`alignment.json`:

```json
{"audio_score_alignment": [
  {"audio_start": 9.43, "audio_end": 9.9, "bbox_number": 1,
   "measure_bbox": [285.08, 355.74, 600.36, 730.80], "page_number": 2}, ...]}
```

One entry per measure **in logical (performed) order** — repeated measures
appear twice with the same bbox. Bboxes are absolute pixels at DPI 200.

| | |
|---|---|
| pieces / annotated pages | **13 / 40** |
| measure instances / unique boxes | 957 / 829 |
| total real audio | **35.9 min** |
| median measure duration | 2.20 s |
| systems (staff lines) | 209 |
| pieces with repeats / non-piano | 2 / 2 |

**License is the wart.** No LICENSE file on the repo (GitHub API `license: null`).
Code is MIT, papers are CC BY 4.0, the **data carries no stated terms**, and
audio provenance is mixed (at least one IMSLP-hosted recording with its own
per-file licence). Fine to download, evaluate on, and cite; **not** clean to
redistribute a derived dataset.

**"No oracle anywhere" needs qualifying.** The README says annotations were
"generated in a **semi-automated** fashion": madmom beat tracking + Waloschek
measure detection produce heuristic boxes and timestamps, which musicians then
audit and correct — 20 annotator-hours total. Good enough at 0.5–1 s
tolerances, but the timestamps are corrected madmom output, not hand-placed.

---

## Can our metric survive? No, not defensibly

The GT is a set of measure boundary times; JLTR builds the continuous map by
**linear interpolation between them**. Within-measure position is *assumed*, not
annotated. At a 2.20 s median measure, our 0.5 s tolerance is **23% of a
measure** — well inside the rubato that linear interpolation cannot represent.
Their ±0.5-measure choice is about the tightest defensible threshold given
measure-level annotation.

Options, most to least defensible:
1. **Adopt MAcc as-is.** Directly comparable to published numbers; `evaluate()`
   is ~40 lines to reimplement.
2. **Time-domain pct@X s** with **X ≥ ~1.0 s**, labelled "% of *time*", not
   "% of onsets."
3. **Onset-weighted variant** using our AMT bridge's detected onsets as sampling
   points only (position GT still human). Non-standard; needs justification.

**What is lost, plainly:** note → measure is a ~2× loosening of tolerance plus a
change of denominator from onsets to time. A model that wins on MAcc has *not*
been shown to win at note-level localisation. The MeSA-13 authors say as much:
"note-level alignment would be the most useful, but we posit that producing such
alignments would be expensive."

---

## Model-side gap is small: an adapter, not an architecture change

**[agent]** We need neither OMR nor page-layout handling — MeSA-13's ground truth
*is* measure boxes with page numbers. Clustering them into systems is the same
top-coordinate rule JLTR already uses, and it ran on the real annotations
cleanly: **209 systems across 40 pages, no failures.**

| | MeSA-13 | our MSMD strips |
|---|---|---|
| system height (DPI 200) | 263–497 px, median 365 | 120 px (fixed) |
| strip aspect ratio W/H | 25–123, median 70 | ~153 |
| measures per system | median 4 (range 1–8) | — |

At our 120 px strip height MeSA-13 strips are ~8,300 px wide — **shorter** than
our ~18,400 px MSMD strips. The architecture is not stressed.

Genuinely new: (a) input is a ~3× downscaled **scan**, not a LilyPond engraving
— expect a real domain hit; (b) 2 pieces have repeats, so logical ≠ graphical
order and any monotone left-to-right assumption fails them outright (this is
exactly the M1 repeat-ambiguity problem); (c) 2 pieces are non-piano.

---

## Baselines — and the one to actually quote

**[agent]** MeSA-13, macro MAcc over 13 pieces:

| system | setting | MAcc | MErr (measures) |
|---|---|---|---|
| Shan & Tsai hierarchical DTW | automatic | 0.33 | 10.9 |
| **JLTR (Bukey et al.)** | **automatic** | **0.72** | 1.9 |
| JLTR + human repeat labels | R | 0.82 | 0.4 |
| JLTR + repeats + **GT measure boxes** | R,M | 0.86 | 0.4 |
| JLTR + repeats + boxes + clef/key | R,M,S | 0.88 | 0.3 |

**The fully automatic number is 0.72, not 0.82.** The abstract's "33% → 82%" is
the human-in-the-loop setting. And since we would consume the GT measure boxes
to build the strip, the honest comparison row is **R,M = 0.86** — or report
against 0.72 and disclose the boxes.

**No CYOLO-family or CODA numbers exist on MeSA-13.** JLTR positions
Dorfer/Henkel (our lineage) as a *different task* — real-time, digital score.
Everything reported there is offline DTW alignment. That is an opportunity
(first score-following numbers on a real-scan benchmark) and a hazard (reviewers
will ask why an online tracker is compared to offline DTW).

---

## Power: MeSA-13 is a validation set, not a benchmark

Using our own calibration (25 clusters → 10 points, MDE ≈ 10·√(25/N)):

| set | clusters | MDE (pts) |
|---|---|---|
| **MeSA-13** | **13** | **≈13.9** |
| our current real-audio set | 25 | 10.0 |
| ours + MeSA-13 pooled | 38 | 8.1 |
| SMR (*synthetic*) | 100 | 5.0 |
| Shan & Tsai real-audio set (**unreleased**) | 200 | 3.5 |

13 pieces powers ~14 points — **worse than our current 25**. And that is
optimistic: MeSA-13 is deliberately heterogeneous, and JLTR's own dispersion
shows it (automatic MErr 1.9 measures, std **3.7**, i.e. std ≈ 2× mean).

Do not count the 40 pages instead (which would give 7.9). Pages within a piece
share performer, recording, room and scan — piece-level clustering is right, for
the same reason we argued it for our own set.

---

## Everything else, ruled out

- **CollabScore — no audio at all.** 26 Saint-Saëns scores, Gallica images +
  MEI/MusicXML, CC BY-NC-SA. An OMR ground-truth set. The "links to audio
  fragments" description does not match what is released.
- **YTSV / U-MusT (arXiv:2505.12863)** — 12,317 videos, ~1,463 h real YouTube
  audio (**762 h piano solo**), score images from video slides, MIT code. But
  alignment is **slide/page-turn granularity, automatically derived** — far too
  coarse for our metric. **Interesting as a real-audio *pretraining* corpus for
  the domain gap, not as an evaluation set.** Did not appear in the earlier survey.
- **MUSCAT (ACM MM 2024)** — 80 h real audio, 1,251 scanned sheets, but
  annotations are score-level symbolic (for transcription), not time→pixel.
  Access by request.
- **ASAP / Magaloff / Zeilinger / Batik / Vienna4x22 / nASAP / PERiScoPe** —
  shelving these was correct. Matchmaker (arXiv:2510.10087) benchmarks on
  (n)ASAP, Batik, Vienna4x22 — all **symbolic/MIDI scores, no sheet images**.
  They cannot score a pixel-position model.
- **Sheet Music Benchmark (ISMIR 2025)** — OMR only, no audio.

**MeSA-13 appears to be the only dataset pairing real scans + real audio + fine
time alignment that exists.** That is itself worth stating in the paper, and it
is why it is small.

---

## Recommendation

**Pursue MeSA-13, but change what we want from it.** It cannot power an
improvement claim. What it *can* do, which nothing we own can, is support an
**oracle-free existence claim**: "given only a real scan and a real recording,
our score follower reaches MAcc X on the same 13 pieces where published
automatic alignment reaches 0.33 and 0.72." One honestly-caveated table row that
closes the "your real-audio result is partly oracle-flavoured" objection without
claiming power it does not have. Report MAcc, macro, their exact `evaluate()`.
**Do not report pct@0.5s on it.**

First concrete step (~1–2 days, 62 MB, no new dependencies, no GPU):
1. `git clone https://github.com/mfeffer/mesa-13` into `/scratch/pmohseni/` (skip `videos/`).
2. Write a MeSA-13 → strip adapter: render each PDF at DPI 200, read
   `alignment.json`, cluster GT boxes into systems by top coordinate, crop and
   concatenate into a 120 px strip, emit `strip_to_page_mapping` in our existing
   schema plus a per-measure `(measure_idx, strip_x_start, strip_x_end,
   audio_start, audio_end)` table. **Traverse in logical order** so the 2 repeat
   pieces unroll correctly — the only nontrivial part.
3. Reimplement `evaluate()` (~40 lines, no TF, no madmom); sanity-check by
   scoring the ground truth against itself — must give exactly 1.0.
4. Run our best real-audio checkpoint zero-shot. Expect a large drop; scans at
   3× downscale are out of domain. **Measuring that drop is the informative
   outcome** — it tells us whether the 25-page result generalises at all.

**Worth doing in parallel:** the only properly-powered external real-audio set
(200 clusters, 3.5-point MDE) is Shan & Tsai's YouTube annotations, which are
unreleased. Email TJ Tsai (HMC-MIR) and ask for `timeAnnot/` and `lineAnnot/`.
Short email, large payoff, and their code is MIT so they are release-friendly —
the annotations look simply to have been left on a lab machine.
