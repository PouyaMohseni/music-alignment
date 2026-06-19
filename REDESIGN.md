# Audio-to-Score Alignment — Honest Post-Mortem and Redesign

*Written 2026-06-18. Grounds every claim in either this repo's code or published
literature (CPJKU MSMD line, MERT, FiLM, Soft-DTW, transcription-pivot alignment).
Produced from a multi-agent diagnosis: 6 independent expert lenses → 3 adversarial
verifications of the leading hypotheses → synthesis + skeptical review. Where the
evidence is confounded or a hypothesis was refuted, this document says so.*

---

## 0. TL;DR

Our best model (`v3_all`) tracks **20.6 % of note onsets within 0.5 s** and has a
**mean error of 5.35 s**. The directly comparable published system — Henkel, Kelz &
Widmer, *Learning to Read and Follow Music in Complete Score Sheet Images*, ISMIR
2020, **on the exact same 353-train / 94-test MSMD split** — reports **85.2 % within
0.5 s** and a median error around half a centimetre on the page. We are roughly
**4× worse at the 0.5 s threshold** and we never follow a whole piece end-to-end.

This is not a tuning gap. It is **three structural choices, each independently
ceiling-ing performance**, and they are mutually reinforcing:

1. **We framed *following* as memoryless *retrieval*.** We build one global
   `(T_audio × N_tiles)` cosine-similarity matrix and DTW it at inference. Every
   audio frame must re-identify its absolute position against the whole score with
   no memory of where tracking currently is. SOTA conditions the score on the audio
   heard *so far* (FiLM + LSTM) and predicts one local position. Their own
   "no-temporal-context" ablation is their *worst* model — it is essentially our
   architecture.
2. **Both encoders are pitch-blind, and we threw away a noiseless pitch signal.**
   `midi_pitch` is stored in every `noteheads.npz` and read by **nothing** in
   `mymodel/`. We pay two frozen foundation models (ImageNet ViT, music-tagging
   MERT) to rediscover "this sound = this notehead" from scratch through a thin
   head, when we have the exact pitches for free on both sides.
3. **The loss has no monotonic/temporal coupling and never forces sharpness.**
   `expected_distance_loss` is fully separable across frames; monotonicity is bolted
   on only at inference (DTW). With `entropy_weight=0` and L1 distance, the model is
   free to produce diffuse, centered similarity rows that DTW cannot ride.

The single highest-value next action is **one experiment that costs no training**
(the oracle pianoroll baseline, §6 E1). It tells us whether the bottleneck is the
*framing* or the *features* before we build anything.

---

## 1. What we built

| Stage | Idea | Encoders | Loss | Window |
|---|---|---|---|---|
| v1_baseline | dual-encoder + cosine | MERT + ViT (frozen) | SoftDTW/anchor | 5 s |
| v1_nce/nce2 | InfoNCE pretrain | MERT + ViT + LoRA r4 | InfoNCE | 5 s |
| v2_crossattn | cross-attention fusion | + LoRA | InfoNCE | 5 s |
| **v3_fullseq/all** | **whole-performance + distance loss** | frozen, precomputed (LoRA-adapted) | expected-distance | **full piece** |
| v3_e2e | end-to-end fine-tune | LoRA trainable | expected-distance | full piece |

Pipeline (`mymodel/v1_baseline/encoders.py`, `msmd_prep/strip.py`): the score is
**unrolled** — each staff system is cropped and concatenated horizontally into one
ribbon **224 px tall**, then tiled into **224×224 columns at stride 56** (~1 tile ≈
1 beat); each tile → ViT **`[CLS]` token** → one 768-d vector. Audio →
MERT-v1-95M **last hidden layer** → mean-pooled 75 Hz → ~10.7 Hz. Both projected to
256-d, L2-normed, cosine → `(T×N)` matrix → `expected_distance_loss` (train) /
banded DTW backtrack (infer).

## 2. What we measured (test split, 94 pieces)

| Model | mean err | within 0.5 s | within 1.0 s | recall@1 |
|---|---|---|---|---|
| **v3_all** | **5.35 s** | **20.6 %** | **37.6 %** | 2.5 % |
| v3_fullseq | 6.71 s | 15.4 % | 28.0 % | — |
| v1_nce | 9.25 s | 8.9 % | 17.2 % | 0.5 % |
| v1_nce2 | 9.60 s | 8.4 % | 16.3 % | — |
| v1_baseline | 10.0 s | 8.3 % | 16.3 % | — |
| v1_dtw | 10.4 s | 8.5 % | 16.5 % | — |
| v2_nce | 11.2 s | 5.0 % | 9.7 % | 2.7 % |
| v3_e2e | 11.3 s | 4.0 % | 8.3 % | 2.5 % |

**SOTA on the same data (Henkel et al., ISMIR 2020, Tables 2–3, synthetic MSMD):**
≤0.05 s: 73.3 %, ≤0.10 s: 74.7 %, **≤0.50 s: 85.2 %**, ≤1.0 s: 88.5 %, ≤5.0 s: 93.7 %.

### Things we proved along the way (negative + positive results worth keeping)
- **Full-sequence + distance-aware loss >> windowed InfoNCE** (5.35 s vs 9.25 s). The
  original instinct to drop 5 s windows and penalize by mis-localization *distance*
  was correct and is the reason v3 is the best family.
- **All-performance augmentation helps** (6.71 → 5.35 s).
- **LoRA-adapted embeddings >> raw frozen** (6.7 s vs 14 s). Feature discriminability
  dominates within a frame.
- **End-to-end fine-tuning HURT** (worse than frozen). But note: we fine-tuned the
  *wrong architecture* (ImageNet ViT on ~945 score pages overfits/drifts). This is
  **not** evidence that learned-from-scratch encoders fail — the literature shows the
  opposite.
- **InfoNCE plateaus** (~9 s): it optimizes retrieval discriminability, not
  positional accuracy; a near-miss is punished as hard as a cross-page miss.
- **Small-γ SoftDTW is unstable** (negative loss / NaN) — this is documented behavior
  of Soft-DTW as γ→0, not a bug in the idea.

## 3. Root-cause analysis (ranked, with confidence and honest caveats)

### RC1 — Memoryless global-retrieval framing  ·  confidence 0.85  ·  **co-primary**
We compute a global `(T×N)` matrix and match each frame independently. On repetitive
piano music many positions look alike, so each frame faces the full
repeat-ambiguity. Henkel 2020's **"no-temporal-context" ablation is their worst model
(3.70 cm vs 1.25 cm)** "because audio excerpts could match several positions" — that
ablation *is* our architecture. Their fix: an **LSTM over audio history** that
**FiLM-conditions** the score encoder and predicts **one** local position. This is
the largest gap and is independent of feature quality.

### RC2 — Pitch-blind towers; the MIDI signal is discarded  ·  confidence 0.80  ·  **co-primary**
`grep -r midi_pitch mymodel/` → nothing. Ground truth is collapsed to a scalar
x-position in `precompute.py:_build_targets`. Everyone who does well on this task
aligns in a **pitch space**: Dorfer/Henkel feed a **2-D** score snippet + **2-D
log-frequency spectrogram** (both keep a pitch axis); classical synchronization
(Ewert/Müller DLNCO chroma; Kwon et al. transcription-pivot) aligns in pitch/chroma.
We align two opaque embeddings instead. *Caveat:* the LoRA-vs-raw evidence (14→6.7 s)
proves "features matter" but is **confounded across both towers** — it cannot prove
the score side alone caps us.

### RC3 — Loss has no monotonic coupling and doesn't force sharpness  ·  confidence 0.80
`expected_distance_loss` is `Σ_n p[t,n]·|pos_tile−pos_target|`, summed over
*independent* frames. L1 expected distance is minimized by any distribution whose
*median* sits on the target — a flat plateau costs ≈ the same as a spike. With
`entropy_weight=0`, `power=1`, the model produces diffuse rows; DTW needs a sharp
ridge. Monotonicity is imposed only at inference, never trained.

### Secondary
- **Onset blur on the audio side.** MERT `last_hidden_state` (not the canonical
  learned 13-layer weighted sum) + **7-frame mean-pool to ~93 ms**. SOTA targets
  50 ms (≤0.05 s bin). Mean-pooling specifically destroys onset transients. Both are
  cheap, code-confirmed fixes.
- **`[CLS]`-pooling a 224×224 tile** discards the spatial layout (notehead height =
  pitch); stride-56 overlap makes adjacent tiles share 75 % of pixels → near-collinear
  embeddings → recall@1 = 2.5 %.
- **Beat-rate quantization** (~1 tile/beat ≈ 0.5 s floor at 120 bpm) caps the 0.5 s bin.

### A hypothesis we tested and **REFUTED** (kept for honesty)
> *"The 1-D strip + repeated-measure aliasing makes the matrix fundamentally
> non-alignable; tracked_until_end=0 proves it derails mid-piece."*

**Refuted (confidence 0.8).** `tracked_until_end` is `within.all()` with an 85 px
(~0.4 s) threshold (`metrics.py`) — an **all-or-nothing statistic that is ≈0 by
construction** for any piece with hundreds of onsets at 5 s mean error. It is *not*
evidence the tracker "loses the thread." The banded DTW already suppresses distant
off-diagonal aliasing, and the literature follows successfully on an *unrolled*
(effectively 1-D) score. The residual error is **local** tile confusion (encoder /
loss / framing), not global non-alignability. **Do not redesign around this metric.**
The honest signal is `global_tracking_ratio ≈ 0.08` = fraction of onsets within
~1.5 tiles — again a *local precision* story.

## 4. What we got right / what we couldn't do

**Right:** the dataset pipeline (`msmd_prep`, spec-compliant, 467 pieces, audio
synth, all-performance aug); the full-sequence + distance-loss insight; rigorous
ablation hygiene (we have clean numbers for 8 variants).

**Couldn't do:** get below ~5 s mean error / ~20 % @0.5 s. We never matched even the
weakest CPJKU baseline. We never used the symbolic pitch ground truth. We never gave
the model temporal memory. We never trained a notation-appropriate encoder (ViT was
frozen ImageNet; e2e fine-tuning of it drifted).

## 5. The redesign

**Core move: align in a pitch-aware space, with explicit temporal state, supervised
by the MIDI we already have.** Two stages sharing one representation.

### Representation
- **Audio tower.** Keep MERT but (a) replace `last_hidden_state[:,0]` with a
  **learned softmax-weighted sum over all 13 hidden states** (~13 params, canonical
  MERT/MARBLE recipe); (b) cut pooling to ~25 Hz with **max-pool or a learned Conv1d**
  (preserve onsets); (c) a small causal **GRU head → 88-key pianoroll logits/frame**,
  supervised by a pianoroll built from `midi_pitch`+`onset_sec`+`midi_offset_sec`
  (exact — the audio is FluidSynth from that MIDI). Also run a **log-CQT/STFT + small
  CNN** audio tower as a non-MERT control (Henkel's 78-bin frontend).
- **Score tower.** **Drop ImageNet ViT and `[CLS]` pooling.** Either a small
  **from-scratch Conv-BN-ELU CNN** (24→48→96, à la CPJKU `audio_sheet_retrieval`)
  over the strip, *or* keep per-patch ViT tokens with column-attention that preserves
  the vertical (pitch) axis. **But for the first baseline, do not learn the score
  side at all** — build the score pianoroll **directly from ground-truth MIDI**
  (perfect, zero OMR risk). Defer the OMR head until pitch-space is proven to help.

### Matching mechanism
- **Stage A — symbolic pivot (near-free).** Cosine/IoU over the two pianorolls →
  `(T×N)` → banded DTW. The proven classical alignment space.
- **Stage B — conditioned following (the SOTA frame).** An **LSTM over audio
  history** produces a context vector that **FiLM-conditions** the score features;
  the head predicts a **single local position** (segmentation mask / regression),
  not a global matrix. This injects the temporal state that fixes RC1.

### Loss
1. **Transcription/OMR auxiliaries (BCE on the 88-key targets).** The load-bearing
   supervision that installs pitch awareness a thin head can't learn implicitly.
2. **Localization.** For Stage A, keep `expected_distance_loss` but **`power=2`,
   `entropy_weight≈0.02`, `temperature≈0.03`** to force sharp ridges; add a
   monotonic term — **Soft-DTW with moderate γ≈0.5 annealed down** (avoids the
   small-γ NaN we hit) **or CTC** (numerically stable, monotonic by construction).
   For Stage B, Henkel's segmentation/Dice loss on the position mask.

### Training
Keep encoders frozen initially (e2e drift already burned us); the from-scratch score
CNN is small enough to train fully. Train transcription/OMR heads to convergence
first (cheap, well-posed), then add localization. Keep tempo + all-performance
augmentation.

## 6. Ranked experiment plan (cheapest, highest-leverage first)

**E1 — Oracle pianoroll DTW. No training, hours of CPU. DO THIS FIRST.**
Build *both* pianorolls directly from MIDI ground truth → cosine → existing
`dtw_backtrack` → `henkel_metrics`. **This single experiment disambiguates the entire
root-cause stack:** if error collapses toward sub-second, pitch-space + the 1-D strip
*is* alignable and RC2 is the lever; if it stays ~5 s even with perfect pitch, the
**framing (RC1)** dominates and no feature fix alone will save us.

**E2 — Band-radius sweep (1 hour).** `band_radius_frac` 0.25 → 0.05 on the current
model. If error barely moves, distant repeats were never the issue (consistent with
the refuted aliasing claim) → invest in features/framing, not decoding.

**E3 — Loss sharpening (config-only, no new code).** Set `power=2`,
`temperature≈0.03`, `entropy_weight≈0.02` on the existing v3 model. If 5.35 s drops,
RC3 is confirmed and isolated — and you've saved the Soft-DTW/CTC engineering for
when it's actually justified.

**E4 — Score pianoroll from MIDI + learn only audio→pitch (days).** Score side =
oracle MIDI pianoroll; train a small transcription head on frozen MERT (BCE vs the
FluidSynth-source pianoroll); align. Collapses RC2 to a one-tower problem with **zero
OMR risk** — the honest transcription-pivot baseline.

**E5 — MERT layer-weighting + reduced pooling (1 day).** Weighted 13-layer sum +
25 Hz max-pool + transcription head. Targets the ≤0.05 / ≤0.1 s bins specifically.

**E6 — Full Stage-A symbolic pivot, then E7 — Stage-B conditioned following (weeks).**
Stage B (LSTM + FiLM + segmentation) is the path to ~85 % @0.5 s — but it is the most
expensive item and should be **gated on E1 showing pitch-space helps at all.**

## 7. Risks and what might still not work

- **OMR-from-tiles is the load-bearing weak link.** Per-tile pitch from a 224 px
  window that may not contain the clef/key signature is the most likely silent
  failure. *Mitigation:* the E1/E4 path uses score-pianoroll-from-MIDI and avoids OMR
  entirely; defer learned OMR until proven necessary.
- **Co-credit confound.** The symbolic pivot may only help *with* the temporal-state
  fix. E1 isolates which lever matters; don't claim the pivot if Stage-A-without-
  temporal-state still wanders.
- **MERT may under-respond to FluidSynth timbre** (pretrained on real music) — the
  CQT-CNN control (E5) checks this.
- **Metric protocol.** `henkel_metrics` uses an 85 px hard threshold; confirm its
  definition matches the published protocol before claiming we've "reached SOTA."
- **Real audio / Iranian classical** (the project's stated extension) is *not*
  addressed here — clean synthetic transcription/OMR targets won't exist there.
  Sub-second on synthetic MSMD is the milestone, not the finish line.

## 8. The one-line recommendation

**Run E1 today.** It is free, it is decisive, and it determines whether the next
month is spent on *features* (pitch pivot) or *framing* (conditioned following) — or,
as the evidence suggests, **both**, in that order.

---

## References
- Henkel, Kelz, Widmer. *Learning to Read and Follow Music in Complete Score Sheet Images.* ISMIR 2020. (same MSMD split; 85.2 % @0.5 s) · code: CPJKU/audio_conditioned_unet
- Henkel, Widmer. *Real-Time Music Following … Multi-Resolution Prediction.* Frontiers in CS 2021. (86.1 % @0.05 s) · *Multi-modal Conditional Bounding Box Regression*, arXiv:2105.04309
- Dorfer, Henkel, Widmer. *Score Following as a Multi-Modal RL Problem.* TISMIR 2019. · *Learning to Listen, Read, and Follow*, ISMIR 2018 (arXiv:1807.06391)
- Dorfer et al. *Learning Audio–Sheet Music Correspondences for Cross-Modal Retrieval.* TISMIR 2018 · CPJKU/audio_sheet_retrieval (from-scratch VGG-style CNN)
- Li et al. *MERT.* ICLR 2024 (arXiv:2306.00107) · Yuan et al. *MARBLE.* NeurIPS 2023 (arXiv:2306.10548) — learned 13-layer weighting; MERT encodes local pitch/beat
- Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018 (arXiv:1709.07871)
- Cuturi, Blondel. *Soft-DTW.* ICML 2017 — γ→0 negativity/instability documented
- Ewert, Müller, Grosche. *High-Resolution Audio Synchronization Using Chroma Onset Features (DLNCO).* ICASSP 2009 · Kwon, Jeong, Nam. *Audio-to-Score Alignment using RNN-based AMT.* ISMIR 2017 (arXiv:1711.04480)
