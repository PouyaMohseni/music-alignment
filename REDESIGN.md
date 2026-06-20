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

## 0.5 Update 2026-06-20 — RC1 confirmed and captured; the bottleneck has moved

We built the temporal-state fix the redesign called for (RC1) before running E1, and
it worked about as predicted. **`v5_recurrent`** adds an LSTM over the projected audio
sequence that emits a temporally-conditioned query per frame; DTW then rides the
conditioned `(T×N)` logit matrix instead of a memoryless cosine matrix. We swept
**13 architectural variants** (capacity, depth, uni/bi-directional, residual delta,
warm-start vs from-scratch, longer schedules, a pitch-aux head).

| | mean err | @0.5 s | vs fair baseline |
|---|---|---|---|
| **best err — `v5l_deep_bidir`** (2 x-attn + bidir + residual) | **2.21 s** | 31.8 % | — |
| **best @0.5 s — `v5m_big`** (shared 512 / LSTM 1024, from scratch) | 3.01 s | **33.9 %** | — |
| `v5i_bidir_residual` (prior best) | 2.38 s | 29.6 % | — |
| **`v3_all` single-perf (fair baseline)** | **6.13 s** | **18.1 %** | — |

(The 5.35 s / 20.6 % in §0/§2 was v3_all on the *all-performance* eval. The fair,
apples-to-apples comparison — both families trained on all 13 performances, both
evaluated on the single-performance test set — is **6.13 s / 18.1 %**, the number to
beat. v5 nearly **3×'d** the mean error and **~2×'d** @0.5 s against it.)

**What this proves.** RC1 was real and is the single largest lever we have pulled:
temporal memory alone closed roughly **half** the gap to Henkel's 85.2 %. The redesign's
top-ranked root cause is confirmed.

**What this also proves — the more important finding.** The result is **plateaued and
architecture-insensitive.** Across 13 variants spanning a wide capacity/depth/direction
range, nothing breaks past **~2.2 s / ~34 %**. Two further signals localize the new
ceiling:
- **Cross-attention is not the lever.** `v5c_noxattn` (no fusion at all) is competitive
  at 2.33 s / 29.2 %. The LSTM memory is doing the work; stacking fusion does little.
- **Raw capacity has diminishing returns.** `v5m_big` buys the best @0.5 s but not a
  better mean error, and not by much.

So the follower architecture is **no longer the bottleneck.** The remaining 34 %→85 %
gap is the *other* two root causes the redesign named — **RC2 (pitch-blind features)**
and the **resolution** secondaries (onset blur, beat-rate tile quantization, `[CLS]`
pooling that discards the pitch axis). The earlier pitch attempts (`v4c`, `v5k`) appeared
to fail — but on inspection that was a **wiring bug, not a real test**: the pitch head
read the *frozen precomputed* embedding, not the aligned feature, so its gradient never
reshaped anything that gets matched ([model.py:132-133](mymodel/v5_recurrent/model.py#L132-L133),
see §9.1). **RC2 is untested, not refuted** — and it has a cheap, never-run test (§9.4 E0).
The v6 redesign (§9) keeps the system **end-to-end on the foundation models with no
pianoroll/transcription at inference**, and instead uses MIDI *only at training time* to
**shape the embeddings** to be pitch-aware. **The free diagnostic E1 and the one-line E0
fix come first (§9).**

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
| **v5l_deep_bidir** (temporal, §0.5) | **2.21 s** | 31.8 % | — | — |
| **v5m_big** (temporal, §0.5) | 3.01 s | **33.9 %** | — | — |
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
- **Temporal memory is the biggest single lever (v5, §0.5).** An LSTM over audio history
  + DTW on the conditioned logits nearly 3×'d mean error (6.13 → 2.21 s). Confirms RC1.
- **The pitch auxiliary was mis-wired, so RC2 is still untested (v4c, v5k).** Gradient-only
  pitch BCE scored 6.00 s vs 6.13 s — flat — but the head read the *frozen* embedding, not
  the aligned feature, so the gradient never reached the matched representation (§9.1). Not
  evidence pitch fails; evidence the wiring was inert. v6 fixes the wiring (§9.4 E0).

## 3. Root-cause analysis (ranked, with confidence and honest caveats)

### RC1 — Memoryless global-retrieval framing  ·  **CONFIRMED & CAPTURED (§0.5)**  ·  was co-primary
> **Update 2026-06-20:** built and validated. An LSTM temporal head (v5) cut mean error
> 6.13 s → 2.21 s and lifted @0.5 s 18.1 % → 33.9 % — about half the gap to SOTA. The
> diagnosis below was correct. This lever is now spent; the residual ceiling is RC2 +
> resolution (see §0.5, §9).

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

> **Superseded 2026-06-20 by §9.** Items E4/E6/E7 below describe a symbolic
> *transcription-pivot* (aligning pianorolls). That conflicts with the current
> constraint — **end-to-end, foundation models, no roll at inference** — so the
> live plan is §9.4 (E0/E1/E2/E3/E4). E1 (oracle pianoroll DTW) survives **only as a
> free diagnostic that ships nothing**, not as an inference path. Kept below as the
> 2026-06-18 record.

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

> **2026-06-20:** we did *framing* first (v5) and it delivered as predicted but
> plateaued. E1 is still unrun and is now the gate for everything in §9. Run it first.

---

## 9. The v6 redesign — pitch-shaped foundation embeddings, end-to-end, no symbolic pivot

*Written 2026-06-20, after the v5 sweep; revised same day to the explicit design
constraint:* **end-to-end image↔audio alignment, built on the foundation models
(MERT + ViT, LoRA-adapted), with NO pianoroll / transcription / symbolic intermediate
at inference.** MIDI is used *only* as a training-time signal to shape the embeddings; the
deployed model sees (score image, audio) and nothing else. This rules out the
transcription-pivot of §9-draft (aligning predicted pianorolls makes a symbolic roll the
inference representation). What we keep from that literature is the *principle*, realized
*inside* the embedding instead of beside it.

### 9.1 The pitch lever was never actually tested — a wiring bug, not a dead end

The original §9 read the v4c/v5k results as "pitch-as-auxiliary fails." **That conclusion is
invalid.** Look at the v5 pitch head ([model.py:132-133](mymodel/v5_recurrent/model.py#L132-L133)):

```python
out["audio_pitch_logits"] = self.audio_pitch(audio_emb)   # RAW frozen precomputed input
out["score_pitch_logits"] = self.score_pitch(tile_emb)    # RAW frozen precomputed input
```

The pitch head reads `audio_emb` / `tile_emb` — the **raw, precomputed, frozen** embeddings —
**not** `a` / `i`, the projected, cross-attended features that are actually matched
([model.py:115-128](mymodel/v5_recurrent/model.py#L115-L128)). Those inputs are frozen tensors
with no gradient path to anything. So the pitch BCE trained the *pitch head's own weights and
stopped there* — it never touched `audio_proj`, the cross-attention, the LSTM, or `query_proj`.
**The pitch supervision was a detached side-branch that could not, by construction, make the
aligned representation pitch-aware.** v4c/v5k tell us nothing about whether pitch helps; they
tell us this *wiring* is inert. RC2 is **untested**, not refuted.

This is good news: the single most important remaining lever has a cheap, never-run test.

### 9.2 What to keep from Juhan Nam's line — the principle, not the pivot

Kwon, Jeong & Nam (*Audio-to-Score Alignment using RNN-based AMT*, ISMIR 2017, already cited)
transcribe audio to a pianoroll and DTW it against the symbolic score. We **reject the
mechanism** (it puts a symbolic roll on the inference path) but **keep three principles**, each
re-homed inside the foundation embeddings:

1. **Pitch must live in the representation that gets matched.** Their whole result rests on
   comparing *pitch content*, not opaque features. Our translation: make MERT/ViT embeddings
   *pitch-discriminative* via an auxiliary objective wired to the **aligned** feature — then
   align those embeddings directly, no roll. (Henkel 2020 does exactly this: learned encoders,
   never a transcription, pitch emerges implicitly from a 2-D log-frequency frontend. Henkel is
   the better architectural fit for our constraint than the Nam pivot.)
2. **Soft / probabilistic, not hard.** Whatever pitch supervision we add stays a soft auxiliary
   (BCE logits), never a thresholded note list — so an imperfect pitch signal degrades
   gracefully and is *dropped entirely at inference*.
3. **Onset-awareness for the sub-second bins.** @0.5 s is an **onset-timing** metric. An onset
   auxiliary channel (separate from sustain) on the audio tower targets exactly the bins our
   beat-rate tiling floors — again, training-time only.

### 9.3 The architecture (v6) — three embedding-native levers, pure (image, audio) inference

```
INFERENCE (no MIDI, no roll):
  audio  ─ MERT (LoRA) ─► audio_proj ─┐
                                       ├─► cross-attn ─► a,i ─► LSTM follower ─► position
  image  ─ ViT  (LoRA) ─► image_proj ─┘                         (v5, kept)

TRAINING ONLY (detached at inference): pitch/onset heads hang off a,i and the LoRA encoders,
  BCE vs MIDI pianoroll, to SHAPE the embeddings. Removed from the graph when deployed.
```

- **Lever A — pitch-shaped embeddings (RC2, finally wired right).** Move the pitch head onto
  the **aligned** feature `a` / `i` (and, in the full version, let its gradient reach the
  **LoRA-adapted** MERT/ViT). Now the pitch BCE reshapes the exact representation that DTW/the
  follower consumes. At inference the pitch head is deleted — embeddings are already
  pitch-aware. This is the clean RC2 test the wiring bug prevented.
- **Lever B — resolution (the secondaries, all foundation-model-native).** (a) Replace MERT
  `last_hidden_state` with a **learned softmax over 13 layers** (canonical MARBLE recipe; MERT
  provably encodes pitch in intermediate layers). (b) Cut pooling 93 ms → ~25-40 ms with
  max/learned-conv pooling to preserve onsets. (c) **Stop `[CLS]`-pooling the score tile** —
  keep patch tokens / column-attention so the **vertical axis (= pitch)** survives; add a
  **sub-tile x-regression** head so we beat the ~0.5 s beat-quantization floor.
- **Lever C — deeper temporal conditioning (RC1++, the Henkel form).** v5's LSTM reads audio
  history; extend it to **FiLM-condition the score features** with that history vector and
  predict a **single local position** rather than re-scoring a global matrix. End-to-end,
  learned, no roll — and it makes the score representation *adapt to what's been heard*, which
  both disambiguates repeats and sharpens pitch relevance.

All three compose into one network with one inference path: foundation encoders → conditioned
follower → position. Pitch is scaffolding that is removed before deployment.

### 9.4 Experiment plan (cheapest, highest-leverage first)

- **E0 — re-wire the pitch head onto `a`/`i` and retrain (HOURS, ~one-line change).** Keep the
  frozen precomputed embeddings; change [model.py:132-133](mymodel/v5_recurrent/model.py#L132-L133)
  from `self.audio_pitch(audio_emb)` → `self.audio_pitch(a)` (score side `i`), add the BCE
  (already in [train.py]). This forces `audio_proj`/cross-attn to *surface* the pitch MERT
  already encodes. **Cheapest possible test of RC2** — if @0.5 s moves, pitch-shaping is real
  and worth the LoRA cost; if not, frozen MERT lacks recoverable pitch and we need Lever B/LoRA.
- **E1 — oracle pitch-space DTW (FREE diagnostic, ~20 lines, throwaway — NOT an inference
  path).** Build both pianorolls from MIDI (`audio_pitchroll`/`score_pitchroll`, exist), cosine
  → `dtw_backtrack` → `henkel_metrics`. This *measures the ceiling*: how well does the 1-D strip
  align when pitch is perfect? It ships nothing and uses no roll at inference — it just tells us
  whether to invest in pitch (Lever A) or whether the strip **resolution** is the hard floor
  (→ prioritize Lever B). Decisive, costs an afternoon.
- **E2 — LoRA + pitch-shaping (DAYS).** Unfreeze MERT/ViT via LoRA with Lever-A pitch BCE live.
  The full RC2 test: do the *foundation models themselves* become pitch-aware. (v3_e2e drifted,
  but it fine-tuned ImageNet ViT with no pitch objective on 945 pages — not evidence against a
  pitch-supervised LoRA, per §2 caveat.)
- **E3 — resolution stack (DAYS).** 13-layer MERT sum + 25 ms pooling + drop-`[CLS]` score
  tokens + sub-tile x-regression. Targets the ≤0.25 / ≤0.1 s bins directly.
- **E4 — FiLM + local-position follower (WEEKS).** Lever C on top of the best of E0-E3. The
  candidate that fuses both validated levers (memory × pitch) toward 85 %.

### 9.5 Honest risks specific to v6

- **Frozen MERT pitch may be unrecoverable by a proj head alone** → E0 underperforms and we are
  forced into the costlier LoRA path (E2) before knowing it pays. E1 hedges this: it tells us
  the *pitch ceiling* independent of whether MERT can reach it.
- **Resolution may dominate pitch.** ~1 tile/beat ≈ 0.5 s and 93 ms pooling could cap @0.5 s
  regardless of pitch quality. E1 separates the two; if resolution is the floor, Lever B leads.
- **LoRA drift.** e2e already burned us once. Keep LoRA rank small, encoders mostly frozen, pitch
  head as the *only* new strong gradient, and watch val closely.
- **MERT vs FluidSynth timbre / onset sparsity** — the CQT-CNN frontend control and onset
  pos-weighting/focal loss are the mitigations (unchanged from §5/§7).
- **Still synthetic MSMD.** Real audio / Iranian-classical has no clean pitch targets; sub-second
  on synthetic is the milestone, not the finish (§7).

### 9.6 One line

**The pitch lever was mis-wired, not disproven — fix one line (E0), measure the ceiling for free
(E1), then shape the LoRA foundation embeddings with pitch + onset supervision and fuse them into
the v5 follower. End-to-end, foundation models, no roll at inference — memory × pitch is the path
to SOTA.**

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
