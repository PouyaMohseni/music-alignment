# CB_TA-Ext Implementation Spec

Backbone = proven SOTA (Henkel & Widmer 2021, Frontiers). Novelty = additive
extensions on top, each independently ablated. No replacement of the backbone.
Machine-readable, step-by-step. No prose beyond what's needed to specify behavior.

---

## 0. Why this supersedes the prior CADP-Score spec

The prior spec (dual-encoder + strip-tiling + similarity-matrix + DTW/INR-decode)
is DEPRECATED. Reason: your own data shows the tile-based dual-encoder paradigm
tops out around 60% (M06-equivalent), while the from-scratch CB_TA reproduction
on correct data reaches 64% and is still climbing toward 85% at time of writing.
The tile-quantization ceiling I diagnosed earlier is specific to the paradigm I
chose — it does not apply to CB_TA, which never tiles the score at all. Do not
resume work on the dual-encoder/DTW-matrix line. All new work extends CB_TA.

---

## 1. Global Constants — Backbone (frozen, do not modify)

```yaml
backbone_name: CB_TA
architecture: ConditionalUNet
audio_encoder: CBEncoder
score_input: full_page_portrait          # 1181x835, downscaled /3
gt_position_axis: y                       # vertical scroll per page
unet_encoder_stages: 4
unet_bottleneck: 1
unet_decoder_stages: 4
unet_filters_start: 8                     # doubles each stage: 8,16,32,64,128
film_stages: [2,3,4,5,6,7,8]
audio_spectrogram: log_mel_78band
audio_freq_range_hz: [60, 6000]
audio_fps: 20
audio_context_frames: 40                  # 2 second context window
audio_embed_dim: 32
lstm_hidden: 128
lstm_layers: 1
lstm_state: persistent_across_bptt_reset_per_piece
bptt_seq_len: 128                         # matches paper
gt_heatmap: gaussian_stripe
gt_heatmap_width_px: 10
loss_backbone: dice
dice_smoothing: 0
train_data: zenodo_msmd
train_split_file: msmd_train_complete.yaml
train_pages: 168                          # complete-tempo pages only
train_tempos: [500,750,950,1000,1050,1250,1500]
val_data: zenodo_msmd_valid
test_data: zenodo_msmd_test
params_backbone: 941873
target_metric: pct_within_0.5s
target_value: 0.851
```

Base code: use the existing unmodified `cpjku-paper-train` run's `train_model.py`
and model code (job 64434430). DO NOT REWRITE. All extensions below hook into
this code via subclassing or monkey-patching, never by editing the base files
directly. Copy base files into `cpjku_base/` and treat as read-only reference.

---

## 2. Repository Layout

```
music-alignment/
  cpjku_base/                        # unmodified reference — READ ONLY
    train_model.py
    models/
      cb_encoder.py
      conditional_unet.py
      film.py
    configs/
      msmd_aug.yaml
      msmd_train_complete.yaml
  extensions/
    audio_encoders/
      mert_projector.py              # B1
    heads/
      pitch_aux_head.py              # B2
      inr_subpixel_head.py           # B3
    losses/
      temporal_consistency.py        # B4
      dense_contrastive_aux.py       # B5
      combined_loss.py
    augmentation/
      impulse_response.py            # B6
    hooks/
      film_feature_extractor.py      # exposes intermediate FiLM feature maps
      position_decoder.py            # decodes heatmap -> (x,y) or y-only
  configs/
    ablation/
      A0_baseline_reproduction.yaml
      B1_mert_audio_swap.yaml
      B2_pitch_aux.yaml
      B3_inr_subpixel.yaml
      B4_temporal_consistency.yaml
      B5_dense_contrastive_aux.yaml
      B6_impulse_response_aug.yaml
      B7_combined_best.yaml
  eval/
    metrics.py                       # existing — extend, do not rewrite
    position_decode.py
    run_eval.py
  scripts/
    run_ablation.py
    precompute_mert.py
    precompute_irs.py
  data/
    impulse_responses/               # NEW — download target
```

---

## 3. Milestone A0 — Confirm the Baseline

Before any extension work starts:

1. Let job 64434430 (`cpjku-paper-train`) finish training to convergence or
   early-stop patience.
2. Record final `pct_within_0.5s` on `zenodo_msmd_test`.
3. If final value is within 5 points of 85.1%, proceed to extensions using this
   checkpoint as `A0_checkpoint.pt`.
4. If final value plateaus below 75%, STOP. Diagnose against the paper's own
   ablations (Table 2/3 of Henkel & Widmer 2021) before adding any extension —
   an extension cannot be evaluated meaningfully against a broken baseline.
5. Freeze `A0_checkpoint.pt` and `A0_config.yaml` as the reference point for
   every subsequent ablation's percentage-point comparison.

Do not proceed to B1 until this milestone is closed.

---

## 4. Extension B1 — Clean Foundation-Model Audio Encoder Swap

### 4.1 Rationale

v10 (MERT + ConditionalUNet, 41.4%) and v9 (crop-tracking, no valid result) both
used a crop-tracking setup later confirmed to have a centered-GT shortcut bug
(the model could predict the crop center regardless of audio and still score
well). v10 was run before this bug was diagnosed, so 41.4% is confounded and
not trustworthy evidence about whether MERT features help. This has never been
cleanly tested. B1 is that clean test, built on the fixed full-page/full-strip
BPTT setup (no crop-centering shortcut possible because GT traverses the whole
page over time).

### 4.2 Architecture

Replace `CBEncoder` only. Everything else (U-Net, FiLM, LSTM, Dice loss,
full-page input, BPTT) stays identical to A0.

```python
# extensions/audio_encoders/mert_projector.py

class MERTProjector(nn.Module):
    """
    Drop-in replacement for CBEncoder. Produces (batch, 32) per 20fps timestep,
    matching CBEncoder's output shape and semantics (each timestep summarizes
    a 2-second audio context window).
    """
    def __init__(self, mert_model="m-a-p/MERT-v1-95M", freeze=True):
        self.mert = AutoModel.from_pretrained(mert_model)
        if freeze:
            for p in self.mert.parameters():
                p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(),
            nn.Linear(256, 32),
        )

    def forward(self, waveform, timestep_20fps):
        """
        waveform: (B, num_samples) at 24kHz
        timestep_20fps: int, which 20fps step to produce embedding for
        Returns: (B, 32)
        """
        # MERT operates at 75Hz -> 150 frames = 2 seconds context
        center_sec = timestep_20fps / 20.0
        window_start_sample = int((center_sec - 1.0) * 24000)
        window_end_sample = int((center_sec + 1.0) * 24000)
        clip = waveform[:, max(0, window_start_sample):window_end_sample]
        mert_out = self.mert(clip).last_hidden_state       # (B, ~150, 768)
        pooled = mert_out.mean(dim=1)                        # (B, 768)
        return self.proj(pooled)                             # (B, 32)
```

Precompute mode (preferred for training speed): precompute MERT features for
every 20fps timestep of every training piece ahead of time, cache to disk,
and load precomputed (B, T_20fps, 768) tensors instead of running MERT live.
Only the `proj` head is trained in the frozen variant.

### 4.3 Config

```yaml
name: B1_mert_audio_swap
base: A0_baseline_reproduction
audio_encoder: mert_projector
mert_model: "m-a-p/MERT-v1-95M"
mert_freeze: true                    # stage 1: frozen
mert_lora_rank: 0                    # stage 2 (B1b): try rank 8 if B1 improves
precompute_features: true
lr_projector: 1e-3
lr_backbone: 1e-4                    # everything else fine-tunes at low LR
batch_size: 1
bptt_seq_len: 128
epochs: 80
```

Run two variants:
- **B1a**: MERT fully frozen, only `proj` head trains.
- **B1b**: only run if B1a shows ANY improvement over A0. LoRA rank 8 on MERT's
  attention layers, trained jointly.

### 4.4 Target and stop condition

Target: match or exceed A0's `pct_within_0.5s`. If B1a underperforms A0 by more
than 5 points, this is a real negative result — foundation-model audio features
do not help this task in this architecture. Report it as such. Do not proceed
to B1b if B1a shows no signal.

---

## 5. Extension B2 — Pitch Auxiliary, Correctly Wired

### 5.1 Rationale

Documented bug in prior work (REDESIGN.md §9.1): the pitch head read the frozen
precomputed embedding, not the FiLM-modulated feature that the network actually
uses to localize. Gradient never reached anything load-bearing. This retests
the same hypothesis with the wiring fixed, on the CB_TA backbone.

### 5.2 Architecture

```python
# extensions/heads/pitch_aux_head.py

class PitchAuxHead(nn.Module):
    """
    Attaches to the POST-FiLM decoder feature map, at the ground-truth
    spatial location, not to any frozen/precomputed input.
    """
    def __init__(self, feature_channels, num_pitches=88):
        self.audio_pitch_head = nn.Linear(lstm_hidden, num_pitches)   # from LSTM state
        self.score_pitch_head = nn.Conv2d(feature_channels, num_pitches, kernel_size=1)

    def forward(self, lstm_hidden_state, film_decoder_feature_map, gt_xy):
        """
        lstm_hidden_state: (B, 128) — the actual audio embedding used for FiLM
        film_decoder_feature_map: (B, C, H, W) — POST-FiLM feature, e.g. decoder stage 6
        gt_xy: (B, 2) — ground truth pixel location this timestep
        """
        audio_pitch_logits = self.audio_pitch_head(lstm_hidden_state)   # (B, 88)

        # sample the decoder feature map AT the ground-truth location
        sampled_feature = bilinear_sample(film_decoder_feature_map, gt_xy)  # (B, C)
        score_pitch_logits = self.score_pitch_head(sampled_feature.unsqueeze(-1).unsqueeze(-1))
        score_pitch_logits = score_pitch_logits.squeeze(-1).squeeze(-1)  # (B, 88)

        return audio_pitch_logits, score_pitch_logits
```

Supervision: at every BPTT timestep, ground truth is the set of MIDI pitches
active at that audio timestamp (from `annotations.json`). BCE loss against
an 88-dim multi-hot vector, for both `audio_pitch_logits` and
`score_pitch_logits`. Both heads deleted at inference.

Critical check before trusting any result: verify with a gradient hook that
`sampled_feature` receives nonzero gradient contribution back into the U-Net
decoder weights. If gradient norm at the sampling point is zero or numerically
negligible, the wiring bug has been reproduced — fix before reporting any
result from this ablation.

### 5.3 Config

```yaml
name: B2_pitch_aux
base: A0_baseline_reproduction        # or best of B1 if B1 succeeded
pitch_aux:
  enabled: true
  feature_map_stage: decoder_6         # which post-FiLM stage to attach to
  loss_weight: 0.3
  drop_at_inference: true
lr: 1e-4
epochs: 60
```

### 5.4 Target

Target: improvement over base by any margin at ≤0.1s and ≤0.25s bins
specifically (pitch precision should sharpen fine-grained localization more
than coarse ≤0.5s/≤1.0s bins). Report even a null result — this closes an
open question from your own prior work.

---

## 6. Extension B3 — Sub-Pixel INR Refinement (local, not global)

### 6.1 Rationale

CB_TA's heatmap is decoded via argmax at the U-Net's native output resolution
(page downscaled /3). This introduces a coarse quantization floor on the
sub-second bins (≤0.05s, ≤0.1s) even when the coarse localization is correct.
Unlike the prior spec's global INR-over-whole-page (compute-expensive and
duplicates work the U-Net already does), this is a LOCAL two-stage refinement:
coarse peak from the existing heatmap, then a small subpixel correction.

### 6.2 Architecture

```python
# extensions/heads/inr_subpixel_head.py

class LocalINRRefiner(nn.Module):
    """
    Stage 1: existing U-Net heatmap -> coarse argmax peak (x0, y0), UNCHANGED.
    Stage 2: sample a small window of the pre-output decoder feature map
    around (x0, y0), and an implicit MLP refines to sub-pixel (dx, dy).
    """
    def __init__(self, feature_channels, window_px=8, hidden=128):
        self.window_px = window_px
        fourier_freqs = torch.tensor([1,2,4,8,16])
        self.mlp = nn.Sequential(
            nn.Linear(feature_channels + len(fourier_freqs)*4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),                 # confidence at query offset
        )

    def forward(self, decoder_feature_map, coarse_peak_xy, query_offsets):
        """
        decoder_feature_map: (B, C, H, W) — layer just before the 1x1 output conv
        coarse_peak_xy: (B, 2) — argmax location from the existing heatmap decode
        query_offsets: (Q, 2) — fine sub-pixel offsets to evaluate, e.g. a
                       4x-upsampled grid within +/- window_px of the coarse peak
        Returns: (B, Q) confidence, from which soft-argmax gives refined (dx,dy)
        """
        local_feature = bilinear_sample(decoder_feature_map, coarse_peak_xy)  # (B, C)
        fourier = fourier_encode(query_offsets, self.fourier_freqs)           # (Q, 4*len(freqs))
        combined = concat_broadcast(local_feature, fourier)                  # (B, Q, C+...)
        confidence = self.mlp(combined).squeeze(-1)                          # (B, Q)
        refined_offset = soft_argmax(confidence, query_offsets)              # (B, 2)
        return coarse_peak_xy + refined_offset
```

Supervision: Gaussian heatmap regression loss at continuous ground-truth
(x, y), sigma ~5px in original page coordinates, evaluated at query
resolution (4x finer than the U-Net's native grid).

### 6.3 Config

```yaml
name: B3_inr_subpixel
base: A0_baseline_reproduction        # or best of B1/B2
inr_refiner:
  enabled: true
  window_px: 8
  query_resolution_multiplier: 4
  heatmap_sigma_px: 5
  loss_weight: 1.0
  attach_to_stage: decoder_final       # layer just before 1x1 output conv
lr: 1e-4
epochs: 40                            # can fine-tune on top of a converged base
```

### 6.4 Target

Target: measurable lift at ≤0.05s and ≤0.1s bins specifically, with ≤0.5s bin
unchanged or improved (should not regress coarse localization). This is the
item most likely to close gap on the tightest thresholds where CB_TA's own
published numbers (Henkel & Widmer 2021) are weakest relative to ≤0.5s.

---

## 7. Extension B4 — Temporal Path Consistency Loss

### 7.1 Rationale

Dice loss is per-frame and per-pixel; nothing explicitly penalizes a decoded
position sequence that jitters or moves backward across a BPTT window. This
operates on the decoded 1-D/2-D position sequence, which is cheap (no need to
touch the 2-D heatmap loss).

### 7.2 Architecture

```python
# extensions/losses/temporal_consistency.py

def temporal_consistency_loss(pred_positions, gt_positions, gamma=0.5):
    """
    pred_positions: (B, T, 2) or (B, T, 1) if y-only — decoded per BPTT step
    gt_positions:   (B, T, 2) or (B, T, 1)
    gamma: SoftDTW smoothing (only used if sequence lengths can differ;
           if lengths match 1:1 per BPTT step, use direct L1 + monotonicity
           penalty instead, which is cheaper and exact)
    """
    # Primary: direct position error (redundant with Dice-derived position
    # error, included for gradient signal on the DECODE path specifically)
    l1 = (pred_positions - gt_positions).abs().mean()

    # Monotonicity penalty: page-position must not move backward in time
    # (score doesn't have repeats after filtering; direction is known)
    delta = pred_positions[:, 1:] - pred_positions[:, :-1]
    backward_penalty = torch.relu(-delta).mean()     # penalize negative steps

    # Jerk penalty: discourage frame-to-frame jitter beyond plausible tempo range
    accel = delta[:, 1:] - delta[:, :-1]
    jerk_penalty = accel.pow(2).mean()

    return l1 + 2.0 * backward_penalty + 0.1 * jerk_penalty
```

This loss is added on top of Dice, not instead of it. It operates on the
already-differentiable decode (soft-argmax on the heatmap, not hard argmax —
use soft-argmax during training so gradient flows through the position).

### 7.3 Config

```yaml
name: B4_temporal_consistency
base: A0_baseline_reproduction         # or best of B1/B2/B3
temporal_consistency:
  enabled: true
  decode_mode: soft_argmax             # must be differentiable for this loss
  weight_l1: 1.0
  weight_backward: 2.0
  weight_jerk: 0.1
loss_combination: dice + temporal_consistency
lr: 1e-4
epochs: 40
```

### 7.4 Target

Target: reduction in mean/median error and reduction in variance of per-piece
error (fewer catastrophic mid-piece derailments). Track a new metric:
`max_backward_jump_px` per piece — should approach zero.

---

## 8. Extension B5 — Dense Contrastive Deep Supervision

### 8.1 Rationale

Adapts DenseAV's (Hamilton et al., CVPR 2024) bidirectional token-alignment
idea to an architecture that has no similarity matrix. Applied as an auxiliary
loss at an intermediate FiLM-modulated feature map: the feature vector at the
ground-truth location should be more similar to the audio embedding than
feature vectors at other sampled page locations.

### 8.2 Architecture

```python
# extensions/losses/dense_contrastive_aux.py

def dense_contrastive_aux_loss(film_feature_map, lstm_hidden_state, gt_xy,
                                 num_negatives=32, temperature=0.07):
    """
    film_feature_map: (B, C, H, W) — an intermediate POST-FiLM decoder stage
    lstm_hidden_state: (B, 128) — audio embedding for this timestep
    gt_xy: (B, 2) — ground truth pixel location
    """
    audio_proj = nn.Linear(128, C)(lstm_hidden_state)          # (B, C)
    audio_proj = F.normalize(audio_proj, dim=-1)

    positive_feature = bilinear_sample(film_feature_map, gt_xy)  # (B, C)
    positive_feature = F.normalize(positive_feature, dim=-1)
    positive_sim = (audio_proj * positive_feature).sum(-1)        # (B,)

    negative_coords = sample_random_coords(film_feature_map.shape,
                                            exclude_radius_px=30,
                                            gt_xy=gt_xy,
                                            n=num_negatives)        # (B, num_negatives, 2)
    negative_features = bilinear_sample_batch(film_feature_map, negative_coords)
    negative_features = F.normalize(negative_features, dim=-1)
    negative_sims = torch.einsum('bc,bnc->bn', audio_proj, negative_features)

    logits = torch.cat([positive_sim.unsqueeze(1), negative_sims], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)
```

### 8.3 Config

```yaml
name: B5_dense_contrastive_aux
base: A0_baseline_reproduction          # or best of B1-B4
dense_contrastive:
  enabled: true
  feature_map_stage: decoder_5
  num_negatives: 32
  exclude_radius_px: 30
  temperature: 0.07
  loss_weight: 0.2
lr: 1e-4
epochs: 40
```

### 8.4 Target

Target: improved robustness on repetitive passages specifically. Track a new
metric: `pct_within_0.5s` on a held-out subset of pieces flagged as having
repetitive/sequential passages, compared against the same metric on
non-repetitive pieces. Improvement should be larger on the repetitive subset.

---

## 9. Extension B6 — Real-Audio Robustness via Impulse Response Augmentation

### 9.1 Rationale

Not novel — this replicates what Henkel & Widmer 2021 (Frontiers) already do to
generalize beyond synthetic MSMD audio. Necessary before any real-audio
evaluation tier is meaningful. Runs in parallel with B1-B5, not sequentially.

### 9.2 Implementation

```python
# extensions/augmentation/impulse_response.py

IR_SOURCES = [
    "Aachen Impulse Response Database (AIR)",
    "MIT McDermott Impulse Response Survey",
    "OpenAIR",
]

def apply_random_ir_augmentation(waveform, ir_bank, p=0.5,
                                   snr_range_db=(10, 30)):
    """
    waveform: (num_samples,) synthetic MSMD audio
    ir_bank: preloaded list of impulse response arrays
    """
    if random.random() > p:
        return waveform
    ir = random.choice(ir_bank)
    convolved = scipy.signal.fftconvolve(waveform, ir, mode='same')
    convolved = normalize_to_original_rms(convolved, waveform)
    noise = generate_pink_noise(len(waveform))
    snr_db = random.uniform(*snr_range_db)
    return mix_at_snr(convolved, noise, snr_db)
```

### 9.3 Config

```yaml
name: B6_impulse_response_aug
base: best_of_B1_through_B5
augmentation:
  impulse_response:
    enabled: true
    probability: 0.5
    snr_range_db: [10, 30]
    ir_sources: [AIR, MIT_survey, OpenAIR]
epochs: 40                            # fine-tune on top of converged checkpoint
eval_tiers: [msmd_rec, magaloff, zeilinger]   # only meaningful tier for this ablation
```

### 9.4 Target

Target: reduced degradation on Tier 2/3 (real audio) relative to the
non-augmented checkpoint's Tier 2/3 performance. This ablation is judged on
real-audio tiers only; synthetic MSMD performance is not the point.

---

## 10. Milestone Order (execute in this order)

1. **A0**: confirm baseline reproduces close to 85.1%. STOP if it doesn't.
2. **B1a**: clean frozen-MERT swap. This is the highest-value open question.
   If no improvement, document as a clean negative result and move on.
3. **B1b**: only if B1a improved. LoRA fine-tune MERT.
4. **B2**: pitch auxiliary, correctly wired. Verify gradient flow before
   trusting any number.
5. **B3**: sub-pixel INR refinement. Check ≤0.05s/≤0.1s bins specifically.
6. **B4**: temporal consistency loss. Check backward-jump metric.
7. **B5**: dense contrastive auxiliary. Check repetitive-passage subset.
8. **B6**: impulse response augmentation, run in parallel with 2-7 on a
   separate GPU if available. Evaluate only on real-audio tiers.
9. **B7 — combined**: take every extension that showed a positive delta over
   its base in isolation, combine them into one model, retrain from A0
   checkpoint, evaluate on all tiers.
10. **Ablation table**: run `scripts/run_ablation.py`, produce final CSV
    comparing A0, B1-B6 individually, and B7 combined.

Do not combine extensions before each has been individually ablated. An
extension that shows zero or negative delta in isolation is excluded from B7
regardless of architectural appeal.

---

## 11. Evaluation Tiers (unchanged structure, backbone-appropriate data)

```yaml
Tier1_MSMD_synthetic:
  data: zenodo_msmd_test
  purpose: primary comparability with Henkel 2020/2021 Table 3

Tier2_MSMD_Rec:
  data: msmd_rec (real piano performances of MSMD pieces)
  purpose: real-audio generalization, zero-shot from Tier1-trained checkpoint

Tier3_RealConcert:
  data: magaloff + zeilinger
  purpose: real-audio + scanned-score generalization

Tier4_Radif_pilot:
  data: radif_corpus_synthesized
  purpose: cross-tradition transfer, qualitative + small-scale quantitative
  note: only run on B7 combined model, not on individual ablations
```

Primary metric: `pct_within_0.5s`. Secondary: `pct_within_0.05s`,
`pct_within_0.1s`, `pct_within_0.25s`, `pct_within_1.0s`, `mean_error_sec`,
`median_error_sec`, `max_backward_jump_px` (new, from B4).

---

## 12. Guardrails

- Never trust a B2 (pitch aux) result without confirming nonzero gradient at
  the sampling point. This bug has already happened once.
- Never trust a B1 result computed with the old crop-tracking code path
  (v9/v10 lineage). Only the fixed full-page/full-strip BPTT setup is valid.
- Every ablation must report delta against A0's frozen checkpoint number, not
  against another ablation's number.
- Every ablation config must be runnable standalone from `A0_checkpoint.pt`
  without requiring any other ablation's code.
- Log per-loss-component breakdown for every combined loss (B4, B5, B7).
- If B7 (combined) underperforms the best individual ablation, do not average
  weights or ensemble as a workaround — diagnose the interaction first.
- Do not run Tier2-4 evaluation on any ablation except B7 combined, to avoid
  overfitting extension design decisions to real-audio tiers with limited data.

---

## 13. Do Not

- Do not resume the tile-based dual-encoder + similarity-matrix + DTW
  paradigm from the prior spec. It is deprecated.
- Do not modify `cpjku_base/` files directly. All changes via `extensions/`.
- Do not run B1b (LoRA) unless B1a (frozen) shows a positive signal first.
- Do not add YTSV pretraining, diffusion models, VLMs, or LLM components —
  out of scope for this spec.
- Do not claim any result without it passing through `eval/metrics.py`.
- Do not combine untested extensions directly into B7 — each needs its own
  ablation row first.
- Do not evaluate real-audio tiers until B6 (impulse response aug) is applied
  to whatever checkpoint is being evaluated on those tiers.

---

## 14. Output Files

```
results/<ablation_name>/
  config.yaml
  checkpoints/
    best.pt
    last.pt
  metrics.json
  gradient_check.json          # for B2 only — confirms wiring is correct
  position_sequences/          # decoded (x,y) or y-only sequences, first 10 pieces
  train.log
  val.log
```

Aggregate: `results/ablation_summary.csv` with columns:
`ablation, delta_vs_A0_pct0.5s, pct@0.05s, pct@0.1s, pct@0.5s, pct@1.0s, mean_err_s, max_backward_jump_px`

---

## 15. Environment

```
python==3.11
torch>=2.2
transformers>=4.40          # MERT loader (B1)
peft>=0.11                  # LoRA (B1b)
scipy                       # impulse response convolution (B6)
librosa
numpy
wandb
```

Base CPJKU code's own environment (madmom, existing deps) takes precedence —
do not upgrade/downgrade packages the base code depends on without verifying
A0 still reproduces after the change.
