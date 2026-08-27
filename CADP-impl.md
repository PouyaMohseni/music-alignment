# CADP-Score Implementation Spec

Machine-readable, step-by-step. No prose. Each section is a directive.

---

## 0. Global Constants

```yaml
project_root: music-alignment/
audio_sr: 24000
mert_frame_rate_hz: 75
audio_window_sec: 5.0
audio_frames_per_window: 375        # 5.0 * 75
strip_height_px: 224
strip_column_width_px: 80
strip_column_stride_px: 40           # 50% overlap
vit_input_size: 224                  # resize each 80x224 column to 224x224
shared_embed_dim: 256
mert_model: "m-a-p/MERT-v1-95M"
mert_hidden_dim: 768
vit_model_default: "facebook/dinov2-base"
vit_hidden_dim: 768
vit_patch_size: 14
device: "cuda"
seed: 42
```

---

## 1. Repository Layout

```
music-alignment/
  configs/
    dataset/msmd.yaml
    dataset/ytsv.yaml
    dataset/radif.yaml
    model/M01_frozen_baseline.yaml
    model/M02_lora_baseline.yaml
    model/M03_lstm_temporal.yaml
    model/M04_dense_tokens.yaml
    model/M05_learned_path.yaml
    model/M06_inr_head.yaml
    model/M07_full_cadp.yaml
    train/pretrain_ytsv.yaml
    train/finetune_msmd.yaml
  data/
    msmd_prep/            # existing pipeline; do not rewrite
    ytsv_prep/            # NEW
    radif_prep/           # NEW
  models/
    encoders/
      mert_audio.py
      vit_score.py
      lora_wrappers.py
    heads/
      projection.py
      lstm_temporal.py
      path_predictor.py
      inr_head.py
    losses/
      softdtw.py
      dense_contrastive.py
      heatmap_inr.py
      combined.py
    architectures/
      M01_frozen_baseline.py
      M02_lora_baseline.py
      M03_lstm_temporal.py
      M04_dense_tokens.py
      M05_learned_path.py
      M06_inr_head.py
      M07_full_cadp.py
  train/
    trainer.py
    schedulers.py
    checkpointing.py
  eval/
    metrics.py            # existing; extend
    dtw_decode.py
    decdtw_decode.py      # NEW
    inr_decode.py         # NEW
    run_eval.py
  scripts/
    precompute_mert.py
    precompute_dinov2.py
    train_one.py
    eval_one.py
    run_ablation.py
```

---

## 2. Data Pipeline Directives

### 2.1 MSMD (assume existing, verify format)

Assert per piece:
```
data/msmd/<split>/<piece_id>/
  strip.png                # unrolled score strip, height=224 px
  audio.wav                # 24kHz mono
  annotations.json         # per-notehead: onset_sec, strip_x_px, pitch_midi
  mapping.json             # strip_x_px -> (page_idx, x_page_px, y_page_px)
```

If format differs, adapt loaders — do NOT rewrite the extraction pipeline.

### 2.2 YTSV pipeline (new)

Steps:
1. Download timestamps + URLs from https://github.com/sogang-mir/u-must (U-MusT release)
2. For each video, extract audio at 24kHz mono, and extract score frames at 1 fps
3. Detect score-page transitions via SSIM between consecutive frames (threshold 0.85)
4. For each contiguous same-page segment, save one representative frame
5. Weak alignment: video timestamp of frame k -> assume linear time-to-position mapping within that page segment
6. Filter out segments shorter than 10 sec or longer than 300 sec
7. Save to same schema as MSMD but with `annotations_weak.json` instead of `annotations.json`

Expected yield: ~800-1000 hours after filtering.

### 2.3 Radif pipeline (new)

Steps:
1. Download Radif Corpus MIDI + CSV from Zenodo (record 17811549)
2. Synthesise audio with FluidSynth using a tar soundfont (tar_arasbaran.sf2 recommended)
3. Engrave notation via LilyPond with Persian notation stylesheet OR use pre-rendered PDFs from corpus if available
4. Extract notehead positions programmatically from LilyPond output
5. Build annotations.json in MSMD schema
6. Handle quarter-tones: add `pitch_bend_cents` field to annotations (may be non-zero for koron/sori notes)

Yield: 228 pieces, ~281 minutes total audio.

### 2.4 Precomputation

```
scripts/precompute_mert.py --dataset msmd --split all --batch_size 4
  -> data/msmd/<piece_id>/mert_features.npz
     Shape: (T, 768) where T = audio_frames at 75 Hz

scripts/precompute_dinov2.py --dataset msmd --split all --batch_size 32
  -> data/msmd/<piece_id>/dinov2_features.npz
     Shape: (num_columns, num_patches_per_column, 768)
     num_patches_per_column = 16 (224/14)
     num_columns = (strip_width - 80) / 40 + 1
```

Precompute for ALL datasets before any training run.

---

## 3. Model Architectures — Seven Ablation Configurations

Each subsection defines ONE model. Implement each as a separate class in `models/architectures/`.

---

### M01 — Frozen Baseline (control)

Purpose: reproduce baseline behavior of previous v3_all.

Architecture:
```
audio_input: (B, T_a=375, 768)   # frozen precomputed MERT
score_input: (B, N_cols, 16, 768) # frozen precomputed DINOv2

audio_proj = Linear(768, 256)      # trainable
score_proj = Linear(768, 256)      # trainable

# Pool audio: 375 -> ~20 tokens by chunking, take mean per chunk
audio_pooled = chunk_mean(audio_input, chunks=20)  -> (B, 20, 768)
audio_emb = L2Norm(audio_proj(audio_pooled))       -> (B, 20, 256)

# Pool score: mean over patches within column
score_pooled = mean(score_input, dim=2)            -> (B, N_cols, 768)
score_emb = L2Norm(score_proj(score_pooled))       -> (B, N_cols, 256)

sim_matrix = audio_emb @ score_emb.T               -> (B, 20, N_cols)
```

Loss: `expected_distance_loss` (existing) on sim_matrix vs ground-truth strip_x.

Trainable params: audio_proj + score_proj ≈ 400K.

Config: `configs/model/M01_frozen_baseline.yaml`:
```yaml
name: M01_frozen_baseline
audio_pool_chunks: 20
loss: expected_distance
loss_weight: 1.0
lr: 1e-3
batch_size: 8
epochs: 30
```

---

### M02 — Frozen + LoRA (baseline+)

Purpose: test whether LoRA adaptation of encoders helps.

Same as M01 but:
- MERT has LoRA rank 4 applied to `q_proj`, `v_proj` in all 12 layers, trainable
- DINOv2 has LoRA rank 4 applied to `q_proj`, `v_proj` in all 12 layers, trainable
- Cannot use precomputed features; must run encoders live

Trainable params: audio_proj + score_proj + LoRA weights ≈ 1.5M.

Config: `configs/model/M02_lora_baseline.yaml`:
```yaml
name: M02_lora_baseline
mert_lora_rank: 4
vit_lora_rank: 4
audio_pool_chunks: 20
loss: expected_distance
loss_weight: 1.0
lr_encoders: 5e-5
lr_heads: 1e-3
batch_size: 4                  # smaller due to live encoders
epochs: 25
```

---

### M03 — LSTM Temporal Follower (v5 replica)

Purpose: reproduce v5_recurrent — confirm temporal memory contribution.

Architecture:
```
# Same as M01 encoders + projections
audio_emb: (B, 20, 256)
score_emb: (B, N_cols, 256)

# NEW: LSTM over audio
lstm = LSTM(input_size=256, hidden_size=512, num_layers=2,
            bidirectional=True, batch_first=True)
query_proj = Linear(1024, 256)

audio_lstm_out, _ = lstm(audio_emb)                # (B, 20, 1024)
audio_query = query_proj(audio_lstm_out)            # (B, 20, 256)
audio_query = L2Norm(audio_query)

sim_matrix = audio_query @ score_emb.T              # (B, 20, N_cols)
```

Loss: `expected_distance_loss` + optional cross-entropy on argmax column.

Config: `configs/model/M03_lstm_temporal.yaml`:
```yaml
name: M03_lstm_temporal
audio_pool_chunks: 20
lstm_hidden: 512
lstm_layers: 2
lstm_bidirectional: true
loss: expected_distance
loss_weight: 1.0
lr: 5e-4
batch_size: 8
epochs: 40
```

Expected on MSMD test: ~33% @ 0.5s (matches v5m_big).

---

### M04 — Dense Tokens (no pooling)

Purpose: test whether keeping dense audio + dense score tokens helps.

Architecture:
```
audio_input: (B, 375, 768)          # NO pooling
score_input: (B, N_cols, 16, 768)   # flatten to (B, N_cols*16, 768)

audio_proj: TwoLayerMLP(768 -> 512 -> 256)
score_proj: TwoLayerMLP(768 -> 512 -> 256)

# NEW: pool DINOv2 patches VERTICALLY only (collapse height dim)
# reshape: (B, N_cols, 4, 4, 768) then mean over height axis (dim=2)
score_reshaped = reshape(score_input, (B, N_cols, 4, 4, 768))
score_vpool = mean(score_reshaped, dim=2)           # (B, N_cols, 4, 768)
score_flat = flatten(score_vpool, dims=1,2)         # (B, N_cols*4, 768)

audio_emb = L2Norm(audio_proj(audio_input))         # (B, 375, 256)
score_emb = L2Norm(score_proj(score_flat))          # (B, N_cols*4, 256)

sim_matrix = audio_emb @ score_emb.T                # (B, 375, N_cols*4)
```

Loss: expected_distance_loss on dense matrix + dense contrastive (see 4.2).

Config: `configs/model/M04_dense_tokens.yaml`:
```yaml
name: M04_dense_tokens
audio_pool: none
score_vertical_pool: mean
score_horizontal_positions_per_column: 4
loss:
  - expected_distance: 1.0
  - dense_contrastive: 0.5
lr: 5e-4
batch_size: 4                     # matrix is bigger
epochs: 40
```

---

### M05 — Learned Path Predictor

Purpose: replace DTW inference with a learned convolutional-attentional predictor.

Add to M04:
```
# Take sim_matrix (B, N_a, N_s) as input to path predictor
class PathPredictor(nn.Module):
    def __init__(self, n_audio_max=400, n_score_max=1000, channels=64):
        # Treat sim matrix as single-channel image (B, 1, N_a, N_s)
        self.conv_stem = nn.Sequential(
            Conv2d(1, 32, kernel=3, padding=1), ReLU(),
            Conv2d(32, 64, kernel=3, padding=1), ReLU(),
            Conv2d(64, 128, kernel=3, padding=1), ReLU(),
        )
        # Attention along score axis
        self.axial_attn_score = MultiheadAttention(128, num_heads=4)
        # Attention along audio axis
        self.axial_attn_audio = MultiheadAttention(128, num_heads=4)
        # Head: for each audio frame, output logits over score positions
        self.head = Linear(128, 1)                   # per-cell logit

    def forward(self, sim):
        # sim: (B, N_a, N_s)
        x = sim.unsqueeze(1)                          # (B, 1, N_a, N_s)
        x = self.conv_stem(x)                         # (B, 128, N_a, N_s)
        # Axial attention: reshape and apply along each axis
        ...
        logits = self.head(x_features)                # (B, N_a, N_s, 1)
        return logits.squeeze(-1)
```

Path prediction: `soft_argmax_over_score(logits)` -> continuous strip_x per audio frame.

Loss: SoftDTW between predicted path and ground-truth path.

Config: `configs/model/M05_learned_path.yaml`:
```yaml
name: M05_learned_path
path_predictor:
  conv_channels: [32, 64, 128]
  attention_heads: 4
  attention_dim: 128
loss:
  - softdtw_path: 1.0
  - dense_contrastive: 0.5
softdtw_gamma: 0.5
lr: 3e-4
batch_size: 4
epochs: 50
```

---

### M06 — INR Head (continuous position)

Purpose: break the tile-quantization resolution ceiling.

Add to M05:
```
# Replace soft_argmax with INR head
class INRHead(nn.Module):
    def __init__(self, cond_dim=64, hidden=256):
        # Encoding for query x-coordinate (Fourier features)
        self.fourier_freqs = torch.tensor([1, 2, 4, 8, 16, 32])
        # MLP that takes [fourier(x), cond_vec] -> confidence
        self.mlp = nn.Sequential(
            Linear(len(self.fourier_freqs)*2 + cond_dim, hidden), ReLU(),
            Linear(hidden, hidden), ReLU(),
            Linear(hidden, 1),
        )

    def forward(self, cond_vec, x_query):
        # cond_vec: (B, N_a, cond_dim) -- one per audio frame
        # x_query: (Q,) -- continuous coordinates to evaluate
        # Returns: (B, N_a, Q) confidence scores
        fourier = fourier_encode(x_query, self.fourier_freqs)  # (Q, 2F)
        # Broadcast and concat with cond_vec
        ...
        return confidence


# Modify path predictor:
class PathPredictorINR(nn.Module):
    ...
    self.cond_extractor = Linear(128, 64)      # per-frame condition
    self.inr_head = INRHead(cond_dim=64)

    def forward(self, sim, query_x=None):
        features = self.conv_stem(...)               # (B, 128, N_a, N_s)
        cond_per_frame = mean_over_score_axis(features)  # (B, N_a, 128)
        cond_vec = self.cond_extractor(cond_per_frame)   # (B, N_a, 64)
        if query_x is None:
            # Default: fine grid over score range
            query_x = torch.linspace(0, N_s, N_s*4)
        confidence = self.inr_head(cond_vec, query_x)    # (B, N_a, Q)
        # Continuous position via soft-argmax
        predicted_pos = soft_argmax(confidence, query_x) # (B, N_a)
        return confidence, predicted_pos
```

Loss additions:
- `heatmap_inr_loss`: Gaussian target at ground-truth strip_x for each annotated onset

Config: `configs/model/M06_inr_head.yaml`:
```yaml
name: M06_inr_head
path_predictor: (same as M05)
inr_head:
  cond_dim: 64
  hidden_dim: 256
  fourier_freqs: [1, 2, 4, 8, 16, 32]
  query_resolution_multiplier: 4      # queries at 4x score grid resolution
heatmap_sigma_px: 20                   # Gaussian width for INR supervision
loss:
  - softdtw_continuous: 1.0
  - dense_contrastive: 0.5
  - heatmap_inr: 2.0
lr: 3e-4
batch_size: 4
epochs: 50
```

---

### M07 — Full CADP-Score (all components)

Purpose: the flagship system.

Additions over M06:
1. Concatenate raw transcription features to audio input (see 3.7.1)
2. Enable LoRA adaptation on both MERT and DINOv2
3. Optional: DecDTW post-processing at inference

3.7.1 Transcription auxiliary channel:
```python
# Load pretrained piano transcription model (Kong et al. 2021)
# hf: "juancopi81/onsets-frames-transcription"
# For each audio frame, extract:
#   - onset probability (88 pitches): (T, 88)
#   - frame probability (88 pitches): (T, 88)
# Total aux channel: (T, 176)
# Concatenate to MERT features: (T, 768 + 176) = (T, 944)
# audio_proj input dim becomes 944
```

Config: `configs/model/M07_full_cadp.yaml`:
```yaml
name: M07_full_cadp
mert_lora_rank: 8
vit_lora_rank: 8
use_transcription_aux: true
audio_input_dim: 944                    # 768 MERT + 176 aux
path_predictor: (same as M05)
inr_head: (same as M06)
loss:
  - softdtw_continuous: 1.0
  - dense_contrastive: 0.5
  - heatmap_inr: 2.0
lr_encoders: 5e-5
lr_heads: 3e-4
batch_size: 2                            # LoRA + live inference expensive
epochs: 60
inference:
  postproc: decdtw
  decdtw_max_iters: 30
```

---

## 4. Loss Implementations

### 4.1 SoftDTW (existing, verify)

Assume `pysdtw` or custom implementation exists. Sanity check:
```python
from models.losses.softdtw import SoftDTW
sdtw = SoftDTW(gamma=0.5, use_cuda=True)
loss = sdtw(pred_path, true_path)      # both (B, T, 1)
```

### 4.2 Dense Contrastive (new)

```python
def dense_contrastive_loss(sim_matrix, gt_path, radius_px=40, tau=0.07):
    """
    sim_matrix: (B, N_a, N_s), already L2-normalized rows and cols
    gt_path: (B, N_annotated, 2) with (audio_frame_idx, score_pos_px)
    radius_px: acceptable distance in score pixels for positive
    """
    # Bidirectional InfoNCE:
    # For each ground-truth (t, x_gt): positive is sim[t, floor(x_gt/stride)]
    # Negatives are all other columns in the same row, and all other rows in the same column
    # Standard InfoNCE with temperature tau
    ...
```

### 4.3 Heatmap INR (new)

```python
def heatmap_inr_loss(inr_confidence, gt_path, x_query, sigma_px=20):
    """
    inr_confidence: (B, N_a, Q), softmaxed along Q
    gt_path: (B, N_annotated, 2) with (audio_frame_idx, score_pos_px)
    x_query: (Q,) continuous coordinates
    """
    # For each (t, x_gt), build target Gaussian centered at x_gt
    # Cross-entropy between inr_confidence[b, t, :] and target
    ...
```

### 4.4 Combined loss module

```python
class CombinedLoss(nn.Module):
    def __init__(self, weights: dict):
        self.weights = weights
        # e.g. {'softdtw': 1.0, 'contrastive': 0.5, 'heatmap': 2.0}

    def forward(self, outputs, batch):
        total = 0
        breakdown = {}
        if 'softdtw' in self.weights:
            l = softdtw(outputs.pred_path, batch.true_path)
            total += self.weights['softdtw'] * l
            breakdown['softdtw'] = l.item()
        if 'contrastive' in self.weights:
            l = dense_contrastive(outputs.sim_matrix, batch.gt_path)
            total += self.weights['contrastive'] * l
            breakdown['contrastive'] = l.item()
        if 'heatmap' in self.weights:
            l = heatmap_inr(outputs.inr_confidence, batch.gt_path,
                            outputs.x_query)
            total += self.weights['heatmap'] * l
            breakdown['heatmap'] = l.item()
        return total, breakdown
```

---

## 5. Training Directives

### 5.1 Optimizers

- Encoder LoRA weights: AdamW, lr from config, weight_decay=0.01
- Head weights: AdamW, lr from config, weight_decay=0
- Warmup: linear over first 500 steps
- Schedule: cosine decay to 10% of peak over remaining epochs

### 5.2 Curriculum

For M04-M07:
- First 5 epochs: heatmap_inr loss ONLY (weight ×5), others ×0
- Epochs 5-15: enable dense_contrastive at full weight
- Epoch 15+: enable softdtw_continuous at full weight
- Motivation: sharpen the INR head before applying sequence-level supervision.

### 5.3 Batch construction

```python
class AlignmentBatch:
    audio_features: Tensor        # (B, T_a, 768) or (T_a, 944)
    score_features: Tensor        # (B, N_cols, 16, 768)
    strip_widths_px: Tensor       # (B,) — variable per piece
    gt_path: List[Tensor]         # per-piece list of (n_notes, 2)
    piece_ids: List[str]
```

Random 5-second window per piece per epoch. Pad batch to max within batch, use masking.

### 5.4 Validation

After each epoch on validation split:
- Compute `pct_within_0.5s`, `mean_error_sec`, `median_error_sec`
- Save best checkpoint on `pct_within_0.5s`
- Early stopping patience: 10 epochs

### 5.5 Logging

Tensorboard/wandb tags: `{model_name}/{dataset}/{split}/{metric}`.
Log every 100 steps: loss breakdown, learning rates, GPU memory.
Log every epoch: eval metrics, sample sim matrix images, sample predicted paths.

---

## 6. YTSV Pretraining Stage

Only for M07 (and optionally M06). Run BEFORE MSMD fine-tuning.

Config: `configs/train/pretrain_ytsv.yaml`:
```yaml
dataset: ytsv
initial_checkpoint: null
epochs: 20
lr_encoders: 3e-5
lr_heads: 1e-4
batch_size: 2
loss_weights:
  softdtw_continuous: 0.5      # lower — labels are noisy
  dense_contrastive: 1.0        # higher — retrieval-like signal is cleaner
  heatmap_inr: 1.0
```

Then MSMD fine-tuning uses the YTSV-pretrained checkpoint as initial state.

---

## 7. Evaluation Directives

### 7.1 Existing metrics — verify they compute:

```python
metrics = {
    'pct_within_0.05s': ...,
    'pct_within_0.1s': ...,
    'pct_within_0.25s': ...,
    'pct_within_0.5s': ...,      # PRIMARY
    'pct_within_1.0s': ...,
    'pct_within_5.0s': ...,
    'mean_error_sec': ...,
    'median_error_sec': ...,
    'recall_at_1': ...,           # for retrieval sanity
}
```

### 7.2 Inference decoders (each model may use one)

- `dtw_decode`: standard DTW on sim_matrix, returns discrete tile indices
- `soft_argmax_decode`: pointwise argmax over sim_matrix rows, returns tile centers
- `inr_decode`: query INR head at fine grid, soft-argmax, returns continuous px
- `decdtw_decode`: continuous-time DTW on inr_decode output, returns refined continuous px

Wire each model to its decoder in `scripts/eval_one.py`.

### 7.3 Evaluation tiers

```yaml
Tier1_MSMD:
  train: msmd_train (354 pieces)
  test: msmd_test (94 pieces)
  purpose: comparability with Henkel 2020/2021 numbers

Tier2_YTSV:
  train: ytsv_train_subset (100 hrs held-in)
  test: ytsv_test (10 hrs held-out)
  purpose: real-audio + real-score generalization

Tier3_RealConcert:
  train: none (zero-shot from MSMD checkpoint)
  test: msmd_rec + magaloff + zeilinger
  purpose: real-audio robustness of MSMD-trained model

Tier4_Radif:
  train: none (zero-shot)
  test: radif_all (228 pieces)
  purpose: cross-tradition transfer
```

### 7.4 Ablation runner

```python
# scripts/run_ablation.py
ABLATION_MODELS = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07']
TIERS = ['Tier1_MSMD', 'Tier2_YTSV', 'Tier3_RealConcert', 'Tier4_Radif']

for model_name in ABLATION_MODELS:
    for tier in TIERS:
        if tier != 'Tier1_MSMD' and model_name != 'M07':
            continue    # only full model runs Tier 2-4
        train(model_name, tier)
        results = evaluate(model_name, tier)
        save_results(model_name, tier, results)

produce_final_table()   # writes results/ablation_summary.csv
```

---

## 8. Milestone Order (execute in this order)

1. **Reproduce baseline**: implement M01, train on MSMD, verify ≥15% @ 0.5s. Sanity check.
2. **Reproduce v5**: implement M03, train on MSMD, verify ≥30% @ 0.5s. Sanity check.
3. **Dense tokens**: implement M04, train on MSMD, target ≥35% @ 0.5s.
4. **Learned path**: implement M05, train on MSMD, target ≥50% @ 0.5s.
5. **INR head**: implement M06, train on MSMD, target ≥70% @ 0.5s. **Critical milestone** — this is where the resolution ceiling should be broken.
6. **Full CADP-Score**: implement M07, train on MSMD alone first, target ≥80% @ 0.5s.
7. **YTSV precompute** and pretrain M07, then fine-tune on MSMD, target ≥85% @ 0.5s. **This is the paper-beating number.**
8. **Real-audio evaluation**: zero-shot M07 on Tier 3, target ≥60% @ 0.5s.
9. **Radif pilot**: zero-shot M07 on Tier 4, report qualitative + quantitative results.
10. **Ablation table**: run `scripts/run_ablation.py`, produce final CSV.

Do NOT skip earlier milestones to jump to M07. Each earlier model teaches something about what breaks and where.

---

## 9. Guardrails

- Every new model must produce a `sim_matrix` visualization in the first epoch, saved to disk. If the matrix looks like uniform noise, stop training and inspect.
- Every model must log the loss breakdown per component. If one loss dominates by >10x the others by epoch 5, adjust weights.
- Every checkpoint must be reproducible from its config file + random seed.
- Never train on YTSV without first verifying MSMD baseline reproduces (M01 ≥ 15%).
- Never claim a result without running the ablation runner end-to-end.

---

## 10. Output Files (per model, per tier)

```
results/<model_name>/<tier_name>/
  config.yaml                # frozen config
  checkpoints/
    best.pt
    last.pt
  metrics.json               # all metrics from 7.1
  sim_matrix_samples/        # first 10 pieces
  predicted_paths_samples/   # first 10 pieces
  train.log
  val.log
```

Aggregate table `results/ablation_summary.csv`:

| model | tier | pct@0.05s | pct@0.1s | pct@0.5s | pct@1.0s | mean_err_s | median_err_s |

---

## 11. Do Not

- Do not rewrite the MSMD data pipeline.
- Do not add speculative components not listed here (no diffusion, no VLMs, no LLM prompting).
- Do not use LoRA on M01/M03/M04/M05 unless explicitly listed.
- Do not evaluate on real-audio tiers until MSMD Tier 1 hits target.
- Do not report a number that hasn't gone through `eval/metrics.py`.
- Do not skip precomputation and encode features live for M01/M03/M04.
- Do not train two models simultaneously on the same GPU.

---

## 12. Environment

```
python==3.11
torch>=2.2
transformers>=4.40         # MERT + DINOv2 loaders
peft>=0.11                 # LoRA
pysdtw                     # SoftDTW GPU
numpy scipy librosa
fluidsynth (system dep)    # for Radif synthesis
lilypond (system dep)      # for Radif engraving
wandb                      # logging
```

`requirements.txt` and `environment.yml` should be committed and reproduce this stack exactly.
