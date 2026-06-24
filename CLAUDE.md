# CLAUDE.md — music-alignment project context

## Repo purpose
Score-following / music alignment: given a live audio performance and a sheet-music
strip image, predict the current position (x pixel on the strip) frame-by-frame.
Dataset: MSMD (Mutopia Score MIDI Dataset), preprocessed into `data/MSMD/processed/`.

---

## Cluster environment (Narval / Alliance)

**Project root on cluster:**
```
/project/def-ichiro/pmohseni/music-alignment
```

**Python venv:** `.venv/` (created by `setup.sh`).  Always activate with:
```bash
source .venv/bin/activate
```

**OpenCV on Alliance cluster:**  `pip install opencv-python-headless` installs a
dummy wheel that fails at import.  Instead, load the system module **before** the venv:
```bash
module load gcc opencv
source .venv/bin/activate
```
(See `eval_cpjku_official.sh` for the canonical pattern.)

**SLURM account:** `def-ichiro`  
**GPU constraint for A100:** `#SBATCH --constraint=a100`

**Never** add `Co-Authored-By: Claude Sonnet 4.6` to commits — user does not want it.

**Always push immediately after every file change** — user runs `git pull` on the
cluster to get updates, so stale local-only edits block them.

---

## Key third-party code

`third_party/cpjku_unet/` — CPJKU ConditionalUNet (Henkel et al. ISMIR 2020).
Branch to use: `ismir-2020`.

```bash
git submodule update --init third_party/cpjku_unet
cd third_party/cpjku_unet && git checkout ismir-2020 && cd ../..
```

Pre-trained model: `third_party/cpjku_unet/models/CB_TA/best_model.pt`
Net config:        `third_party/cpjku_unet/models/CB_TA/net_config.json`

**CBEncoder** (their audio encoder):
- Input shape: `(seq_len, bs, c=1, h=78, w=40)` — 78 frequency bands, 40 frames
- 4 × MaxPool2d: 78→39→19→9→4 height, 40→20→10→5→2 width → Linear(768, 32)
- `n_input_frames = 40`

---

## madmom compatibility (`mymodel/cpjku_adapter/madmom_compat.py`)

madmom does not install on Python ≥3.11 / NumPy ≥1.24.  We stub it out and replace
with librosa.  **Always call `madmom_compat.patch()` before importing any
`audio_conditioned_unet.*` module.**

```python
from mymodel.cpjku_adapter import madmom_compat
madmom_compat.patch()
# now safe to: from audio_conditioned_unet.network import ConditionalUNet
```

`patch()` stubs cv2 (if missing) and all madmom sub-modules, then monkey-patches
`audio_conditioned_unet.utils.wav_to_spec_otf`.

**Worker processes** (multiprocessing pool.map) do **not** inherit `sys.modules` stubs
from the parent.  Any function run inside a worker must re-apply stubs itself.
See `_patched_load_piece` in `eval_official.py` — it re-stubs at the top.

---

## CPJKU eval pipeline (`mymodel/cpjku_adapter/`)

### `convert.py`
Converts `data/MSMD/processed/` → `data/MSMD/cpjku_fmt/` (NPZ per piece + WAV).
NPZ contains: `sheet` (H×W uint8), `coords` (N×2 float32: [y, x]),
`coord2onset`, `onset_frames` (N int64).

### `eval_official.py`
Runs CB_TA on our MSMD test split.  Key design decisions:

1. **`_patched_load_piece`** is at **module level** (not inside `main()`) so
   `pool.map` can pickle it.

2. **interpol_fnc must return 3 values** `[y, x, height]`.  Their dataset code does:
   ```python
   true_position, height = result[:-1], result[-1]
   ```
   We pad coords to `(N, 3)` by appending `H_strip // 2` as the height column.

3. **Spectrogram**: `_wav_to_spec_librosa` at module level — 78-band log-mel,
   60–6000 Hz, fps=20.  Defined at module level so workers can use it without
   importing from `audio_conditioned_unet.utils` (which tries to load real madmom).

4. **`add_per_staff`**: `[staff_coords, np.array([0]*n_staves)]`.  For our
   single-staff strips this is always `[[H//2], [0]]`.

5. **Use `ScoreAudioDataset` directly** in `iterate_dataset` — do NOT wrap in
   `NonSequentialDatasetWrapper`.  The wrapper strips `add_per_staff` and
   `interpol_c2o` from the piece dict; `calculate_batch_stats` needs both.
   (Their own `eval_model.py` passes the raw dataset.)

6. **`batch_size=1`**: strips have variable widths; `prepare_batch` concatenates
   along axis=1 and fails with mismatched widths.

### `madmom_compat.py`
Note: `_librosa_spec` in this file still uses `n_mels=12` (historical, for training
compatibility).  `eval_official.py` bypasses this and calls `_wav_to_spec_librosa`
(78 bands) directly.

---

## v11 — our CPJKU faithful reproduction

**Files:**
- `mymodel/v11_cpjku_fullstrip/data.py` — `FullStripDataset`, `make_gt_mask`
- `mymodel/v11_cpjku_fullstrip/train.py` — BPTT training loop
- `mymodel/v11_cpjku_fullstrip/eval.py` — inference
- `configs/v11_cpjku_fullstrip.yaml` — hyperparameters
- `train_v11.sh` — SLURM job (A100, 64GB, 24h)

**Key choices matching their paper:**
- Full strip as score per frame (GT at TRUE position → model must use audio)
- `batch_size=1` (variable strip widths)
- BPTT `seq_len=16`, LSTM state maintained across chunks, reset per piece
- `dice smoothing=0.` (their exact `iterate_dataset` call)
- 100 epochs max, ReduceLROnPlateau patience=5, early stop at patience×2=10
- Spectrogram normalisation from training set before epoch 1

**Score tensor shape going into the network:**
```
score: (sl, bs=1, c=1, H, W_sc)   # full strip, expanded per BPTT chunk
perf:  (sl, bs=1, c=1, n_mels=78, n_frames=40)
seg:   (sl, c=1, H, W_sc)         # output segmentation
```

**w_scale=4**: strip width is divided by 4 before training (memory efficiency).

---

## Data layout

```
data/MSMD/
  processed/          # our preprocessed format (strips, specs, annotations)
    splits.json       # train/val/test piece lists
    <piece_id>/
      strip.png
      performance_*.npz
  cpjku_fmt/          # converted for CPJKU eval
    score/<piece>.npz
    performance/<piece>.wav
    split_test.yaml
```

**Authoritative READMEs** (read before touching data/strip/annotation/training-config):
- `README_main.md`
- `README_dataset.md`

---

## Model version history (brief)

| Version | Approach | Best metric |
|---------|----------|-------------|
| v3_all  | full-seq, all-perf | 5.35s median |
| v8      | Henkel repro (crop tracking) | — |
| v9_cpjku | CPJKU UNet, crop | — |
| v11     | CPJKU UNet, **full strip**, BPTT | running |
| CB_TA   | CPJKU pre-trained (our data) | TBD from eval |

---

## Common SLURM commands

```bash
squeue -u pmohseni            # check running jobs
sbatch train_v11.sh           # submit training
sbatch eval_cpjku_official.sh # submit CPJKU eval
tail -f results/<job>.log     # follow log
scancel <jobid>               # cancel job
```

## Active jobs (as of 2026-06-23)
- Job 63808967: v11 training (started ~13h ago, 24h limit)
- CPJKU eval: to be resubmitted after `git pull`
