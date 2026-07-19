#!/bin/bash
#SBATCH --job-name=eval-all
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_all-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_all-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

EMB_LORA=/lustre07/scratch/pmohseni/music-alignment/data/MSMD/embeddings_lora
EMB_ALL=$SCRATCH/embeddings_all_tar

latest() { ls "results/$1/"checkpoint_*.pt 2>/dev/null | sort | tail -1; }
run()    { echo; echo "===== $* ====="; "$@" || echo "!! FAILED: $*"; }
# Prefer the early-stopped best checkpoint (results/<model>/best_model.pt) over
# the last periodic checkpoint_NNNNNN.pt -- for models prone to overfitting
# (v3_fullseq / v3_all), the latest checkpoint is not the best one. Falls back
# to latest() for older/in-progress runs that predate best_model.pt.
best_or_latest() {
  local f="results/$1/best_model.pt"
  [ -f "$f" ] && { echo "$f"; return; }
  latest "$1"
}

# --- v1 family (windowed; v1_baseline=no-LoRA, rest=LoRA) ---
for m in v1_baseline v1_dtw v1_lora v1_nce v1_nce2; do
  ck=$(latest "$m"); [ -z "$ck" ] && { echo "skip $m (no ckpt)"; continue; }
  cfg=configs/v1_lora.yaml; [ "$m" = "v1_baseline" ] && cfg=configs/v1_baseline.yaml
  run python -m mymodel.v1_baseline.eval --checkpoint "$ck" --config "$cfg" --split test
done

# --- v2 (cross-attention, live encoders) ---
ck=$(latest v2_nce); [ -n "$ck" ] && \
  run python -m mymodel.v2_crossattn.eval --checkpoint "$ck" --config configs/v2_crossattn.yaml --split test

# --- v2 DTW phase (the real cross-attention+DTW training, warm-started from v2_nce) ---
ck=$(best_or_latest v2_crossattn_dtw); [ -n "$ck" ] && \
  run python -m mymodel.v2_crossattn.eval --checkpoint "$ck" --config configs/v2_crossattn_dtw.yaml --split test

# --- v3 full-seq (cached embeddings) ---
ck=$(best_or_latest v3_fullseq); [ -n "$ck" ] && \
  run python -m mymodel.v3_fullseq.eval --checkpoint "$ck" --emb_root "$EMB_LORA" --split test
ck=$(best_or_latest v3_all);     [ -n "$ck" ] && \
  run python -m mymodel.v3_fullseq.eval --checkpoint "$ck" --emb_root "$EMB_ALL"  --split test

# --- v3 e2e (live encoders) ---
# Prefer the best-val checkpoint from the current retrain (results/v3_e2e_v2),
# then fall back to the older, ambiguously-versioned v3_e2e_long/v3_e2e dirs.
ck=$(best_or_latest v3_e2e_v2)
[ -z "$ck" ] && ck=$(latest v3_e2e_long)
[ -z "$ck" ] && ck=$(latest v3_e2e)
[ -n "$ck" ] && \
  run python -m mymodel.v3_e2e.eval --checkpoint "$ck" --config configs/v3_e2e.yaml --split test

# --- v10 (CPJKU UNet + MERT audio encoder) ---
ck=$(latest v10_mert_unet); [ -n "$ck" ] && \
  run python -m mymodel.v10_mert_unet.eval \
    --checkpoint "$ck" \
    --config configs/v10_mert_unet.yaml \
    --split test \
    --processed data/MSMD/processed \
    --mert_emb  data/MSMD/mert_emb \
    --out_dir   results/v10_mert_unet/eval

# --- comparison table ---
echo; echo "===================== SUMMARY ====================="
for m in v10_mert_unet v3_all v3_fullseq v1_nce v1_baseline v1_dtw v1_nce2 v2_nce v2_crossattn_dtw v3_e2e_v2 v3_e2e_long v3_e2e; do
  f="results/$m/eval/test/summary.json"
  [ -f "$f" ] && python -c "
import json; d=json.load(open('$f'))
print(f\"{'$m':16s} err={d['mean_mean_abs_err_sec']:6.2f}s  \"
      f\"within0.5s={d['mean_pct_within_0.5s']:5.1f}%  \"
      f\"within1.0s={d['mean_pct_within_1.0s']:5.1f}%\")"
done

echo "Job finished at $(date)"
