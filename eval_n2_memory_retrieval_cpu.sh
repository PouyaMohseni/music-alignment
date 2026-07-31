#!/bin/bash
#SBATCH --job-name=eval-n2-memret-cpu
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_n2_memory_retrieval_cpu-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_n2_memory_retrieval_cpu-%j.log

# Eval of N2 (gated lag-aware memory retrieval).
# CPU-only: MERT embeddings are precomputed on disk and the new temporal
# module sits in the lightweight per-frame conditioning path (not inside
# the per-decoder-block loop that made MERT+DINOv2-crossattn unusable on
# CPU), so only the small U-Net runs live.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

CKPT_DIR=$(find results/cb_ta_ext/N2_memory_retrieval/params -maxdepth 1 -name "*_N2_memory_retrieval" -type d | sort | tail -1)
echo "Using checkpoint dir: $CKPT_DIR"
CKPT="$(readlink -f "$CKPT_DIR/best_model.pt")"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MERT_TEST_EMB_ROOT=/scratch/pmohseni/mert_emb_zenodo/msmd_test

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo "=== Native eval_model.py: N2_memory_retrieval on msmd_test (CPU) ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_eval_native_n2_memory_retrieval.py \
    --param_path  "$CKPT" \
    --test_dir    ../data/msmd/msmd_test \
    --config      configs/msmd.yaml \
    --scale_factor 3 \
    --batch_size  1 \
    --seq_len     128 \
    --eval_onsets \
    --piecewise_stats

echo ""
echo "Job finished at $(date)"
