#!/bin/bash
#SBATCH --job-name=eval-b1a-spatialfilm-cpu
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_b1a_spatialfilm_cpu-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_b1a_spatialfilm_cpu-%j.log

# First-ever eval of B1a_spatial_film (SPADE-inspired spatially-varying
# gamma/beta, on top of B1a's frozen-MERT audio swap). CPU-only: MERT
# embeddings are precomputed on disk, only the small U-Net + spatial-FiLM
# module run live.

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

CKPT_DIR=$(find results/cb_ta_ext/B1a_spatial_film/params -maxdepth 1 -name "*_B1a_spatial_film" -type d | sort | tail -1)
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

echo "=== Native eval_model.py: B1a_spatial_film on msmd_test (CPU) ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_eval_native_b1a_spatial_film.py \
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
