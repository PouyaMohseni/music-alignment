#!/bin/bash
#SBATCH --job-name=eval-mert-dinov2
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_mert_dinov2-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_mert_dinov2-%j.log

# Interim spot-check of MERT_DINOv2_bottleneck -- not a final result. Uses
# the MERT+DINOv2-aware native eval (extensions/hooks/run_eval_native_mert_dinov2.py)
# since this checkpoint's audio_encoder is MERTDINOv2Projector.

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

CKPT_DIR=$(find results/cb_ta_ext/MERT_DINOv2_bottleneck/params -maxdepth 1 -name "*_MERT_DINOv2_bottleneck" -type d | sort | tail -1)
echo "Using checkpoint dir: $CKPT_DIR"
if [ -z "$CKPT_DIR" ]; then
    echo "No checkpoint dir found yet. Exiting."
    exit 0
fi
CKPT="$(readlink -f "$CKPT_DIR/best_model.pt")"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MERT_TEST_EMB_ROOT=/scratch/pmohseni/mert_emb_zenodo/msmd_test
export DINOV2_ROOT=/scratch/pmohseni/dinov2_emb_native

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo "=== Native MERT+DINOv2-aware eval_model.py: MERT_DINOv2_bottleneck on msmd_test ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_eval_native_mert_dinov2.py \
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
