#!/bin/bash
#SBATCH --job-name=eval-vdinov2-cpu
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_vdinov2_cpu-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_vdinov2_cpu-%j.log

# First-ever eval of V_dinov2_full_encoder (genuine visual-architecture
# change: DINOv2 tile-grid neck replaces CB_TA's from-scratch conv encoder,
# plain CBEncoder audio). CPU-only: no live foundation-model forward pass is
# needed at eval time -- DINOv2 tile grids are precomputed on disk
# (DINOV2_TILED_ROOT), so only the small U-Net + LSTM run live, same as the
# other CPU-only eval conversions this session.

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

CKPT_DIR=$(find results/cb_ta_ext/V_dinov2_full_encoder/params -maxdepth 1 -name "*_V_dinov2_full_encoder" -type d | sort | tail -1)
echo "Using checkpoint dir: $CKPT_DIR"
CKPT="$(readlink -f "$CKPT_DIR/best_model.pt")"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DINOV2_TILED_ROOT=/scratch/pmohseni/dinov2_emb_tiled_native

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo "=== Native eval_model.py: V_dinov2_full_encoder on msmd_test (CPU) ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_eval_native_dinov2_full_encoder.py \
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
