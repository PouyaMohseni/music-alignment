#!/bin/bash
#SBATCH --job-name=eval-mert-dinov2-xattn-gpu
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_mert_dinov2_xattn_gpu-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_mert_dinov2_xattn_gpu-%j.log

# First-ever eval of MERT_dinov2_cross_attention. The CPU-only attempt
# (eval_mert_dinov2_crossattn_cpu.sh, job 66522168) was killed after 11/125
# pieces in ~4h: TokenCrossAttentionFiLM runs cross-attention over the FULL
# raw DINOv2 token grid at every frame of every decoder block, and
# eval_model.py evaluates whole pieces at once (unlike training's short BPTT
# chunks) -- on CPU this blew up to 30+ min/piece for longer pieces. Needs
# real GPU throughput to finish in reasonable time.

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

CKPT_DIR=$(find results/cb_ta_ext/MERT_dinov2_cross_attention/params -maxdepth 1 -name "*_MERT_dinov2_cross_attention" -type d | sort | tail -1)
echo "Using checkpoint dir: $CKPT_DIR"
CKPT="$(readlink -f "$CKPT_DIR/best_model.pt")"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export DINOV2_TILED_ROOT=/scratch/pmohseni/dinov2_emb_tiled_native
export MERT_TEST_EMB_ROOT=/scratch/pmohseni/mert_emb_zenodo/msmd_test

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo "=== Native eval_model.py: MERT_dinov2_cross_attention on msmd_test (GPU) ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_eval_native_mert_dinov2_crossattn.py \
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
