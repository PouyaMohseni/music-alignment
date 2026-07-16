#!/bin/bash
#SBATCH --job-name=eval-b5
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_b5-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_b5-%j.log

# Interim spot-check of B5_dense_contrastive_aux on its current best_model.pt
# -- training is still in progress as of submission, so this is not a final
# result.
#
# Switched from mymodel.cpjku_adapter.eval_official to CPJKU's own native
# eval_model.py (2026-07-16): eval_official.py fed models a wide strip-format
# score image (our own strip.png-derived NPZ, e.g. 224x2968) instead of the
# native portrait per-page format B5 was actually trained on (e.g. 1181x835,
# third_party/cpjku_unet/data/msmd/msmd_test/score/<pid>_page_N.npz) -- a
# visual-domain mismatch far bigger than the spectrogram approximation
# already fixed (that fix alone only moved the frozen official checkpoint's
# score 15.1%->18.6%, vs 85.1% through this native path). No convert.py
# involved now -- this is exactly the pipeline already proven to reproduce
# the paper's ~85% number for the frozen official checkpoint.

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

CKPT_DIR=$(find results/cb_ta_ext/B5_dense_contrastive_aux/params -maxdepth 1 -name "*_B5_dense_contrastive_aux" -type d | sort | tail -1)
echo "Using checkpoint dir: $CKPT_DIR"
CKPT="$(readlink -f "$CKPT_DIR/best_model.pt")"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo "=== Native eval_model.py: B5_dense_contrastive_aux on msmd_test ==="
python eval_model.py \
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
