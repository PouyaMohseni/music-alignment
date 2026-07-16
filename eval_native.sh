#!/bin/bash
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Generic native-format eval for any cpjku_adapter-family checkpoint
# (b2/b3/b4/b5/b6/c2/cpjku_aug, or the frozen official CB_TA). Fixes the
# visual-domain mismatch found 2026-07-16: eval_official.py fed models a
# wide strip-format score image (our own strip.png-derived NPZ, e.g.
# 224x2968) instead of the native portrait per-page format every one of
# these checkpoints was actually trained on (e.g. 1181x835,
# third_party/cpjku_unet/data/msmd/msmd_test/score/<pid>_page_N.npz) -- a
# far bigger distribution shift than the spectrogram approximation (whose
# fix alone only moved the frozen official checkpoint's score 15.1%->18.6%,
# vs 85.1% through this native path). No convert.py involved, no strip
# conversion -- this is exactly CPJKU's own unmodified eval_model.py against
# their real native msmd_test split, the same invocation already proven to
# reproduce the paper's ~85% number for the frozen official checkpoint.
#
# Does NOT work for MERT-swap checkpoints (B1a) -- those need MERT
# embeddings per native page, which is a separate, not-yet-built path
# (eval_official_mert.py's --mert_root only has embeddings for our strip
# format, not native pages).
#
# Usage: sbatch -J eval-<tag>-native -o results/eval_<tag>_native-%j.log \
#               -e results/eval_<tag>_native-%j.log \
#               eval_native.sh </absolute/path/to/best_model.pt>

set -euo pipefail
CKPT="$(readlink -f "$1")"
echo "Job started on $(hostname) at $(date)"
echo "Checkpoint: $CKPT"

cd /project/def-ichiro/pmohseni/music-alignment

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo "=== Native eval_model.py: msmd_test, onset timing ==="
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
