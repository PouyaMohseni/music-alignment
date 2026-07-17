#!/bin/bash
#SBATCH --job-name=eval-b1a-native
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_b1a_native-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_b1a_native-%j.log

# B1a's FIRST trustworthy eval: previous runs (e.g. eval_b1a-65370662.log,
# 37.7% pct@0.5s) went through mymodel.cpjku_adapter.eval_official_mert,
# which has the same strip-vs-native-page visual-domain-mismatch bug found
# 2026-07-16 in the plain (non-MERT) eval_official.py path -- B1a's
# checkpoint was trained on native pages via CPJKU's own train_model.py, so
# it must be evaluated the same way. This uses CPJKU's own unmodified
# eval_model.py against the real native msmd_test split, with MERT
# embeddings substituted for the live spectrogram (extensions/hooks/
# run_eval_native_mert.py, same mert_patch.py used for training), matching
# how B1a was actually trained. No convert.py, no strip conversion.

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

CKPT_DIR=$(find results/cb_ta_ext/B1a_mert_swap/params -maxdepth 1 -name "*_B1a_mert_swap" -type d | sort | tail -1)
echo "Using checkpoint dir: $CKPT_DIR"
CKPT="$(readlink -f "$CKPT_DIR/best_model.pt")"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MERT_TEST_EMB_ROOT=/scratch/pmohseni/mert_emb_zenodo/msmd_test

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

echo "=== Native MERT-aware eval_model.py: B1a_mert_swap on msmd_test ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_eval_native_mert.py \
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
