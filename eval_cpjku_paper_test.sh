#!/bin/bash
#SBATCH --job-name=cpjku-paper-eval
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper_eval-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper_eval-%j.log

# Reproduce CB_TA paper results (Table 2 + Table 3) exactly as in Henkel et al. ISMIR 2020.
# Uses their unmodified eval_model.py, their msmd_test data, synthesized audio via FluidSynth.
#
# Their exact README eval commands:
#   python eval_model.py --param_path ../models/CB_TA/best_model.pt \
#     --test_dir ../data/msmd/msmd_test --config configs/msmd.yaml
#   python eval_model.py ... --eval_onsets   (for Table 3)
#
# Pass --param_path to evaluate a custom trained model instead of their pretrained weights:
#   sbatch eval_cpjku_paper_test.sh /path/to/your/best_model.pt
#
# Prerequisites: run setup_cpjku_paper_login.sh on login node first.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

git submodule update --init third_party/cpjku_unet
cd third_party/cpjku_unet && git checkout ismir-2020 && cd ../..

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

# FluidSynth shared library
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}

# Prevent BLAS-fork deadlock
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet

# Default to their pretrained CB_TA; override with $1 to eval a custom model
PARAM_PATH="${1:-$REPO/models/CB_TA/best_model.pt}"

OUT=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper
mkdir -p "$OUT"

cd "$REPO/audio_conditioned_unet"

echo "=== Eval CB_TA: msmd_test, synthesized audio (Table 2 — pixel/F1 metrics) ==="
python eval_model.py \
    --param_path  "$PARAM_PATH" \
    --test_dir    ../data/msmd/msmd_test \
    --config      configs/msmd.yaml \
    --scale_factor 3 \
    --batch_size  1 \
    --seq_len     128 \
    --piecewise_stats \
    2>&1 | tee "$OUT/eval_CB_TA_test_f1.log"

echo ""
echo "=== Eval CB_TA: msmd_test, synthesized audio (Table 3 — onset timing) ==="
python eval_model.py \
    --param_path  "$PARAM_PATH" \
    --test_dir    ../data/msmd/msmd_test \
    --config      configs/msmd.yaml \
    --scale_factor 3 \
    --batch_size  1 \
    --seq_len     128 \
    --eval_onsets \
    --piecewise_stats \
    2>&1 | tee "$OUT/eval_CB_TA_test_onsets.log"

echo ""
echo "Job finished at $(date)"
