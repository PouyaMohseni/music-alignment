#!/bin/bash
#SBATCH --job-name=cpjku-paper-rp
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper_rp-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper_rp-%j.log

# Reproduce their paper results (Table 3) on real piano performances.
# Runs their unmodified eval_model.py from inside their repo with their exact paths.
# real_perf=True: loads WAVs directly — no FluidSynth needed.
# 25 pieces, two microphone conditions (di-left and room).
#
# Their README eval command:
#   python eval_model.py --param_path ../models/CB_TA/best_model.pt \
#     --test_dir ../data/msmd/msmd_test --config configs/msmd.yaml --eval_onsets
# (msmd_test needs FluidSynth; msmd_real_performances uses WAVs — same CB_TA model)

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

# Ensure submodule is on the right branch
git submodule update --init third_party/cpjku_unet
cd third_party/cpjku_unet && git checkout ismir-2020 && cd ../..

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

# Prevent BLAS-fork deadlock: eval_model.py uses Pool(8).map for data loading
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
MODEL=$REPO/models/CB_TA/best_model.pt
TEST_DIR=$REPO/data/msmd/msmd_real_performances
SPLIT=$TEST_DIR/rp_split.yaml
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper_rp
mkdir -p "$OUT"

# Run from inside their audio_conditioned_unet/ dir so config sf_path resolves correctly
cd "$REPO/audio_conditioned_unet"

echo "=== CB_TA eval: real performances (di-left microphone) ==="
python eval_model.py \
    --param_path  "$MODEL" \
    --test_dir    "$TEST_DIR" \
    --config      configs/msmd_rp_di.yaml \
    --split_file  "$SPLIT" \
    --scale_factor 3 \
    --batch_size  1 \
    --seq_len     128 \
    --eval_onsets \
    --piecewise_stats \
    2>&1 | tee "$OUT/eval_CB_TA_rp_di.log"

echo ""
echo "=== CB_TA eval: real performances (room microphone) ==="
python eval_model.py \
    --param_path  "$MODEL" \
    --test_dir    "$TEST_DIR" \
    --config      configs/msmd_rp_room.yaml \
    --split_file  "$SPLIT" \
    --scale_factor 3 \
    --batch_size  1 \
    --seq_len     128 \
    --eval_onsets \
    --piecewise_stats \
    2>&1 | tee "$OUT/eval_CB_TA_rp_room.log"

echo ""
echo "Job finished at $(date)"
