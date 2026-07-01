#!/bin/bash
#SBATCH --job-name=cpjku-paper-train
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper_train-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper_train-%j.log

# Train CB_TA model exactly as in Henkel et al. ISMIR 2020.
# Uses their unmodified train_model.py, their msmd dataset, their msmd_aug.yaml config.
#
# Exact equivalent of their run_experiments.sh case 1:
#   python train_rnn.py --film_layers 2 3 4 5 6 7 8 \
#     --train_set ../data/msmd/msmd_train --val_set ../data/msmd/msmd_valid \
#     --use_lstm --augment --config configs/msmd_aug.yaml \
#     --audio_encoder CBEncoder --tag CB_TA
# (train_rnn.py is the same as train_model.py — just a rename)
#
# Prerequisites: run setup_cpjku_paper_login.sh on login node first.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

# Ensure submodule is on right branch
git submodule update --init third_party/cpjku_unet
cd third_party/cpjku_unet && git checkout ismir-2020 && cd ../..

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

# FluidSynth: shared library + CLI binary (midi_to_spec_otf uses subprocess "fluidsynth -F ...")
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}

# Prevent BLAS-fork deadlock (Pool.map forks workers that inherit BLAS thread locks)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper/CB_TA
mkdir -p "$OUT/runs" "$OUT/params"

# Run from inside their audio_conditioned_unet/ directory so relative paths
# (configs/, sound_fonts/) resolve exactly as intended
cd "$REPO/audio_conditioned_unet"

# Performance MIDIs generated to scratch (avoids /project file-count quota).
# Score NPZs stay on project; scratch dir has a symlink to them.
SCRATCH_TRAIN=/scratch/pmohseni/msmd_train_full

if [ ! -d "$SCRATCH_TRAIN/performance" ]; then
    echo "ERROR: $SCRATCH_TRAIN/performance not found. Run gen_perf_midis.sh first." >&2
    exit 1
fi
NPERF=$(ls "$SCRATCH_TRAIN/performance/"*.mid 2>/dev/null | wc -l)
echo "Performance MIDIs on scratch: $NPERF / 6615"

echo "=== Train CB_TA (paper-faithful: no split_file, all 945 pages) ==="
echo "train:  $SCRATCH_TRAIN  (945 pages x 7 tempi = 6615 pairs)"
echo "val:    ../data/msmd/msmd_valid  (28 pieces, all complete)"
echo "config: msmd_aug.yaml (tempo augmentation: 500/750/950/1000/1050/1250/1500)"
echo "encoder: CBEncoder + LSTM + FiLM layers 2-8"
echo ""

python train_model.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$SCRATCH_TRAIN" \
    --val_set   ../data/msmd/msmd_valid \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder CBEncoder \
    --tag CB_TA

echo ""
echo "Training finished at $(date)"
echo "Best model: $OUT/params/CB_TA/best_model.pt"
