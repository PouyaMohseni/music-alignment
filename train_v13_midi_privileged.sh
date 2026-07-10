#!/bin/bash
#SBATCH --job-name=v13-midi-privileged
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v13_midi_privileged-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v13_midi_privileged-%j.log

# E2/E3: v13 architecture + MIDI-privileged repeat-aware GT + MIDI->audio
# distillation (train-time only -- eval.py is v13's own, unmodified).

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1

OUT=/scratch/pmohseni/results/v13_midi_privileged
mkdir -p $OUT

RESUME_FLAG=""
if ls $OUT/checkpoint_epoch*.pt 2>/dev/null | grep -q .; then
    LATEST=$(ls $OUT/checkpoint_epoch*.pt | sort | tail -1)
    echo "Resuming from $LATEST"
    RESUME_FLAG="--resume $LATEST"
fi

python -m mymodel.v13_midi_privileged.train \
    --config configs/v13_midi_privileged.yaml \
    train.out_dir=$OUT \
    $RESUME_FLAG

echo ""
echo "Training done. Running test eval (v13's own eval.py, unmodified -- no MIDI at inference)..."
python -m mymodel.v13_mert_unet.eval \
    --checkpoint $OUT/best_model.pt \
    --config     configs/v13_midi_privileged.yaml \
    --split      test \
    --out_dir    $OUT/eval

echo "Job finished at $(date)"
