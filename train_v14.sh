#!/bin/bash
#SBATCH --job-name=music-v14
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v14_mert_bilstm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v14_mert_bilstm-%j.log

# v14: MERT 8-frame window → BiLSTM(768→256,bidir) → Linear(512→32) → ConditionalUNet

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1

OUT=/scratch/pmohseni/results/v14_mert_bilstm
mkdir -p $OUT

RESUME_FLAG=""
if ls $OUT/checkpoint_epoch*.pt 2>/dev/null | grep -q .; then
    LATEST=$(ls $OUT/checkpoint_epoch*.pt | sort | tail -1)
    echo "Resuming from $LATEST"
    RESUME_FLAG="--resume $LATEST"
fi

python -m mymodel.v13_mert_unet.train \
    --config configs/v14_mert_bilstm.yaml \
    train.out_dir=$OUT \
    $RESUME_FLAG

echo "Job finished at $(date)"
