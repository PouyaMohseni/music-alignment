#!/bin/bash
#SBATCH --job-name=eval-v13-f1
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v13_f1-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v13_f1-%j.log

# F1 first checkpoint eval: v13's own eval.py, unmodified -- F1's
# checkpoint state_dict is byte-identical to v13's original format (MidiEncoder
# and the soft-DTW consistency term are training-only, never attached to
# `network`), so this needs zero special-casing. First best_model.pt just
# appeared -- interim read, training is still in progress.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1

python -m mymodel.v13_mert_unet.eval \
    --checkpoint /scratch/pmohseni/results/v13_f1_combined/best_model.pt \
    --config     configs/v13_f1_combined.yaml \
    --split      test \
    --out_dir    /scratch/pmohseni/results/v13_f1_combined/eval

echo "Job finished at $(date)"
