#!/bin/bash
#SBATCH --job-name=dinomap
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dinomap%a-%A.log
# DINOv2-base feature maps for every page the scorers train and test on:
# msmd_train (train shards and the held-out split), msmd_valid, and the room
# test set. dino_annotate.sh then reads them at each candidate.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source /project/def-ichiro/pmohseni/music-alignment/.venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
D=/scratch/pmohseni/datasets/cyolo_data/msmd
python extensions/analysis/dino_maps.py --out /scratch/pmohseni/omr/dino_maps \
    --shard "${SLURM_ARRAY_TASK_ID}" --num_shards 4 \
    --npz "$D"/msmd_train/*.npz "$D"/msmd_valid/*.npz "$D"/msmd_rp/*_room.npz
echo "Job finished at $(date)"
