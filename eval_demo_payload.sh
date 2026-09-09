#!/bin/bash
#SBATCH --job-name=payload
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/payload-%j.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0 ffmpeg 2>/dev/null || module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export OURS_TRAJ=/scratch/pmohseni/omr/traj/velp8_room.traj.npz
which ffmpeg || echo "WARNING: no ffmpeg, the two new excerpts cannot be cut"
python extensions/analysis/build_demo_payload.py
