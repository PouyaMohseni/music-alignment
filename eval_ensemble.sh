#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/ensemble-%j.log
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "########## ROOM ##########"
python extensions/analysis/ensemble_rollout.py --dump /scratch/pmohseni/omr/candf/room.npz
echo ""; echo "########## HELD-OUT snr12 (no shared pieces, real power) ##########"
python extensions/analysis/ensemble_rollout.py --dump /scratch/pmohseni/omr/candhv/valid_snr12.npz
