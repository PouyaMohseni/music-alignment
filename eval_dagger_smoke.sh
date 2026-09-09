#!/bin/bash
#SBATCH --job-name=dagsmoke
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dagsmoke-%j.log
# Did the DAgger path actually run? The training jobs pipe through `tail -25`,
# which cut the "DAgger: N visited states" line printed near the start, so the
# mechanism at the centre of the experiment is currently unverified. One epoch
# on one chunk, with output NOT piped, settles it.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python extensions/analysis/train_cand_scorer.py \
    --out /scratch/pmohseni/omr/scorer/dagger/_smoke.pt \
    --train /scratch/pmohseni/omr/candf/train_c0.npz \
    --valid /scratch/pmohseni/omr/candf/valid.npz \
    --use_feat --featproj 8 --seed 99 --epochs 1 \
    --dagger_init /scratch/pmohseni/omr/scorer/grid/vel_p8.pt \
    --dagger_frac 0.5 --dagger_rounds 1
