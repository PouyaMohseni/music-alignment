#!/bin/bash
#SBATCH --job-name=noz-only
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/noz_only-%j.log
# The missing control. z_only (95.60) is compared against ir_only (95.12), but
# those were fitted on different dumps. Same data, z withheld, is the only
# comparison that isolates z.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
python extensions/analysis/train_cand_scorer.py \
    --out /scratch/pmohseni/omr/scorer/noz_only.pt \
    --train "/scratch/pmohseni/omr/candz_ir/train_c*.npz" \
    --valid "/scratch/pmohseni/omr/candz_ir/valid.npz" 2>&1 | grep -vE "^\s*$"
