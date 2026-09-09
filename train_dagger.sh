#!/bin/bash
#SBATCH --job-name=dagger
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dagger%a-%A.log
#SBATCH --array=0-3
# Train on the states the policy actually visits.
#
# The selector's input includes its OWN previous position, so teacher forcing
# fits it on a history it never encounters. The cost is measurable and large:
# removing the previous-position noise -- the crude stand-in for that history --
# drops validation rollout from 95.33 to 92.38, below the detector's own
# argmax. Exposure bias is the dominant term, not a detail.
#
# Laplace noise guesses the shape of the model's mistakes. DAgger measures it:
# roll the policy out, keep the states it reaches, relabel each with the oracle
# choice (known -- it is the candidate nearest ground truth, already in the
# dumps) and refit on the mixture. Two rounds, so the second re-rolls with the
# model being trained rather than its ancestor.
#
# This targets the failure we actually have: 71% of lost onsets sit in
# contiguous episodes, i.e. error propagation, which is precisely what a
# policy trained only on correct histories cannot handle.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer/dagger; mkdir -p "$M"
S=${SLURM_ARRAY_TASK_ID}
T="$M/dagger_s$S.pt"
[ -f "$T" ] && { echo "seed $S present"; exit 0; }
echo "##### DAgger seed $S"
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj 8 --seed $S \
    --dagger_init /scratch/pmohseni/omr/scorer/grid/vel_p8.pt \
    --dagger_frac 0.5 --dagger_rounds 2 2>&1 | tail -25
# `| tail` hid the traceback and the trailing echo returned 0, so the array
# reported COMPLETED in 106 seconds having written nothing. Check the artefact.
[ -f "$T" ] || { echo "!!!!! seed $S produced no checkpoint"; exit 1; }
echo "##### seed $S done"
