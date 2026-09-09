#!/bin/bash
#SBATCH --job-name=dagrec
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dagrec%a-%A.log
#SBATCH --array=0-3
# DAgger again, with the defect that most likely sank it removed.
#
# Plain DAgger lost 8 points of validation headroom (66.7 against vel_p8's
# 74.8) while its own variance collapsed to a range of 3.5. The targets were
# already soft, so softness was never the issue. The issue is that half its
# training signal comes from states the policy reached while lost, and at those
# states NO candidate is inside the threshold -- yet softmax(-E/tau) still
# concentrates on the least-bad wrong box, so the model is trained to prefer a
# wrong answer. Teacher forcing essentially never visits those states.
#
# recover_w down-weights them. Sweeping 0 (drop entirely) and 0.25 against the
# unweighted run tests whether unwinnable states were the whole problem.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
F=/scratch/pmohseni/omr/candf
M=/scratch/pmohseni/omr/scorer/dagrec; mkdir -p "$M"
case ${SLURM_ARRAY_TASK_ID} in
  0) W=0.0;  S=0 ;;
  1) W=0.0;  S=1 ;;
  2) W=0.25; S=0 ;;
  3) W=0.25; S=1 ;;
esac
T="$M/dagrec_w${W}_s$S.pt"
[ -f "$T" ] && { echo "present"; exit 0; }
echo "##### DAgger + recover_w=$W seed=$S"
python extensions/analysis/train_cand_scorer.py --out "$T" \
    --train "$F/train_c*.npz" --valid "$F/valid.npz" \
    --use_feat --featproj 8 --seed $S \
    --dagger_init /scratch/pmohseni/omr/scorer/grid/vel_p8.pt \
    --dagger_frac 0.5 --dagger_rounds 2 --recover_w $W 2>&1 | grep -vE "^\s*$"
[ -f "$T" ] || { echo "!!!!! no checkpoint"; exit 1; }
