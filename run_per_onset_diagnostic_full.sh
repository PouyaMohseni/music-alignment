#!/bin/bash
#SBATCH --job-name=per-onset-diag-full
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/per_onset_diag_full-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/per_onset_diag_full-%j.log

# Full 94-piece version of the per-onset error diagnostic -- the 14-piece
# sample (WORST_SHARED_PIECES) only covers already-known-worst pieces, so it
# can't show contrast against "normal" pieces where the repeat-ambiguity /
# sparse-density correlations might look different. Runs on CPU (much
# shorter queue wait than the saturated A100 partition this session);
# observed throughput on the 14-piece sample was ~5 min/piece, so 94 pieces
# extrapolates to ~7-8 hours -- 16h budget gives real margin. Saves
# incrementally per piece, so a timeout still leaves a large usable sample.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8

python scripts/per_onset_error_diagnostic.py --all_pieces --split test \
    --out results/per_onset_diagnostic_full.json

echo "Job finished at $(date)"
