#!/bin/bash
#SBATCH --job-name=eval-f3-4model
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_f3_4model-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_f3_4model-%j.log

# F3 variant B: 4-member ensemble adding v13_midi_privileged (E2/E3's
# checkpoint, only ~epoch 5-6 into training, interim solo eval 49.5%
# pct@0.5s) down-weighted to 0.1 alongside v13/v14/v15 at 0.3 each. Shares
# identical strip geometry (h_strip=128, w_scale=4, fps=20) so it's a valid
# ensemble member despite being far less converged than the other three.
# Decode both ways (original, offline_dtw) in one pass.
#
# NOTE: this uses whatever v13_midi_privileged/best_model.pt is on disk
# RIGHT NOW (from the first training attempt, timed out at 24h). Job
# 65216961 (resumed training) is still queued -- if/when it produces a
# better checkpoint, this eval should be re-run against it.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15,v13_midi \
    --weights 0.3,0.3,0.3,0.1 \
    --decoders original,offline_dtw \
    --split test \
    --out_dir results/f3_ensemble_decode/v13+v14+v15+v13_midi

echo "Job finished at $(date)"
