#!/bin/bash
#SBATCH --job-name=g1-gnn-full
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/g1_gnn_full-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/g1_gnn_full-%j.log

# Gated by --dependency=afterany:<smoke-g1-gnn jobid> at submission time.
# Auto-decides whether the full 94-piece run is worth running by checking
# the smoke test's own result first -- no human needs to inspect it.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

SMOKE_LOG="$1"
if [ -z "$SMOKE_LOG" ]; then
    echo "ERROR: must pass the smoke-g1-gnn log path as \$1" >&2
    exit 1
fi

python scripts/check_smoke_and_decide.py --log "$SMOKE_LOG" \
    --baseline original --candidate gnn_repeat_snap --min_delta -1.0
if [ $? -ne 0 ]; then
    echo "Skipping full 94-piece G1 run based on smoke test result above."
    exit 0
fi

python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15 --decoders original,hybrid_snap,gnn_repeat_snap \
    --snap_frac 0.2 --repeat_graph_radius 8 --repeat_graph_window 5 \
    --gnn_checkpoint /scratch/pmohseni/results/g1_repeat_gnn/best_model.pt --gnn_sim_threshold 0.85 \
    --split test --out_dir results/f3_ensemble_decode/gnn_repeat_full

echo "Job finished at $(date)"
