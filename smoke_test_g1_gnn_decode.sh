#!/bin/bash
#SBATCH --job-name=smoke-g1-gnn
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/smoke_g1_gnn-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/smoke_g1_gnn-%j.log

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Same 60-piece slice as F6's smoke test (covers the repeat-ambiguous cluster:
# Czerny idx 38, Satie gymnopedie_1 idx 51, Schumann choral idx 54,
# chanson-populaire idx 57), so this is directly comparable to F6's result.
python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15 --decoders original,hybrid_snap,gnn_repeat_snap \
    --snap_frac 0.2 --repeat_graph_radius 8 --repeat_graph_window 5 \
    --gnn_checkpoint /scratch/pmohseni/results/g1_repeat_gnn/best_model.pt --gnn_sim_threshold 0.85 \
    --split test --limit 60 --out_dir results/f3_smoke/gnn_repeat

echo "Job finished at $(date)"
