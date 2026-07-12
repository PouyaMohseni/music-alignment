#!/bin/bash
#SBATCH --job-name=g1-repeat-gnn
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/g1_repeat_gnn-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/g1_repeat_gnn-%j.log

# G1: a real GNN (learned weights, contrastive training), not a heuristic.
# Score-only, tiny graphs (100-2000 nodes/piece) -- CPU-only, no GPU needed.
# Trains on ~350 train-split pieces to embed notes such that D2's existing
# repeat-detection heuristic's matches land close together in embedding
# space, aiming to generalize beyond its rigid exact-interval-match rule.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export OMP_NUM_THREADS=4

python -m mymodel.g1_repeat_gnn.train --epochs 40 --pieces_per_step 8 \
    --out_dir /scratch/pmohseni/results/g1_repeat_gnn

echo "Job finished at $(date)"
