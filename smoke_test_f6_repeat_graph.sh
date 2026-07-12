#!/bin/bash
#SBATCH --job-name=smoke-f6-graph
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f6_graph-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/smoke_f6_graph-%j.log

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# 60 pieces (indices 0-59) -- verified this covers the repeat-ambiguous
# cluster pieces (Czerny idx 38, Satie gymnopedie_1 idx 51, Schumann choral
# idx 54, chanson-populaire idx 57) so the graph actually gets exercised.
python -m mymodel.f3_ensemble_decode.eval \
    --models v13,v14,v15 --decoders original,hybrid_snap,repeat_graph_snap \
    --snap_frac 0.2 --repeat_graph_radius 8 --repeat_graph_window 5 \
    --split test --limit 60 --out_dir results/f3_smoke/repeat_graph

echo "Job finished at $(date)"
