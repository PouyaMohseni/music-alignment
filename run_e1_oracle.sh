#!/bin/bash
#SBATCH --job-name=oracle-e1
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/oracle_e1/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/oracle_e1/slurm-%j.log

# E1 — oracle pianoroll alignment ceiling. Training-free, CPU-only, ships NOTHING.
# Sweeps fps so we read BOTH levers at once:
#   - pitch ceiling      : how well does perfect pitch align on the 1-D strip?
#   - resolution ceiling : how much does finer audio frame-rate help @0.5s?
# Decision: if @0.5s jumps high -> pitch is the lever (build E0/E2). If it stays
# ~34% even with perfect pitch -> resolution/framing is the floor (build E3 first).

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate

echo ""
echo "=== E1 oracle pianoroll DTW — test split ==="
echo "fps   err(s)   @0.5s   @0.25s   @0.1s   recall@1"
echo "----  -------  ------  -------  ------  --------"

for FPS in 10 20 40; do
  python -m mymodel.diagnostics.oracle_e1 \
    --split test --fps $FPS --band 0.25 \
    --processed data/MSMD/processed \
    --out_dir results/oracle_e1_fps${FPS} > /dev/null 2>&1

  python3 -c "
import json
d = json.load(open('results/oracle_e1_fps${FPS}/test/summary.json'))
g = lambda k: d.get(k, float('nan'))
print(f'{$FPS:<4}  {g(\"mean_mean_abs_err_sec\"):>6.3f}  {g(\"mean_pct_within_0.5s\"):>5.1f}%  "
      f'{g(\"mean_pct_within_0.25s\"):>6.1f}%  {g(\"mean_pct_within_0.1s\"):>5.1f}%  "
      f'{g(\"mean_recall_at_1\"):>7.1f}%')
" 2>/dev/null || echo "$FPS   (parse failed — see results/oracle_e1_fps${FPS}/test/summary.json)"
done

echo ""
echo "Reference: v5l (best model) = 2.21s / 31.8% @0.5s ; SOTA (Henkel) = 85.2% @0.5s"
echo "Job finished at $(date)"
