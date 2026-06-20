#!/bin/bash
#SBATCH --job-name=band-sweep-v5i
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v5i_bidir_residual/band_sweep-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v5i_bidir_residual/band_sweep-%j.log

echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

CKPT=results/v5i_bidir_residual/checkpoint_010000.pt
CFG=configs/v5i_bidir_residual.yaml

echo ""
echo "=== DTW Band Sweep on v5i_bidir_residual ==="
echo "band      err(s)   @0.5s"
echo "--------  -------  -----"

for BAND in 0.10 0.15 0.20 0.25 0.30 0.40 0.50; do
    python -m mymodel.v5_recurrent.eval \
        --checkpoint $CKPT \
        --config $CFG \
        --split test \
        --emb_root /lustre07/scratch/pmohseni/music-alignment/data/MSMD/embeddings_lora \
        --processed data/MSMD/processed \
        --advance_factor $BAND \
        --out_dir results/v5i_bidir_residual/eval_band${BAND} 2>&1 | tail -5

    python3 -c "
import json, sys
try:
    d = json.load(open('results/v5i_bidir_residual/eval_band${BAND}/test/summary.json'))
    print(f'band=${BAND}  err={d[\"mean_mean_abs_err_sec\"]:.3f}s  @0.5s={d[\"mean_pct_within_0.5s\"]:.1f}%')
except Exception as e:
    print(f'band=${BAND}  ERROR: {e}')
"
done

echo ""
echo "Job finished at $(date)"
