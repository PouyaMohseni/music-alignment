#!/bin/bash
# Evaluate all v5 variants on the test split.
# Run this tomorrow after training jobs finish.
# Usage: bash eval_v5_all.sh

cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate

EMB=/lustre07/scratch/pmohseni/music-alignment/data/MSMD/embeddings_lora
PROC=data/MSMD/processed

declare -A CONFIGS=(
  ["v5_recurrent"]="configs/v5_recurrent.yaml"
  ["v5b_large"]="configs/v5b_large.yaml"
  ["v5c_noxattn"]="configs/v5c_noxattn.yaml"
  ["v5d_long"]="configs/v5d_long.yaml"
)

echo "========================================"
echo "v5 family evaluation — $(date)"
echo "========================================"

for MODEL in v5_recurrent v5b_large v5c_noxattn v5d_long; do
  CKPT=$(ls results/$MODEL/checkpoint_*.pt 2>/dev/null | sort | tail -1)
  if [ -z "$CKPT" ]; then
    echo "[$MODEL] no checkpoint found — skipping"
    continue
  fi
  CFG=${CONFIGS[$MODEL]}
  echo ""
  echo "[$MODEL] checkpoint: $CKPT"
  python -m mymodel.v5_recurrent.eval \
    --checkpoint $CKPT \
    --config $CFG \
    --split test \
    --emb_root $EMB \
    --processed $PROC \
    2>&1 | grep -E "(mean_abs_err_sec|pct_within_0.5s|n_pieces|n_errors|checkpoint)"
done

echo ""
echo "========================================"
echo "Baseline (v3_all, same single-perf eval)"
echo "========================================"
V3_CKPT=$(ls results/v3_all/checkpoint_*.pt | sort | tail -1)
python -m mymodel.v3_fullseq.eval \
  --checkpoint $V3_CKPT \
  --split test \
  --emb_root $EMB \
  --processed $PROC \
  2>&1 | grep -E "(mean_abs_err_sec|pct_within_0.5s|n_pieces|checkpoint)"
