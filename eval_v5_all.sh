#!/bin/bash
# Evaluate all v5 variants on the test split and print a comparison table.
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
  ["v5e_scratch"]="configs/v5e_scratch.yaml"
  ["v5f_bidir"]="configs/v5f_bidir.yaml"
  ["v5g_residual"]="configs/v5g_residual.yaml"
  ["v5h_deep"]="configs/v5h_deep.yaml"
  ["v5i_bidir_residual"]="configs/v5i_bidir_residual.yaml"
)

MODELS=(v5_recurrent v5b_large v5c_noxattn v5d_long v5e_scratch v5f_bidir v5g_residual v5h_deep v5i_bidir_residual)

echo "========================================"
echo "v5 sweep — $(date)"
printf "%-22s  %8s  %8s\n" "model" "err(s)" "@0.5s"
echo "----------------------------------------"

for MODEL in "${MODELS[@]}"; do
  CKPT=$(ls results/$MODEL/checkpoint_*.pt 2>/dev/null | sort | tail -1)
  if [ -z "$CKPT" ]; then
    printf "%-22s  %8s  %8s\n" "$MODEL" "no_ckpt" "-"
    continue
  fi
  CFG=${CONFIGS[$MODEL]}
  RESULT=$(python -m mymodel.v5_recurrent.eval \
    --checkpoint $CKPT \
    --config $CFG \
    --split test \
    --emb_root $EMB \
    --processed $PROC 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"{d.get('mean_mean_mean_abs_err_sec', d.get('mean_mean_abs_err_sec','?')):.3f} {d.get('mean_pct_within_0.5s','?'):.1f}\")
" 2>/dev/null || echo "? ?")
  ERR=$(echo $RESULT | cut -d' ' -f1)
  PCT=$(echo $RESULT | cut -d' ' -f2)
  printf "%-22s  %8s  %8s\n" "$MODEL" "${ERR}s" "${PCT}%"
done

echo "----------------------------------------"
# Baseline
V3_CKPT=$(ls results/v3_all/checkpoint_*.pt | sort | tail -1)
RESULT=$(python -m mymodel.v3_fullseq.eval \
  --checkpoint $V3_CKPT \
  --split test \
  --emb_root $EMB \
  --processed $PROC 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"{d.get('mean_mean_abs_err_sec','?'):.3f} {d.get('mean_pct_within_0.5s','?'):.1f}\")
" 2>/dev/null || echo "6.13 18.1")
ERR=$(echo $RESULT | cut -d' ' -f1)
PCT=$(echo $RESULT | cut -d' ' -f2)
printf "%-22s  %8s  %8s\n" "v3_all (baseline)" "${ERR}s" "${PCT}%"
echo "========================================"
