#!/bin/bash
#SBATCH --job-name=eval-v5-all
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v5_all-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v5_all-%j.log

# Evaluate all v5 variants and print a comparison table.
# Usage: sbatch eval_v5_all.sh   (or: bash eval_v5_all.sh on a GPU node)

cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate 2>/dev/null || true
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

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
  ["v5j_long"]="configs/v5j_long.yaml"
  ["v5k_pitch"]="configs/v5k_pitch.yaml"
  ["v5l_deep_bidir"]="configs/v5l_deep_bidir.yaml"
  ["v5m_big"]="configs/v5m_big.yaml"
  ["v6e0_pitch_aligned"]="configs/v6e0_pitch_aligned.yaml"
)

MODELS=(v5_recurrent v5b_large v5c_noxattn v5d_long v5e_scratch v5f_bidir v5g_residual v5h_deep v5i_bidir_residual v5j_long v5k_pitch v5l_deep_bidir v5m_big v6e0_pitch_aligned)

_parse() {
  python3 -c "
import json, sys
try:
    d = json.load(open('$1'))
    err = d.get('mean_mean_abs_err_sec', 0)
    pct = d.get('mean_pct_within_0.5s', 0)
    print(f'{err:.3f} {pct:.2f}')
except Exception as e:
    print('? ?')
"
}

echo "========================================"
echo "v5 sweep — $(date)"
printf "%-26s  %8s  %9s\n" "model" "err(s)" "@0.5s(%)"
echo "----------------------------------------"

for MODEL in "${MODELS[@]}"; do
  CKPT=$(ls results/$MODEL/checkpoint_*.pt 2>/dev/null | sort | tail -1)
  if [ -z "$CKPT" ]; then
    printf "%-26s  %8s  %9s\n" "$MODEL" "no_ckpt" "-"
    continue
  fi
  CFG=${CONFIGS[$MODEL]}
  SUMM=results/$MODEL/eval/test/summary.json
  if [ ! -f "$SUMM" ]; then
    echo "  evaluating $MODEL ($CKPT)..."
    python -m mymodel.v5_recurrent.eval \
      --checkpoint $CKPT \
      --config $CFG \
      --split test \
      --emb_root $EMB \
      --processed $PROC \
      --out_dir results/$MODEL/eval \
      > /dev/null 2>&1
  fi
  PARSED=$(_parse "$SUMM")
  ERR=$(echo $PARSED | cut -d' ' -f1)
  PCT=$(echo $PARSED | cut -d' ' -f2)
  printf "%-26s  %8s  %9s\n" "$MODEL" "${ERR}s" "${PCT}%"
done

echo "----------------------------------------"
V3_CKPT=$(ls results/v3_all/checkpoint_*.pt 2>/dev/null | sort | tail -1)
V3_SUMM=results/v3_all/eval_singleperf/test/summary.json
if [ -n "$V3_CKPT" ] && [ ! -f "$V3_SUMM" ]; then
  echo "  evaluating v3_all baseline..."
  python -m mymodel.v3_fullseq.eval \
    --checkpoint $V3_CKPT \
    --split test \
    --emb_root $EMB \
    --processed $PROC \
    --out_dir results/v3_all/eval_singleperf \
    > /dev/null 2>&1
fi
if [ -f "$V3_SUMM" ]; then
  PARSED=$(_parse "$V3_SUMM")
  ERR=$(echo $PARSED | cut -d' ' -f1)
  PCT=$(echo $PARSED | cut -d' ' -f2)
  printf "%-26s  %8s  %9s\n" "v3_all (baseline)" "${ERR}s" "${PCT}%"
fi
echo "========================================"
