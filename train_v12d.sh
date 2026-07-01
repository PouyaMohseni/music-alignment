#!/bin/bash
#SBATCH --job-name=music-v12d
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v12d-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v12d-%j.log

# v12d: MERT(LoRA r=8) + BiLSTM + CrossAttention(audio->score) + InfoNCE
# Most expressive variant; cross-attention lets audio frames explicitly query score columns

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source .venv/bin/activate

export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1

OUT=results/v12d
mkdir -p $OUT

RESUME_FLAG=""
if [ -f "$OUT/latest.pt" ]; then
    echo "Resuming from $OUT/latest.pt"
    RESUME_FLAG="--resume $OUT/latest.pt"
fi

python -m mymodel.v12_mert_align.train_variant \
    --variant    v12d \
    --data_root  data/MSMD/processed \
    --out        $OUT \
    --epochs     30 \
    --tau        0.07 \
    --w_infonce  1.0 \
    --w_expected 0.5 \
    --embed_dim  256 \
    --lstm_hidden 512 \
    --lstm_layers 2 \
    --lora_rank  8 \
    --attn_heads 4 \
    --patience   8 \
    --device     cuda \
    $RESUME_FLAG

echo ""
echo "Training done. Running test eval..."
python -m mymodel.v12_mert_align.eval_variant \
    --checkpoint $OUT/best_model.pt \
    --split test \
    --data_root data/MSMD/processed \
    --out $OUT/test_results.json

echo "Job finished at $(date)"
