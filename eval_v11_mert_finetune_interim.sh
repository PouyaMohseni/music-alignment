#!/bin/bash
#SBATCH --job-name=eval-v11-mert-finetune-interim
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v11_mert_finetune_interim-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v11_mert_finetune_interim-%j.log

# Interim spot-check of v11-mert-finetune's current best_model.pt --
# train_v11_mert_finetune.sh only evaluates AFTER its own training loop
# fully finishes (up to 24h), so without this there'd be no read on
# progress until then. Scheduled with --begin=now+5hours so it runs against
# whatever checkpoint exists at that point, not a fixed epoch count.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT=results/v11_mert_finetune/best_model.pt
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls results/v11_mert_finetune/checkpoint_epoch*.pt 2>/dev/null | sort | tail -1)
fi
if [ -z "$CKPT" ]; then
    echo "No checkpoint exists yet -- training hasn't completed even one epoch. Exiting."
    exit 0
fi
echo "Evaluating: $CKPT"

python -m mymodel.v11_mert_finetune.eval \
    --checkpoint $CKPT \
    --config     configs/v11_mert_finetune.yaml \
    --split      test \
    --processed  data/MSMD/processed \
    --out_dir    results/v11_mert_finetune/eval_interim

echo "Job finished at $(date)"
