#!/bin/bash
#SBATCH --job-name=eval-v11-madmom-interim
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v11_madmom_interim-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v11_madmom_interim-%j.log

# Interim spot-check of v11-madmom's current best_model.pt --
# train_v11_madmom.sh only evaluates AFTER its own training loop fully
# finishes (up to 24h), so without this there'd be no read on progress
# until then.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

PROC=data/MSMD/processed
CKPT=results/v11_madmom/best_model.pt
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls results/v11_madmom/checkpoint_epoch*.pt 2>/dev/null | sort | tail -1)
fi
if [ -z "$CKPT" ]; then
    echo "No checkpoint exists yet. Exiting."
    exit 0
fi
echo "Evaluating: $CKPT"

python -m mymodel.v11_cpjku_fullstrip.eval \
    --checkpoint $CKPT \
    --config     configs/v11_madmom.yaml \
    --split      test \
    --processed  $PROC \
    --cpjku_fmt_root data/MSMD/cpjku_fmt \
    --out_dir    results/v11_madmom/eval_interim

echo "Job finished at $(date)"
