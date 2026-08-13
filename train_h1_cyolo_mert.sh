#!/bin/bash
#SBATCH --job-name=h1-cyolo-mert
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/h1_cyolo_mert-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/h1_cyolo_mert-%j.log

# H1 -- MERT audio tower inside CYOLO's detector.
#
#   usage: sbatch train_h1_cyolo_mert.sh [cyolo|cyolo_sb] [aug_prob]
#          aug_prob = 0 (clean only) | 0.5 (multi-condition with degraded bank)
#
# End-to-end: audio + score image -> position. No symbolic intermediate, no
# MIDI, no OMR anywhere in the model.
#
# The two banks:
#   /scratch/pmohseni/mert_emb_cyolo      clean
#   /scratch/pmohseni/mert_emb_cyolo_ir   real-IR degraded (multi-condition)
# The degraded bank exists because CYOLO's own --ir_path convolves WAVEFORMS
# and cannot run once MERT is precomputed; without it, H1 could only be
# compared against the no-IR baseline row.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi | head -12

CFG=${1:-cyolo_sb}
AUG_PROB=${2:-0.5}
# H1_BANK selects the audio representation. The 176-dim AMT posteriorgram is the
# LOW-CAPACITY option and, on this data, capacity is the binding constraint:
# room accuracy fell monotonically with input dimension across every model we
# ran (78-dim mel 67.1 > 768 MERT 56.6 > 768+xattn 35.3 > 19.3 > 2.6), and H1
# with MERT overfit 12.6x (train frame-diff 2.04, val 25.73). The posteriorgram
# is 4.4x smaller AND room-invariant by measurement.
BANK=${H1_BANK:-mert}
case "$BANK" in
  mert) CLEAN=/scratch/pmohseni/mert_emb_cyolo;      DEG=/scratch/pmohseni/mert_emb_cyolo_ir ;;
  post) CLEAN=/scratch/pmohseni/amt_post_cyolo;      DEG="" ;;
  *) echo "FATAL: unknown H1_BANK=$BANK (mert|post)"; exit 1 ;;
esac
export H1_FEAT_AUG=${H1_FEAT_AUG:-0.5}
TAG=H1_${CFG}_${BANK}$([ "$AUG_PROB" != "0" ] && echo "_mc${AUG_PROB}")$([ "$H1_FEAT_AUG" != "0" ] && echo "_fa${H1_FEAT_AUG}")

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
python -c "import torch,madmom,librosa,cv2" || { echo "FATAL: venv_cyolo incomplete"; exit 1; }

CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
OUT=/scratch/pmohseni/h1_cyolo_mert/$TAG
mkdir -p "$OUT/params" "$OUT/runs"
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# CYOLO's init_distributed_mode branches on SLURM_PROCID and sets rank without
# world_size (utils/dist_utils.py:16), then dies on the missing attribute.
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
# OpenMP is not fork-safe and dataset.py:301 uses get_context("fork").Pool(8);
# with >1 thread the pool deadlocks before the first batch.
export OMP_NUM_THREADS=1

export H1_EMB_MAP="$DATA/msmd_train=$CLEAN/msmd_train;$DATA/msmd_valid=$CLEAN/msmd_valid"
if [ "$AUG_PROB" != "0" ] && [ -n "$DEG" ]; then
    export H1_AUG_MAP="$DATA/msmd_train=$DEG/msmd_train"
    export H1_AUG_PROB=$AUG_PROB
    echo "multi-condition ON: p(degraded)=$AUG_PROB (train set only; validation stays clean"
    echo "  so val loss remains comparable with every other run)"
fi

PARAM_FLAG=""
LAST=$(find "$OUT/params" -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$LAST" ] && { LAST=$(readlink -f "$LAST"); echo "Resuming from $LAST"; PARAM_FLAG="--param_path $LAST"; }

echo "=== $TAG: bank=$BANK feat_aug=$H1_FEAT_AUG (config=$CFG) ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_h1_cyolo_mert.py \
    --train_sets "$DATA/msmd_train" \
    --val_sets   "$DATA/msmd_valid" \
    --config "$CY/cyolo_score_following/models/configs/${CFG}.yaml" \
    --augment \
    --dump_root "$OUT/params" \
    --log_root  "$OUT/runs" \
    --tag ${TAG} \
    --num_workers 2 \
    $PARAM_FLAG
STATUS=$?

echo ""
find "$OUT/params" -name "*.pt" -printf "  ckpt %p (%s bytes)\n" 2>/dev/null | head -4
echo "exit status: $STATUS"
echo "Job finished at $(date)"
exit $STATUS
