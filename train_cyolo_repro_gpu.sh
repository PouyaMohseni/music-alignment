#!/bin/bash
#SBATCH --job-name=cyolo-repro
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cyolo_repro-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cyolo_repro-%j.log

# Train CPJKU's CYOLO from scratch on the data from their own Zenodo release --
# the first REAL training run of the paper's model here. Everything before this
# was CPU smoke-testing that only proved the pipeline executes.
#
# Why it matters: their released cyolo_sb_a checkpoint scores 86.5 on real
# room audio where our best model scores 41.8. Reproducing their TRAINING (not
# just their inference) is the prerequisite for the experiment that follows --
# MERT inside the CYOLO backbone.
#
#   usage: sbatch train_cyolo_repro_gpu.sh [cyolo|cyolo_sb]
#
# CAVEAT on "+A": the paper's best row (cyolo_sb_a, 86.5) additionally trains on
# scanned scores with only system/bar-level alignments (Mozart, Beethoven,
# Debussy, Schubert, Schumann). That extra corpus is NOT in the Zenodo msmd.zip,
# so what is reproducible here is `cyolo` (58.1 real) and `cyolo_sb` (63.0 real),
# not the 70.6/86.5 "+A" row. Stating that up front rather than discovering it
# in the numbers later.
#
# --augment is ON and depends on the phase_vocoder port in
# models/custom_modules.py (backup: custom_modules.py.orig): torch>=2 requires a
# COMPLEX stft for torchaudio.functional.phase_vocoder. Their augmentation is
# IR convolution + tempo 0.5-2.0 + image shifts, which is plausibly what buys
# CYOLO its -4.3pt synthetic->real degradation vs our -48, so it must be active.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi | head -12

CFG=${1:-cyolo_sb}
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
python -c "import torch,madmom,librosa,cv2;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" \
  || { echo "FATAL: venv_cyolo broken"; exit 1; }

CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
OUT=/scratch/pmohseni/cyolo_repro/$CFG
mkdir -p "$OUT/params" "$OUT/runs"
export PYTHONPATH=$CY:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
cd "$CY/cyolo_score_following"

# Resume across the 24h wall if a previous round left a checkpoint.
PARAM_FLAG=""
LAST=$(find "$OUT/params" -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$LAST" ] && { LAST=$(readlink -f "$LAST"); echo "Resuming from $LAST"; PARAM_FLAG="--param_path $LAST"; }

echo "=== training CYOLO config=$CFG (full msmd_train, --augment) ==="
python train.py \
    --train_sets "$DATA/msmd_train" \
    --val_sets   "$DATA/msmd_valid" \
    --config ./models/configs/${CFG}.yaml \
    --augment \
    --dump_root "$OUT/params" \
    --log_root  "$OUT/runs" \
    --tag ${CFG}_repro \
    $PARAM_FLAG
STATUS=$?

echo ""
find "$OUT/params" -name "*.pt" -printf "  ckpt %p (%s bytes)\n" 2>/dev/null | head -4
echo "train.py exit status: $STATUS"
echo "Job finished at $(date)"
exit $STATUS
