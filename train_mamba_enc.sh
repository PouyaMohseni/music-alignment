#!/bin/bash
#SBATCH --job-name=mambaenc
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/mambaenc_%x-%j.log
# CODA's Mamba against CYOLO-SB's LSTM, one module apart, budget matched.
#
#   sbatch train_mamba_enc.sh lstm       the control: CNN + LSTM, our budget
#   sbatch train_mamba_enc.sh mamba      CODA's: a 2-layer causal Mamba IS the
#                                        audio tower, no CNN, 78-bin frames in
#   sbatch train_mamba_enc.sh cnnmamba   CNN kept, only the LSTM replaced
#
# The third arm decomposes the second. CODA's design changes the frame encoder
# and the recurrence at once, so on its own it cannot say which half matters;
# with MERT (frame encoder only, LSTM kept) the four rows separate them.
#
# CODA's Table 3 ablates the cascade, cross-attention, beam search, the temporal
# priors and scheduled sampling, and never the encoder, so nothing in their
# paper shows the Mamba tower earns its place. Both arms train here from the
# same init seed on the same data for the same wall clock, and only they are
# compared with each other -- never against Henkel's converged release.
set -uo pipefail
ARM=${1:?usage: train_mamba_enc.sh mamba|lstm}
case "$ARM" in
  lstm)     export MAMBA_ENC=0 ;;
  mamba)    export MAMBA_ENC=1 MAMBA_MODE=tower ;;
  cnnmamba) export MAMBA_ENC=1 MAMBA_MODE=seq ;;
  *) echo "FATAL: arm must be lstm, mamba or cnnmamba"; exit 1 ;;
esac
echo "Job started on $(hostname) at $(date), arm=$ARM"
nvidia-smi | head -12
CFG=${2:-cyolo_sb}
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
python -c "import torch,cv2;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" \
  || { echo "FATAL: venv_cyolo broken"; exit 1; }
[ "$ARM" != lstm ] && { python -c "from extensions.hooks.mamba_patch import _import_mamba; _import_mamba()" || { echo "FATAL: no mamba_ssm"; exit 1; }; }

CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
# separate dump roots: the resume block picks the newest .pt under $OUT/params,
# and a shared directory would warm-start one arm from the other's checkpoint
OUT=/scratch/pmohseni/mambaenc/${CFG}_${ARM}
mkdir -p "$OUT/params" "$OUT/runs"
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
# OpenMP is not fork-safe and dataset.py loads through get_context("fork").Pool(8)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# SLURM_PROCID alone sends init_distributed_mode down a branch that never sets
# world_size, which crashes every single-process run
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK

PARAM_FLAG=""
LAST=$(find "$OUT/params" -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$LAST" ] && { LAST=$(readlink -f "$LAST"); echo "Resuming from $LAST"; PARAM_FLAG="--param_path $LAST"; }

IR_PATH=/scratch/pmohseni/ir_bank
N_IR=$(find "$IR_PATH" -name '*.wav' 2>/dev/null | wc -l)
[ "$N_IR" -eq 0 ] && { echo "FATAL: no IRs under $IR_PATH"; exit 1; }
echo "=== $ARM arm, config=$CFG, IR on ($N_IR wavs) ==="
python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_mamba.py \
    --train_sets "$DATA/msmd_train" \
    --val_sets   "$DATA/msmd_valid" \
    --config "$CY/cyolo_score_following/models/configs/${CFG}.yaml" \
    --augment \
    --ir_path "$IR_PATH" \
    --dump_root "$OUT/params" \
    --log_root  "$OUT/runs" \
    --tag ${CFG}_${ARM} \
    --num_workers 2 \
    $PARAM_FLAG
STATUS=$?
find "$OUT/params" -name "*.pt" -printf "  ckpt %p (%s bytes)\n" 2>/dev/null | head -4
echo "exit $STATUS at $(date)"
exit $STATUS
