#!/bin/bash
#SBATCH --job-name=cyolo-aug-check
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cyolo_aug_check-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cyolo_aug_check-%j.log

# Does CPJKU's CYOLO TRAINING code actually run in our environment? This is the
# prerequisite for modifying their architecture (e.g. swapping MERT into the
# CYOLO backbone, which is the obvious experiment given CYOLO reaches 70.6% on
# real audio where our conditional-U-Net models reach 41.8%, while MERT is what
# doubles OUR real-audio robustness).
#
# Deliberately CPU + 1 epoch + 8 train / 4 val pieces: this answers "does the
# pipeline execute end-to-end and produce a checkpoint", not "does it converge".
# The GPU queue has been frozen for days, and there is no reason to spend a
# scarce GPU slot discovering a missing import. Their train.py exposes
# --device, --num_epochs and --train_split_files, so this needs no code changes.
#
# --no_log: train.py imports tensorboard only for logging, which is not
# installed in venv_cyolo. Adding it now would mean pip-installing into a venv
# that the SOTA eval job is actively using, risking that run for no benefit --
# this smoke test is about whether the TRAINING LOOP executes, not about
# logging. tensorboard gets added before any real training run.
#
# Runs WITH --augment so the augmentation path (on-the-fly IR convolution,
# tempo 0.5-2.0, image shifts) is exercised too -- that is the part we would
# most likely want to reuse.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
python -c "import torch,madmom,librosa,torchvision,torchaudio" \
  || { echo "FATAL: venv_cyolo incomplete -- see build_venv_cyolo log"; exit 1; }

CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
OUT=/scratch/pmohseni/cyolo_smoke
mkdir -p "$OUT/params" "$OUT/runs"
export PYTHONPATH=$CY:${PYTHONPATH:-}
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
cd "$CY/cyolo_score_following"

# train.py -> init_distributed_mode() takes its `elif SLURM_PROCID in environ`
# branch inside any SLURM job and then does
#     args.gpu = args.rank % torch.cuda.device_count()
# which is a ZeroDivisionError on CPU (device_count()==0). Clearing the rank
# variables makes it fall through to the plain "Not using distributed mode"
# path. This touches none of their code, and is only needed for the CPU smoke
# test -- a real GPU run has device_count()>=1 and the branch is harmless.
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK

echo "=== CYOLO training smoke: 1 epoch, CPU, 8 train / 4 val pieces ==="
python train.py \
    --train_sets "$DATA/msmd_train" \
    --val_sets   "$DATA/msmd_valid" \
    --train_split_files "$DATA/split_files/tiny_train_split.yaml" \
    --val_split_files   "$DATA/split_files/tiny_valid_split.yaml" \
    --config ./models/configs/cyolo.yaml \
    --augment \
    --no_log \
    --device cpu \
    --num_epochs 1 \
    --batch_size 8 \
    --num_workers 2 \
    --dump_root "$OUT/params" \
    --log_root  "$OUT/runs" \
    --tag cyolo_aug_check
STATUS=$?

echo ""
echo "=== produced checkpoints ==="
find "$OUT/params" -name "*.pt" -printf "  %p (%s bytes)\n" 2>/dev/null
echo "train.py exit status: $STATUS"
echo "Job finished at $(date)"
exit $STATUS
