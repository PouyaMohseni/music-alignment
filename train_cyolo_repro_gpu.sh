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
# COMPLEX stft for torchaudio.functional.phase_vocoder.
#
# CORRECTION (2026-08-08): an earlier version of this comment claimed --augment
# covers "IR convolution + tempo + image shifts". It does NOT include IR.
# Verified: train.py:241 declares --ir_path with default=None, and
# dataset.py:317-319 constructs ImpulseResponse only `if ir_path is not None`.
# --augment alone gives tempo 0.5-2.0 and image shifts only. Every run this
# script produced before today therefore had NO reverb augmentation, which is
# why it landed on the paper's no-IR row. See the IR_PATH block below.

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
# IR and no-IR runs MUST NOT share a dump root: the resume block below picks the
# newest .pt under $OUT/params, so a shared directory would silently warm-start
# the IR run from the no-IR checkpoint and destroy the ablation.
IR_TAG=$([ "${2:-/scratch/pmohseni/ir_bank}" = "none" ] && echo noir || echo ir)
OUT=/scratch/pmohseni/cyolo_repro/${CFG}_${IR_TAG}
mkdir -p "$OUT/params" "$OUT/runs"
export PYTHONPATH=$CY:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
cd "$CY/cyolo_score_following"

# Resume across the 24h wall if a previous round left a checkpoint.
PARAM_FLAG=""
LAST=$(find "$OUT/params" -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$LAST" ] && { LAST=$(readlink -f "$LAST"); echo "Resuming from $LAST"; PARAM_FLAG="--param_path $LAST"; }

# CYOLO's init_distributed_mode (utils/dist_utils.py:11) branches on env vars:
#   'RANK'+'WORLD_SIZE' present -> reads all three, fine
#   'SLURM_PROCID' present      -> sets args.rank and args.gpu but NOT
#                                  args.world_size, then line 37 passes
#                                  world_size=args.world_size to
#                                  init_process_group -> AttributeError
#   neither                     -> "Not using distributed mode", rank=0, returns
#
# SLURM sets SLURM_PROCID in EVERY job, so the middle branch always fires and
# always crashes for a single-process run. That killed job 66979946 after 8min
# with AttributeError: 'Namespace' object has no attribute 'world_size', and on
# CPU the same branch dies as ZeroDivisionError instead (args.rank %
# torch.cuda.device_count() with 0 GPUs). Unsetting these forces the clean
# non-distributed path. No code change to their repo needed.
#
# This fix was already applied to the CPU smoke-test script and I failed to
# carry it into this one.
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK

# OpenMP is NOT fork-safe, and cyolo_score_following/dataset.py:301 loads the
# training set through get_context("fork").Pool(8). If the OpenMP runtime has
# already spawned its thread pool in the parent (which it does lazily on the
# first numpy/torch call), the forked children inherit locks held by threads
# that do not exist in the child and block forever on a futex.
#
# That is exactly how job 88219 died: it printed "Loading 353 file(s)...",
# then sat at 0/353 for 50 minutes with FOUR python processes in
# futex_wait_queue, 8 SECONDS of CPU between them, and the GPU at 0%.
# It was deadlocked, not slow.
#
# The two smoke-test scripts that DO work (smoke_train_cyolo_cpu.sh:44,
# smoke_train_cyolo_noaug_cpu.sh:44) both set OMP_NUM_THREADS explicitly and
# pass --num_workers 2; this script set neither. 1 thread x Pool(8) also
# matches --cpus-per-task=8 exactly instead of oversubscribing it.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# IR augmentation is gated on --ir_path, NOT on --augment (train.py:241 defaults
# it to None; dataset.py:317-319 only builds ImpulseResponse when it is set).
# Without it this script reproduced the paper's published NO-IR row -- 80.4
# synth / 46.0 room -- which is exactly where our own models sit (81.1 / 45.6).
# Henkel & Widmer report the IR delta as 46.0 -> 71.2 on room, so this flag is
# worth more than every architecture change we have tried.
#
# IR_PATH=none reproduces the old no-IR behaviour on purpose, so the pair of
# runs is a clean controlled ablation rather than a replacement.
IR_PATH=${2:-/scratch/pmohseni/ir_bank}
IR_FLAG=""
if [ "$IR_PATH" != "none" ]; then
    N_IR=$(find "$IR_PATH" -name '*.wav' 2>/dev/null | wc -l)
    [ "$N_IR" -eq 0 ] && { echo "FATAL: no .wav IRs under $IR_PATH"; exit 1; }
    echo "IR augmentation ON: $N_IR wavs under $IR_PATH (CYOLO applies its own"
    echo "  FILTER_LIST and drops virtual-membranes, so the used count is lower)"
    IR_FLAG="--ir_path $IR_PATH"
else
    echo "IR augmentation OFF (reproduces the published no-IR row)"
fi

echo "=== training CYOLO config=$CFG (full msmd_train, --augment, ir=$IR_PATH) ==="
python train.py \
    --train_sets "$DATA/msmd_train" \
    --val_sets   "$DATA/msmd_valid" \
    --config ./models/configs/${CFG}.yaml \
    --augment \
    $IR_FLAG \
    --dump_root "$OUT/params" \
    --log_root  "$OUT/runs" \
    --tag ${CFG}_repro \
    --num_workers 2 \
    $PARAM_FLAG
STATUS=$?

echo ""
find "$OUT/params" -name "*.pt" -printf "  ckpt %p (%s bytes)\n" 2>/dev/null | head -4
echo "train.py exit status: $STATUS"
echo "Job finished at $(date)"
exit $STATUS
