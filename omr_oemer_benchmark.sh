#!/bin/bash
#SBATCH --job-name=oemer-bench
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/pmohseni/omr/logs/oemer-%j.log
#SBATCH --error=/scratch/pmohseni/omr/logs/oemer-%j.log
#
# Measure oemer notehead accuracy on clean LilyPond-engraved MSMD piano pages.
#
# Environment notes (all load-bearing on Narval):
#   * venv lives on /scratch -- /project is at ~96% of its inode quota.
#     Call the interpreter by ABSOLUTE PATH; do not `source .../activate`,
#     which can point VIRTUAL_ENV back at the /project venv.
#   * `pip install opencv-python-headless` installs a dummy wheel here, so
#     oemer was installed with --no-deps and cv2 comes from the system module.
#     Module order matters: python FIRST, then gcc+opencv.
#   * Compute nodes have no internet.  The two onnx checkpoints were fetched
#     on the login node into $PY_PREFIX/oemer/checkpoints/{unet_big,seg_net}/
#     and are asserted present below.
#
# Usage: sbatch omr_oemer_benchmark.sh <variant> [n_workers]
#        variant = native | hires | smoke

set -u
VARIANT=${1:-native}
NPROC=${2:-6}
WORK=/scratch/pmohseni/omr
PY=/scratch/pmohseni/venv_oemer/bin/python
SITE=/scratch/pmohseni/venv_oemer/lib/python3.11/site-packages

module load python/3.11
module load gcc/12.3 opencv/4.11.0

mkdir -p "$WORK/logs" "$WORK/out/$VARIANT"

echo "host=$(hostname) start=$(date) variant=$VARIANT nproc=$NPROC"
$PY -c "import cv2,onnxruntime,oemer;print('cv2',cv2.__version__,'ort',onnxruntime.__version__)" || exit 1

for c in unet_big seg_net; do
    f=$SITE/oemer/checkpoints/$c/model.onnx
    if [ ! -s "$f" ]; then echo "MISSING CHECKPOINT $f (must be fetched on login node)"; exit 1; fi
    echo "checkpoint ok: $f ($(stat -c%s "$f") bytes)"
done

# onnxruntime + BLAS threads: keep each worker narrow so NPROC workers fit.
THREADS=$(( ${SLURM_CPUS_PER_TASK:-24} / NPROC ))
[ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
echo "threads per worker=$THREADS"

case "$VARIANT" in
  smoke) SRC=native; LIMIT=1 ;;
  *)     SRC=$VARIANT; LIMIT=0 ;;
esac

LIST=$($PY - "$WORK/pages/manifest.json" "$SRC" "$LIMIT" <<'EOF'
import json,sys
man=json.load(open(sys.argv[1])); src=sys.argv[2]; lim=int(sys.argv[3])
rows=[m for m in man if m.get(src)]
if lim: rows=rows[:lim]
print("\n".join("%s\t%s"%(m["key"],m[src]) for m in rows))
EOF
)

echo "$LIST" | wc -l | xargs echo "pages to process:"

echo "$LIST" | xargs -P "$NPROC" -I{} bash -c '
  IFS=$'"'"'\t'"'"' read -r key img <<< "{}"
  out='"$WORK/out/$VARIANT"'/$key.json
  if [ -s "$out" ]; then echo "SKIP $key"; exit 0; fi
  s=$(date +%s)
  '"$PY"' '"$PWD"'/scripts/omr_run_oemer.py --img "$img" --out "$out" \
      --musicxml '"$WORK/out/$VARIANT"'/$key.musicxml \
      > '"$WORK/logs"'/'"$VARIANT"'_$key.log 2>&1
  rc=$?
  echo "$key rc=$rc $(( $(date +%s) - s ))s"
'

echo "done=$(ls $WORK/out/$VARIANT/*.json 2>/dev/null | wc -l) end=$(date)"
