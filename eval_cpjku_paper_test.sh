#!/bin/bash
#SBATCH --job-name=cpjku-paper-eval
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper_eval-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper_eval-%j.log

# Reproduce CB_TA paper results (Table 2 + Table 3) exactly as in Henkel et al. ISMIR 2020.
# Uses their unmodified eval_model.py, their msmd_test data, synthesized audio via FluidSynth.
#
# Their exact README eval commands:
#   python eval_model.py --param_path ../models/CB_TA/best_model.pt \
#     --test_dir ../data/msmd/msmd_test --config configs/msmd.yaml
#   python eval_model.py ... --eval_onsets   (for Table 3)
#
# Pass --param_path to evaluate a custom trained model instead of their pretrained weights:
#   sbatch eval_cpjku_paper_test.sh /path/to/your/best_model.pt
#
# Prerequisites: run setup_cpjku_paper_login.sh on login node first.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

git submodule update --init third_party/cpjku_unet || true
cd third_party/cpjku_unet && git checkout ismir-2020 && cd ../.. || true

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

# FluidSynth: shared library + CLI binary (midi_to_spec_otf uses subprocess "fluidsynth -F ...")
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}

# Prevent BLAS-fork deadlock
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
PARAMS_ROOT=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper/CB_TA/params

# Priority:
#   1. Explicit path passed as $1
#   2. Latest best_model.pt under results/cpjku_paper/CB_TA/params/ (auto-discover)
#   3. Pretrained CB_TA weights from submodule
_TMPDIR=""
if [ -n "${1:-}" ]; then
    PARAM_PATH="$1"
else
    LATEST_DIR=$(ls -dt "$PARAMS_ROOT"/*/  2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -f "${LATEST_DIR}best_model.pt" ]; then
        PARAM_PATH="${LATEST_DIR}best_model.pt"
        echo "Auto-discovered model: $PARAM_PATH"
    else
        PARAM_PATH="$REPO/models/CB_TA/best_model.pt"
        echo "No trained model found — using pretrained CB_TA: $PARAM_PATH"
    fi
fi

if [ -n "${1:-}" ]; then
    _TMPDIR=$(mktemp -d /scratch/pmohseni/eval_checkpoint_XXXXXX)
    python3 - <<PYEOF
import torch, json, os, sys, shutil
src = "$1"
dst_dir = "$_TMPDIR"
sd = torch.load(src, map_location='cpu', weights_only=False)
if isinstance(sd, dict) and 'state_dict' in sd:
    torch.save(sd['state_dict'], os.path.join(dst_dir, 'best_model.pt'))
    cfg = sd.get('net_config')
    if cfg is None:
        shutil.copy("$REPO/models/CB_TA/net_config.json", os.path.join(dst_dir, 'net_config.json'))
    else:
        with open(os.path.join(dst_dir, 'net_config.json'), 'w') as f:
            json.dump(cfg, f)
    print(f"Extracted state_dict from {src} -> {dst_dir}/best_model.pt")
else:
    # Bare state dict: copy as-is, find net_config.json from same dir
    shutil.copy(src, os.path.join(dst_dir, 'best_model.pt'))
    nc = os.path.join(os.path.dirname(src), 'net_config.json')
    if os.path.exists(nc):
        shutil.copy(nc, os.path.join(dst_dir, 'net_config.json'))
    else:
        shutil.copy("$REPO/models/CB_TA/net_config.json", os.path.join(dst_dir, 'net_config.json'))
    print(f"Copied checkpoint {src} -> {dst_dir}/best_model.pt")
PYEOF
    PARAM_PATH="$_TMPDIR/best_model.pt"
    echo "Model path for eval: $PARAM_PATH"
fi

trap '[ -n "$_TMPDIR" ] && rm -rf "$_TMPDIR"' EXIT

OUT=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_paper
mkdir -p "$OUT"

cd "$REPO/audio_conditioned_unet"

echo "=== Eval CB_TA: msmd_test, synthesized audio (Table 2 — pixel/F1 metrics) ==="
python eval_model.py \
    --param_path  "$PARAM_PATH" \
    --test_dir    ../data/msmd/msmd_test \
    --config      configs/msmd.yaml \
    --scale_factor 3 \
    --batch_size  1 \
    --seq_len     128 \
    --piecewise_stats \
    2>&1 | tee "$OUT/eval_CB_TA_test_f1.log"

echo ""
echo "=== Eval CB_TA: msmd_test, synthesized audio (Table 3 — onset timing) ==="
python eval_model.py \
    --param_path  "$PARAM_PATH" \
    --test_dir    ../data/msmd/msmd_test \
    --config      configs/msmd.yaml \
    --scale_factor 3 \
    --batch_size  1 \
    --seq_len     128 \
    --eval_onsets \
    --piecewise_stats \
    2>&1 | tee "$OUT/eval_CB_TA_test_onsets.log"

echo ""
echo "Job finished at $(date)"
