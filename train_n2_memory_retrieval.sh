#!/bin/bash
#SBATCH --job-name=n2-memory-retrieval
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/n2_memory_retrieval-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/n2_memory_retrieval-%j.log

# N2: keep CB_TA's LSTM exactly as-is and ADD a zero-init-gated, lag-aware
# retrieval read over a compressed bank of the piece's own past conditioning
# vectors. Same target as N1 (burst mislocalisation on repeat-heavy pieces:
# median onset error 0.000s but mean 1.3-12.4s in B1a), but the conservative
# variant -- because in this project every FiLM REPLACEMENT lost ground
# (spatial 44.3%, cross-attention 71.1%, gated 82.9% vs B1a 89.2%) while the
# only thing that beat B1a was B3 (89.8%), an ADDITIVE change on a converged
# checkpoint. The LSTM keeps stock parameter names and the gate starts at
# zero, so warm-started from B1a this network computes EXACTLY B1a at step
# zero (asserted in scripts/smoke_test_temporal_arch.py) and can only earn
# its way up from 89.2%. The lag embedding is the load-bearing part: a strong
# match to audio ~40s old is evidence of being on the SECOND pass, which is
# what tells the model to advance rather than jump back.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/N2_memory_retrieval
mkdir -p "$OUT/runs" "$OUT/params"

# Resume from our own latest; on the FIRST run warm-start from B1a's converged
# checkpoint so only the new temporal core trains from scratch.
PARAM_FLAG=""
LATEST_CKPT=$(find "$OUT/params" -name "latest_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
              | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from $LATEST_CKPT"
    PARAM_FLAG="--param_path $LATEST_CKPT"
else
    # MUST be absolute: this script cd's to $REPO/audio_conditioned_unet before
    # invoking python, so a relative path resolves against the wrong directory
    # and train_model.py dies with FileNotFoundError on a checkpoint that does
    # exist. (Exactly what killed jobs 66850715/16/17.)
    B1A_CKPT=$(find /project/def-ichiro/pmohseni/music-alignment/results/cb_ta_ext/B1a_mert_swap/params \
               -name "best_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null \
               | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "$B1A_CKPT" ] && B1A_CKPT=$(readlink -f "$B1A_CKPT")
    if [ -n "$B1A_CKPT" ]; then
        echo "Cold start: warm-starting from B1a best_model.pt -> $B1A_CKPT"
        PARAM_FLAG="--param_path $B1A_CKPT"
    else
        echo "No B1a checkpoint found, training from scratch"
    fi
fi

TRAIN_SET=/scratch/pmohseni/msmd_train_full
VAL_SET=../data/msmd/msmd_valid
export MERT_PATH_MAP="${TRAIN_SET}=/scratch/pmohseni/mert_emb_zenodo/train_full;${VAL_SET}=/scratch/pmohseni/mert_emb_zenodo/msmd_valid"

echo "=== N2: LSTM kept verbatim + zero-init-gated lag-aware retrieval read ==="
cd "$REPO/audio_conditioned_unet"

python /project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_train_n2_memory_retrieval.py \
    --film_layers 2 3 4 5 6 7 8 \
    --log_root  "$OUT/runs" \
    --dump_root "$OUT/params" \
    --train_set "$TRAIN_SET" \
    --val_set   "$VAL_SET" \
    --use_lstm \
    --augment \
    --config    configs/msmd_aug.yaml \
    --audio_encoder MERTProjector \
    --tag N2_memory_retrieval \
    $PARAM_FLAG

echo ""
echo "Training finished at $(date)"
