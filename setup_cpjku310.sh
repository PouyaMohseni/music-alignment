#!/bin/bash
#SBATCH --job-name=setup-cpjku310
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/setup_cpjku310-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/setup_cpjku310-%j.log

# Create a Python 3.10 virtualenv with madmom + CPJKU dependencies.
# Run once before eval_cpjku_native.sh / train_cpjku_native.sh.
#
#   sbatch setup_cpjku310.sh
#
# numpy is pinned to 1.23.x: their dataset.py uses dtype=np.int which was
# removed in numpy 1.24.  All other deps are unpinned.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv python/3.10

# Venv lives on /scratch (separate inode quota, ~600k free vs 0 on /lustre06).
# /scratch is user-private and persists ≥60 days of access on Alliance Canada.
VENV=/scratch/pmohseni/venv_cpjku310

if [ -d "$VENV" ]; then
    echo "Removing existing $VENV ..."
    rm -rf "$VENV"
fi

echo "=== Creating venv $VENV ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip wheel

echo "=== Installing numpy (pinned <1.24 for madmom np.int compatibility) ==="
# Cluster has: 1.21.2, 1.22.4, 1.24.4+.  Must stay below 1.24 (np.int removed).
pip install "numpy==1.22.4"

echo "=== Installing scipy 1.10.1 + Cython (madmom build deps) ==="
# scipy pinned to 1.10.1: newest cluster wheel compatible with numpy 1.22.4
# (scipy >= 1.11 requires numpy >= 1.23.5 which isn't available here)
pip install "scipy==1.10.1" "Cython>=0.25"

echo "=== Installing madmom (--no-build-isolation uses venv Cython) ==="
# madmom 0.16.1 uses Cython extensions; pip's isolated build env won't have
# Cython, so we disable isolation.
pip install "madmom==0.16.1" --no-build-isolation

echo "=== Patching madmom for Python 3.10 ==="
# madmom 0.16.1 uses 'from collections import MutableSequence' which was
# removed in Python 3.10 (moved to collections.abc).
PROCESSORS="${VENV}/lib/python3.10/site-packages/madmom/processors.py"
sed -i 's/from collections import MutableSequence/from collections.abc import MutableSequence/' "$PROCESSORS"
echo "  patched $PROCESSORS"

echo "=== Installing soundfile (madmom audio I/O) ==="
pip install soundfile

echo "=== Installing PyTorch (CUDA 12.1 wheels) ==="
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

echo "=== Installing remaining deps ==="
pip install tqdm PyYAML tensorboard

echo "=== Making audio_conditioned_unet importable via .pth (avoids setup.py download) ==="
# Their setup.py downloads MSMD dataset (~GB) on install; we bypass it entirely
# by writing a .pth file into site-packages so Python finds the package directly.
git submodule update --init third_party/cpjku_unet
cd third_party/cpjku_unet && git checkout ismir-2020 && cd ../..
SITE="${VENV}/lib/python3.10/site-packages"
echo "$(pwd)/third_party/cpjku_unet" > "${SITE}/cpjku_unet.pth"
echo "  wrote ${SITE}/cpjku_unet.pth"

echo ""
echo "=== Verifying key imports ==="
python3 -c "
import numpy as np; print(f'numpy {np.__version__}')
import madmom; print(f'madmom {madmom.__version__}')
import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}')
import audio_conditioned_unet; print('audio_conditioned_unet OK')
from madmom.audio.signal import SignalProcessor; print('madmom.audio.signal OK')
"

echo "Setup done at $(date)."
echo "Activate with: source $VENV/bin/activate"
