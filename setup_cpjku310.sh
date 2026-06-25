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

echo "=== Installing madmom from local tarball (compute nodes have no internet) ==="
# Tarball pre-downloaded to scratch on the login node.
# --no-build-isolation: pip's isolated build env won't have Cython.
MADMOM_TGZ=/scratch/pmohseni/pip_packages/madmom-0.16.1.tar.gz
if [ ! -f "$MADMOM_TGZ" ]; then
    echo "ERROR: $MADMOM_TGZ not found. Run on login node: curl -L -o $MADMOM_TGZ 'https://files.pythonhosted.org/packages/c7/a3/9f3de3e8068a3606331134d96b84c8db4f7624d6715be8ab3c1f56e6731d/madmom-0.16.1.tar.gz'"
    exit 1
fi
pip install "$MADMOM_TGZ" --no-build-isolation

echo "=== Patching madmom for Python 3.10 ==="
# madmom 0.16.1 uses 'from collections import MutableSequence' which was
# removed in Python 3.10 (moved to collections.abc).
PROCESSORS="${VENV}/lib/python3.10/site-packages/madmom/processors.py"
sed -i 's/from collections import MutableSequence/from collections.abc import MutableSequence/' "$PROCESSORS"
echo "  patched $PROCESSORS"

echo "=== Installing soundfile (madmom audio I/O) ==="
pip install soundfile

echo "=== Installing PyTorch from CVMFS (no internet needed on compute nodes) ==="
pip install torch --no-index

echo "=== Installing remaining deps from CVMFS ==="
pip install packaging tqdm PyYAML tensorboard soundfile --no-index

echo "=== Installing cv2 stub (system OpenCV is Python 3.11-only, ABI-incompatible with 3.10) ==="
# The cluster's opencv module provides cv2 compiled for Python 3.11 only.
# We install a pure-Python stub that implements resize/cvtColor/constants via PIL.
CPJKU_CV2_STUB=/lustre06/project/6002780/pmohseni/music-alignment/mymodel/cpjku_adapter/cv2_stub
mkdir -p "${SITE}/cv2"
cp "${CPJKU_CV2_STUB}/__init__.py" "${SITE}/cv2/__init__.py"
echo "  wrote ${SITE}/cv2/__init__.py"

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
