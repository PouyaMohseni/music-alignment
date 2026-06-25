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

VENV=.venv_cpjku310

if [ -d "$VENV" ]; then
    echo "Removing existing $VENV ..."
    rm -rf "$VENV"
fi

echo "=== Creating venv $VENV ==="
python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip wheel

echo "=== Installing numpy (pinned <1.24 for madmom np.int compatibility) ==="
pip install "numpy==1.23.5"

echo "=== Installing scipy + Cython (madmom build deps) ==="
pip install "scipy>=1.7" "Cython>=0.25"

echo "=== Installing madmom ==="
pip install "madmom==0.16.1"

echo "=== Installing soundfile (madmom audio I/O) ==="
pip install soundfile

echo "=== Installing PyTorch (CUDA 12.1 wheels) ==="
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121

echo "=== Installing remaining deps ==="
pip install tqdm PyYAML tensorboard

echo "=== Installing CPJKU package (editable, no deps) ==="
# --no-deps: avoids overwriting our pinned numpy with their unpinned requirement
git submodule update --init third_party/cpjku_unet
cd third_party/cpjku_unet && git checkout ismir-2020 && cd ../..
pip install --no-deps -e third_party/cpjku_unet

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
