#!/bin/bash
# ============================================================
# Run on LOGIN NODE (has internet) — NOT via sbatch.
# Usage:  bash setup_cpjku_paper_login.sh
# or in Claude Code: ! bash setup_cpjku_paper_login.sh
#
# What this does:
#   1. Downloads micromamba to /scratch (no root needed)
#   2. Creates a minimal conda env containing libfluidsynth.so
#   3. Installs pyfluidsynth into venv_cpjku310 (ctypes wrapper)
#   4. Verifies the import
#
# After this, SLURM scripts set:
#   export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:...
# to make pyfluidsynth find libfluidsynth at runtime.
# ============================================================

set -euo pipefail

SCRATCH=/scratch/pmohseni
MAMBA_BIN=$SCRATCH/micromamba_bin
MAMBA_ROOT=$SCRATCH/micromamba
VENV=$SCRATCH/venv_cpjku310
FLUID_LIB=$MAMBA_ROOT/envs/fluidsynth/lib

echo "=== Step 1: Download micromamba ==="
mkdir -p "$MAMBA_BIN"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$MAMBA_BIN" bin/micromamba
MAMBA="$MAMBA_BIN/bin/micromamba"
echo "micromamba: $($MAMBA --version)"

echo ""
echo "=== Step 2: Create minimal FluidSynth conda env ==="
"$MAMBA" create -y -n fluidsynth -c conda-forge fluidsynth --root-prefix "$MAMBA_ROOT"
echo "libfluidsynth.so location:"
ls "$FLUID_LIB"/libfluidsynth* 2>/dev/null || { echo "ERROR: libfluidsynth not found in $FLUID_LIB"; exit 1; }

echo ""
echo "=== Step 3: Install pyfluidsynth into venv_cpjku310 ==="
source "$VENV/bin/activate"
# pyfluidsynth is a ctypes wrapper — install only needs pip, runtime needs LD_LIBRARY_PATH
LD_LIBRARY_PATH="$FLUID_LIB:${LD_LIBRARY_PATH:-}" pip install pyfluidsynth

echo ""
echo "=== Step 4: Verify import ==="
LD_LIBRARY_PATH="$FLUID_LIB:${LD_LIBRARY_PATH:-}" python3 -c "
import fluidsynth
synth = fluidsynth.Synth()
synth.delete()
print('pyfluidsynth + libfluidsynth: OK')
"

echo ""
echo "============================================================"
echo "Setup complete."
echo "FluidSynth lib path: $FLUID_LIB"
echo "SLURM scripts will set LD_LIBRARY_PATH automatically."
echo "============================================================"
