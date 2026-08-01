#!/bin/bash
#SBATCH --job-name=build-venv-cyolo
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/build_venv_cyolo-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/build_venv_cyolo-%j.log

# Dedicated venv for CPJKU's CYOLO (Henkel & Widmer 2021), whose released
# checkpoints are the real SOTA baseline: 70.6% on MSMD-Rec vs the CUNet
# family's published 12.5%.
#
# Why a NEW venv rather than extending venv_cpjku310:
#  * venv_cpjku310 is missing librosa/torchvision/torchaudio, but is in active
#    use by ~9 running eval jobs. Installing into it mid-flight risks breaking
#    them, and dependency resolution could silently move torch or numpy.
#  * .venv has librosa but no madmom.
#  * madmom is NOT in the Alliance wheelhouse (`pip download --no-index madmom`
#    -> no matching distribution), so it cannot simply be reinstalled. It was
#    source-built into venv_cpjku310 as madmom 0.16.1 -- exactly the version
#    CYOLO's environment.yml pins -- so the compiled package is copied across.
#    Both venvs are py3.10 on the same architecture, so the built extensions
#    are ABI-compatible; the copy is verified by importing madmom afterwards.
#
# torchvision 0.21.0 / torchaudio 2.6.0 are the wheelhouse builds matching
# torch 2.6.0, so no torch upgrade is pulled in.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
module load gcc python/3.10 opencv/4.10.0

SRC=/scratch/pmohseni/venv_cpjku310
DST=/scratch/pmohseni/venv_cyolo
rm -rf "$DST"
# `python -m venv --without-pip` left the venv with no pip at all, so every
# subsequent `pip install` was a no-op against the module-level pip and the
# venv ended up empty. virtualenv --no-download is the Alliance-supported way
# and seeds pip/setuptools/wheel from the local wheelhouse.
# Pin the EXACT 3.10 interpreter venv_cpjku310 was built from. `module load
# python/3.10` silently resolves to 3.11 in batch context here, which is how
# the first build produced a py3.11 venv -- and madmom is a COMPILED package
# built for cp310, so its ABI does not transfer and the copy target
# lib/python3.10/site-packages did not even exist.
PY310=$(grep -E "^home" /scratch/pmohseni/venv_cpjku310/pyvenv.cfg | awk '{print $3}')/python3.10
echo "base interpreter: $PY310 -> $($PY310 --version 2>&1)"
"$PY310" --version | grep -q "3\.10" || { echo "FATAL: base interpreter is not 3.10"; exit 1; }
"$PY310" -m venv "$DST"
source "$DST/bin/activate"
python -m pip --version || { echo "FATAL: venv has no pip"; exit 1; }
# madmom 0.16.1 does `import pkg_resources`, which setuptools REMOVED in 81.
# Upgrading setuptools unpinned is what broke the previous build.
pip install --no-index --upgrade pip wheel 2>&1 | tail -1
pip install --no-index "setuptools<81" 2>&1 | tail -1
python -c "import pkg_resources; print('pkg_resources OK')" || { echo "FATAL: no pkg_resources"; exit 1; }

echo "=== installing wheelhouse deps ==="
# madmom 0.16.1 uses the np.float/np.int aliases removed in numpy 1.24, so the
# ceiling is load-bearing; the exact 1.22.4 build is not in the py3.10
# wheelhouse, so pin the constraint rather than the version.
# A pinned first install is NOT enough: every later pip call is free to
# resolve numpy upward again, and that is what re-broke madmom
# ("numpy has no attribute 'float'" -- the aliases removed in 1.24). Use a
# CONSTRAINTS file honoured by all subsequent installs.
CONSTRAINTS=$DST/constraints.txt
echo "numpy<1.24" > "$CONSTRAINTS"
pip install --no-index -c "$CONSTRAINTS" "numpy<1.24" 2>&1 | tail -1
pip install --no-index -c "$CONSTRAINTS" torch==2.6.0 torchvision torchaudio 2>&1 | tail -1
pip install --no-index -c "$CONSTRAINTS" librosa scipy pyyaml tqdm soundfile matplotlib 2>&1 | tail -1
# madmom 0.16.1 Requires-Dist: numpy>=1.13.4, scipy>=0.16, cython>=0.25, mido>=1.2.8
pip install --no-index -c "$CONSTRAINTS" mido cython 2>&1 | tail -1

echo "=== copying source-built madmom 0.16.1 from venv_cpjku310 ==="
SP_SRC=$(ls -d $SRC/lib/python3.10/site-packages)
SP_DST=$(ls -d $DST/lib/python3.10/site-packages)
echo "  src=$SP_SRC"; echo "  dst=$SP_DST"
[ -d "$SP_DST" ] || { echo "FATAL: no site-packages at $SP_DST"; exit 1; }
cp -r "$SP_SRC"/madmom "$SP_DST"/            # no 2>/dev/null: a silent cp
cp -r "$SP_SRC"/madmom-0.16.1.dist-info "$SP_DST"/   # failure hid the last bug

echo "=== numpy must still be <1.24 after ALL installs ==="
python -c "import numpy,sys; print('numpy',numpy.__version__); sys.exit(0 if tuple(map(int,numpy.__version__.split('.')[:2]))<(1,24) else 1)" \
  || { echo "FATAL: numpy drifted >=1.24; madmom will break"; exit 1; }

echo "=== verify ==="
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:${PYTHONPATH:-}
python - <<'PY'
import importlib, sys
print('python', sys.version.split()[0])
ok=True
for m in ['numpy','torch','torchvision','torchaudio','librosa','scipy','yaml','tqdm','soundfile','cv2','madmom']:
    try:
        mod=importlib.import_module(m); print(f'  OK   {m:12s} {getattr(mod,"__version__","?")}')
    except Exception as e:
        print(f'  MISS {m:12s} {type(e).__name__}: {str(e)[:60]}'); ok=False
print('--- CYOLO package ---')
for m in ['cyolo_score_following.models.yolo','cyolo_score_following.dataset',
          'cyolo_score_following.utils.general']:
    try: importlib.import_module(m); print(f'  OK   {m}')
    except Exception as e: print(f'  FAIL {m} -> {type(e).__name__}: {str(e)[:110]}'); ok=False
print('RESULT:', 'READY' if ok else 'INCOMPLETE')
PY
echo "Job finished at $(date)"
