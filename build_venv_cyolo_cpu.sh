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
module load gcc python/3.10 opencv

SRC=/scratch/pmohseni/venv_cpjku310
DST=/scratch/pmohseni/venv_cyolo
rm -rf "$DST"
# `python -m venv --without-pip` left the venv with no pip at all, so every
# subsequent `pip install` was a no-op against the module-level pip and the
# venv ended up empty. virtualenv --no-download is the Alliance-supported way
# and seeds pip/setuptools/wheel from the local wheelhouse.
virtualenv --no-download "$DST"
source "$DST/bin/activate"
python -m pip --version || { echo "FATAL: venv has no pip"; exit 1; }
pip install --no-index --upgrade pip setuptools wheel 2>&1 | tail -1

echo "=== installing wheelhouse deps ==="
# madmom 0.16.1 uses the np.float/np.int aliases removed in numpy 1.24, so the
# ceiling is load-bearing; the exact 1.22.4 build is not in the py3.10
# wheelhouse, so pin the constraint rather than the version.
pip install --no-index "numpy<1.24" 2>&1 | tail -1
pip install --no-index torch==2.6.0 torchvision torchaudio 2>&1 | tail -1
pip install --no-index librosa scipy pyyaml tqdm soundfile matplotlib 2>&1 | tail -1

echo "=== copying source-built madmom 0.16.1 from venv_cpjku310 ==="
SP_SRC=$(ls -d $SRC/lib/python3.10/site-packages)
SP_DST=$(ls -d $DST/lib/python3.10/site-packages)
cp -r "$SP_SRC"/madmom "$SP_DST"/ 2>/dev/null
cp -r "$SP_SRC"/madmom-0.16.1.dist-info "$SP_DST"/ 2>/dev/null

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
