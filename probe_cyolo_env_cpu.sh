#!/bin/bash
#SBATCH --job-name=probe-cyolo-env
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:25:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/probe_cyolo_env-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/probe_cyolo_env-%j.log
# Can venv_cpjku310 (py3.10, REAL madmom) run CPJKU's CYOLO instead of building
# the py3.7 env from environment.yml? venv_cpjku310 already exists for the
# madmom-dependent CPJKU eval path, so if it works we skip an env build.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:${PYTHONPATH:-}
python - <<'PY'
import importlib, sys
print('python', sys.version.split()[0])
for m in ['torch','numpy','cv2','madmom','librosa','scipy','yaml','tqdm','soundfile','torchvision','torchaudio']:
    try:
        mod=importlib.import_module(m)
        print(f'  OK   {m:12s} {getattr(mod,"__version__","?")}')
    except Exception as e:
        print(f'  MISS {m:12s} {type(e).__name__}: {str(e)[:60]}')
print('--- CYOLO package imports ---')
for m in ['cyolo_score_following','cyolo_score_following.models.yolo',
          'cyolo_score_following.dataset','cyolo_score_following.utils.general']:
    try:
        importlib.import_module(m); print(f'  OK   {m}')
    except Exception as e:
        print(f'  FAIL {m} -> {type(e).__name__}: {str(e)[:110]}')
PY
echo "Job finished at $(date)"
