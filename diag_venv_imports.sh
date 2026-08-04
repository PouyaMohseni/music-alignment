#!/bin/bash
#SBATCH --job-name=diag-venv
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/diag_venv-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/diag_venv-%j.log

# Stop guessing why the MERT precompute hangs. Two wrong diagnoses so far
# (/project inode quota; missing module load) were both falsified by a pilot
# that hung anyway. Get a STACK TRACE instead of another hypothesis.
#
# faulthandler.dump_traceback_later(N, exit=True) fires from a watchdog thread
# and prints where every thread actually is, which a plain `timeout` cannot.
# Each import is timed separately so the culprit is unambiguous.

set -uo pipefail
echo "host=$(hostname) $(date)"
module load gcc opencv
echo "modules: $(module list 2>&1 | tr '\n' ' ' | head -c 300)"

probe () {
  local name="$1" py="$2"
  echo ""
  echo "################ $name ################"
  echo "python: $py"
  "$py" -c "
import faulthandler, sys, time
faulthandler.dump_traceback_later(90, exit=True)   # hard stack dump if wedged
print('exe    :', sys.executable, flush=True)
print('version:', sys.version.split()[0], flush=True)
for mod in ('numpy','torch','transformers','librosa','soundfile'):
    t=time.time()
    try:
        __import__(mod)
        print(f'  OK   {mod:<13} {time.time()-t:6.1f}s', flush=True)
    except Exception as e:
        print(f'  FAIL {mod:<13} {type(e).__name__}: {str(e)[:160]}', flush=True)
print('ALL IMPORTS COMPLETED', flush=True)
" 2>&1 | sed 's/^/    /'
  echo "    exit=$?"
}

probe "A: .venv (py3.11, /project)"            /project/def-ichiro/pmohseni/music-alignment/.venv/bin/python
probe "B: music-alignment-venv (py3.11, /scratch)" /scratch/pmohseni/music-alignment-venv/bin/python
probe "C: venv_cpjku310 (py3.10, /scratch) -- the one that WORKS in eval jobs" /scratch/pmohseni/venv_cpjku310/bin/python

echo ""
echo "done $(date)"
