#!/bin/bash
#SBATCH --job-name=check-n3-gate
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/check_n3_gate-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/check_n3_gate-%j.log
# N3 scored 89.3 vs B1a's 89.2. Because N3 is warm-started from B1a and its gate
# is zero-initialised, it computes EXACTLY B1a while gate==0 -- so 89.3 could
# mean "the filter learned something mildly useful" OR "the gate never opened
# and this is just B1a again". Those demand opposite next actions, so read the
# learned parameters rather than inferring from the score.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
python - <<'PY'
import glob, torch, numpy as np
for exp, keys in [('N3_belief_propagation', ['belief_filter.gate','belief_filter.jump_logit',
                                             'belief_filter.evidence_scale','belief_filter.kernel_logits']),
                  ('N2_memory_retrieval',   ['mem_read.gate','mem_read.out_proj.weight'])]:
    cks = sorted(glob.glob(f'results/cb_ta_ext/{exp}/params/*/best_model.pt'))
    if not cks: print(f'{exp}: no checkpoint'); continue
    sd = torch.load(cks[-1], map_location='cpu')
    print(f'\n=== {exp} ({cks[-1].split("/")[-2]}) ===')
    for k in keys:
        if k not in sd: print(f'  {k}: ABSENT'); continue
        v = sd[k].float()
        print(f'  {k:34s} mean={v.mean():+.4e}  absmax={v.abs().max():.4e}  ' +
              ('<-- still ~ZERO: branch inactive' if v.abs().max() < 1e-6 else '<-- NONZERO: branch active'))
    if 'belief_filter.jump_logit' in sd:
        j = torch.sigmoid(sd['belief_filter.jump_logit'].float()).item()
        print(f'  uniform escape floor j = {j:.4f} (init 0.100)')
PY
echo "Job finished"
