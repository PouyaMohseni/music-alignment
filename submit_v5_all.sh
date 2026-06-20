#!/bin/bash
# Submit all v5 training variants at once.
# Usage:  bash submit_v5_all.sh
# After:  bash eval_v5_all.sh

cd /project/def-ichiro/pmohseni/music-alignment

VARIANTS=(v5b_large v5c_noxattn v5d_long v5e_scratch v5f_bidir v5g_residual v5h_deep v5i_bidir_residual)

for V in "${VARIANTS[@]}"; do
  mkdir -p results/$V
  JID=$(sbatch --parsable train_${V}.sh)
  echo "submitted $V  -> job $JID"
done

echo ""
echo "All 8 jobs submitted. Monitor with:  squeue -u $USER"
echo "Eval tomorrow with:  bash eval_v5_all.sh"
