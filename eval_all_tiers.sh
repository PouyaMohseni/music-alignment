#!/bin/bash
# Submit the SAME checkpoint against all three acoustic tiers, so every new
# model is reported on the full grid instead of whichever tier happened to be
# convenient.
#
#   bash eval_all_tiers.sh <EXPERIMENT> [WRAPPER|plain] [--after JOBID]
#
# WHY ALL THREE, ALWAYS. The 2026-08-03 sweep showed the tiers disagree so
# strongly that any single one is misleading on its own:
#
#   * synth   -- fluidsynth, same soundfont as training. In-domain. B1a is #1
#                here (90.0) and only #8 on room (38.5), so a synth win says
#                almost nothing about real audio.
#   * di-left -- real piano, DIRECT pickup. Real instrument, no room. This is
#                the tier that separates "survives a real piano" from
#                "survives a real ROOM" -- and the CBEncoder models WIN here
#                (62-69) while losing badly on room (15-28).
#   * room    -- real piano, room microphone. The hard case and the objective.
#                cyolo_sb = 63.0; our best = 44.7.
#
# Reporting only room would hide that MERT trades clean accuracy for
# robustness; reporting only synth would have promoted B6 (87.7 synth, 15.6
# room). The three-way spread IS the result.
#
# --after JOBID makes the evals depend on a training job, so a model is
# measured on the whole grid the moment it converges without anyone remembering
# to come back for it.

set -uo pipefail
EXP="${1:?usage: bash eval_all_tiers.sh <EXPERIMENT> [WRAPPER|plain] [--after JOBID]}"
WRAPPER="${2:-plain}"
DEP=""
if [ "${3:-}" = "--after" ] && [ -n "${4:-}" ]; then
    # afterANY, not afterok. The training jobs are 3h allocations that are
    # DESIGNED to hit their walltime and be resubmitted (they auto-resume from
    # latest_model.pt), so they terminate as TIMEOUT, not COMPLETED. With
    # afterok SLURM treats that as failure and CANCELS the dependent evals --
    # which is exactly what happened to 64849-64854: all six cancelled, zero
    # logs, while both checkpoints existed and were perfectly evaluable.
    DEP="--dependency=afterany:${4}"
    echo "chaining after job ${4} (afterany: TIMEOUT is the expected end state)"
fi

cd "$(dirname "$0")"
for TIER in synth di-left room; do
    JID=$(sbatch --parsable $DEP eval_any_cpu.sh "$EXP" "$TIER" "$WRAPPER" 2>/dev/null | tail -1)
    printf "  %-10s %-38s job %s\n" "$TIER" "$EXP" "${JID:-SUBMIT FAILED}"
done
