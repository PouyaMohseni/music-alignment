"""Train the scorer WITHOUT any feature computed from the tracker's history.

The paper's analysis says the features that reduce error propagation are the
ones that do not depend on the previous position. The direct test is a scorer
that never sees the history at all: displacement, backward and staff-change
flags, vertical move and the four velocity features are removed, while the
decoder still blends in the zero-parameter prior (which is where the history
then enters, and only there).

Removal is done through the scorer's own normalisation buffers: sd = 1e9 turns
those inputs into zeros during training AND at inference, since the buffers
ship inside the checkpoint. Every other argument is train_cand_scorer.py's.
"""
import runpy
import sys

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.heads import cand_scorer

# d_norm, d_tanh50, d_tanh200, d_is_back, dy_tanh, same_staff,
# d_extrap, d_extrap_tanh, v_ratio, has_vel
HISTORY_COLS = [7, 8, 9, 10, 12, 13, 20, 21, 22, 23]

_set_norm = cand_scorer.CandScorer.set_norm


def set_norm(self, mu, sd):
    _set_norm(self, mu, sd)
    self.sd[HISTORY_COLS] = 1e9
    print(f'[nohist] history features divided out: columns {HISTORY_COLS}', flush=True)


cand_scorer.CandScorer.set_norm = set_norm
sys.argv[0] = '/project/def-ichiro/pmohseni/music-alignment/extensions/analysis/train_cand_scorer.py'
runpy.run_path(sys.argv[0], run_name='__main__')
