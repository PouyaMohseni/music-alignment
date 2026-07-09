"""D2 smoke test on REAL data (test pieces -- MERT + MIDI already exist).
Verifies: repeat detection finds real groups on a real piece, soft CE loss
degenerates to D1's plain CE when no repeats are present, full 3-term loss is
finite with gradient into audio tower / score tower / MIDI encoder, and that
eval.py's code path (D1's, re-exported) never imports or references MIDI."""
import sys, ast
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

import numpy as np
import torch
import yaml

from mymodel.d1_align_matrix.model import D1Model
from mymodel.d1_align_matrix.losses import dense_ce_loss, soft_dtw_matrix_loss
from mymodel.d2_midi_privileged.midi_encoder import MidiEncoder
from mymodel.d2_midi_privileged.losses import soft_multi_target_ce_loss, midi_distill_loss
from mymodel.d2_midi_privileged import data as d2data
from mymodel.d1_align_matrix.dtw import dtw_decode

cfg = yaml.safe_load(open('configs/d2_midi_privileged.yaml'))
dc = cfg['data']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, flush=True)

pieces = d2data.load_split('test', dc['processed_root'], dc['cpjku_data'],
                           dc['mert_roots'], dc['scale_factor'],
                           cfg['model']['w_downsample'], repeat_k=dc['repeat_k'], limit=5)
assert len(pieces) >= 1, 'FAIL: no test pieces loaded'
n_repeat_frames = [sum(1 for a in p.repeat_alt_cols if a) for p in pieces]
print('onset frames with >=1 repeat alternate, per piece:', n_repeat_frames, flush=True)
assert any(n > 0 for n in n_repeat_frames), 'FAIL: repeat detector found ZERO repeats across 5 real pieces (suspicious)'

p = next(pc for pc, n in zip(pieces, n_repeat_frames) if n > 0)
print(f'using piece={p.piece_name} with {sum(1 for a in p.repeat_alt_cols if a)} repeat-ambiguous onsets', flush=True)

model = D1Model(**cfg['model']).to(device)
midi_encoder = MidiEncoder(d_model=cfg['model']['d_model']).to(device)

A, B = model.encode(p.mert.to(device), p.strip.to(device))
S = model.similarity(A, B)
W_col = S.shape[1]
gt = p.gt_cols.to(device).clamp(max=W_col - 1)

# --- degeneracy check: soft CE with all-empty alt lists == D1's plain CE ---
empty_alts = [[] for _ in range(S.shape[0])]
ce_plain = dense_ce_loss(S, gt, sigma_cols=cfg['loss']['ce_sigma_cols'])
ce_soft_empty = soft_multi_target_ce_loss(S, gt, repeat_alt_cols=empty_alts,
                                          sigma_cols=cfg['loss']['ce_sigma_cols'],
                                          alt_weight=cfg['loss']['repeat_alt_weight'])
print(f'ce_plain={float(ce_plain):.6f}  ce_soft(empty alts)={float(ce_soft_empty):.6f}', flush=True)
assert abs(float(ce_plain) - float(ce_soft_empty)) < 1e-4, 'FAIL: soft CE does not degenerate to plain CE with no repeats'

# --- real repeat-aware CE (with actual alternates for this piece) ---
alt_cols_clamped = [[min(c, W_col - 1) for c in alts] for alts in p.repeat_alt_cols]
ce_soft_real = soft_multi_target_ce_loss(S, gt, repeat_alt_cols=alt_cols_clamped,
                                         sigma_cols=cfg['loss']['ce_sigma_cols'],
                                         alt_weight=cfg['loss']['repeat_alt_weight'])
print(f'ce_soft(real alts)={float(ce_soft_real):.6f}  (should differ from plain: '
      f'{abs(float(ce_soft_real)-float(ce_plain)) > 1e-4})', flush=True)
assert np.isfinite(float(ce_soft_real))
assert abs(float(ce_soft_real) - float(ce_plain)) > 1e-4, \
    'FAIL: repeat-aware CE identical to plain CE despite real alternates present'

# --- full 3-term loss + gradient flow (incl. MIDI encoder) ---
dtw = soft_dtw_matrix_loss(S, gamma=cfg['loss']['dtw_gamma'],
                           max_t=cfg['loss']['dtw_max_t'], max_w=cfg['loss']['dtw_max_w'])
T = A.shape[0]
sub = min(T, 300)
idx = np.random.default_rng(0).choice(T, size=sub, replace=False)
idx_t = torch.from_numpy(idx).to(device)
M = midi_encoder(p.pitch_roll.to(device)[idx_t])
distill = midi_distill_loss(A[idx_t], M, temperature=cfg['loss']['distill_temperature'])
print(f'dtw={float(dtw):.4f}  distill={float(distill):.4f}', flush=True)
assert np.isfinite(float(dtw)) and np.isfinite(float(distill))

loss = ce_soft_real + cfg['loss']['dtw_weight'] * dtw + cfg['loss']['distill_weight'] * distill
loss.backward()

def gsum(module):
    return sum(float(pr.grad.abs().sum()) for pr in module.parameters() if pr.grad is not None)
ga, gsc, gm = gsum(model.audio_tower), gsum(model.score_tower), gsum(midi_encoder)
print(f'grad abs-sum: audio_tower={ga:.4f}  score_tower={gsc:.4f}  midi_encoder={gm:.4f}', flush=True)
assert ga > 0, 'FAIL: no gradient into audio tower'
assert gsc > 0, 'FAIL: no gradient into score tower'
assert gm > 0, 'FAIL: no gradient into MIDI encoder'

# --- decode sanity (unchanged from D1) ---
S_np = S.detach().float().cpu().numpy()
path = dtw_decode(S_np, band_frac=0.15)
assert np.all(np.diff(path) >= 0), 'FAIL: decode path not monotonic'
print(f'DTW path monotonic, range=[{path.min()},{path.max()}]', flush=True)

# --- static check: eval.py never IMPORTS anything MIDI-related (checking
# actual imports via ast, not a raw string search -- docstrings legitimately
# mention "MIDI" when explaining why it's absent from the code path) ---
for path in ['mymodel/d1_align_matrix/eval.py', 'mymodel/d2_midi_privileged/eval.py']:
    tree = ast.parse(open(path).read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module)
    bad = [n for n in names if n and 'midi' in n.lower()]
    assert not bad, f'FAIL: {path} imports something MIDI-related: {bad}'
print('Confirmed: neither eval.py imports anything MIDI-related.', flush=True)

print('\nSMOKE TEST PASSED', flush=True)
