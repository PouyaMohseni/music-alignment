import sys
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

import numpy as np
import torch

from mymodel.d2_midi_privileged.data_multitempo import load_piece_multitempo
from mymodel.d1_align_matrix.model import D1Model
from mymodel.d1_align_matrix.losses import dense_ce_loss

CPJKU_DATA = 'data/MSMD/cpjku_fmt'
MERT_ROOTS = ['/scratch/pmohseni/mert_emb_zenodo/cpjku_fmt_test_eval']
MERT_ROOTS_MT = ['/tmp/e4_multitempo_mert']
RENDER_DIR = '/tmp/e4_multitempo_test'

piece = 'AndreJ__O34__andre-sonatine'

p1000 = load_piece_multitempo(piece, CPJKU_DATA, MERT_ROOTS, MERT_ROOTS_MT, RENDER_DIR,
                              scale_factor=3, w_downsample=4, tempo_factor=1000)
p750 = load_piece_multitempo(piece, CPJKU_DATA, MERT_ROOTS, MERT_ROOTS_MT, RENDER_DIR,
                             scale_factor=3, w_downsample=4, tempo_factor=750)
p1250 = load_piece_multitempo(piece, CPJKU_DATA, MERT_ROOTS, MERT_ROOTS_MT, RENDER_DIR,
                              scale_factor=3, w_downsample=4, tempo_factor=1250)

assert p1000 is not None and p750 is not None and p1250 is not None, 'FAIL: a tempo variant failed to load'

for tag, p in [('1000', p1000), ('750', p750), ('1250', p1250)]:
    print(f'tempo_{tag}: T={p.mert.shape[0]} n_onsets={len(p.onset_frames)} '
          f'last_onset_frame={p.onset_frames[-1] if len(p.onset_frames) else None} '
          f'n_repeat_alt={sum(1 for a in p.repeat_alt_cols if a)}', flush=True)

# --- sanity: last onset frame should scale roughly by tempo ratio, and be < T ---
last_1000 = p1000.onset_frames[-1]
last_750 = p750.onset_frames[-1]
last_1250 = p1250.onset_frames[-1]
ratio_750 = last_750 / last_1000
ratio_1250 = last_1250 / last_1000
print(f'last-onset-frame ratios: tempo_750/1000={ratio_750:.3f} (expect ~0.75)  '
      f'tempo_1250/1000={ratio_1250:.3f} (expect ~1.25)', flush=True)
assert 0.65 < ratio_750 < 0.85, f'FAIL: tempo_750 onset-frame ratio {ratio_750} not close to 0.75'
assert 1.1 < ratio_1250 < 1.4, f'FAIL: tempo_1250 onset-frame ratio {ratio_1250} not close to 1.25'
assert last_750 < p750.mert.shape[0], 'FAIL: last onset frame exceeds T at tempo_750'
assert last_1250 < p1250.mert.shape[0], 'FAIL: last onset frame exceeds T at tempo_1250'

# --- pitch_roll sanity: should be nonzero and roughly track note density ---
for tag, p in [('1000', p1000), ('750', p750), ('1250', p1250)]:
    frac_active = float(p.pitch_roll.sum() > 0)
    print(f'tempo_{tag}: pitch_roll nonzero-frame-frac={float((p.pitch_roll.sum(dim=1) > 0).float().mean()):.3f}',
          flush=True)

# --- repeat_alt_cols should be IDENTICAL in content (same score, same repeats)
# just at different frame positions ---
n_alt_1000 = sum(len(a) for a in p1000.repeat_alt_cols if a)
n_alt_750 = sum(len(a) for a in p750.repeat_alt_cols if a)
print(f'total repeat-alternate entries: tempo_1000={n_alt_1000} tempo_750={n_alt_750} '
      f'(should match -- same score, tempo-independent repeat structure)', flush=True)
assert n_alt_1000 == n_alt_750, 'FAIL: repeat-group count differs across tempo (should be tempo-independent)'

# --- one real training step combining a tempo_1000 piece and a tempo_750 piece ---
import yaml
cfg = yaml.safe_load(open('configs/d2_midi_privileged.yaml'))
model = D1Model(**cfg['model'])
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for p in [p1000, p750]:
    S = model(p.mert, p.strip)
    gt = p.gt_cols.clamp(max=S.shape[1] - 1)
    loss = dense_ce_loss(S, gt, sigma_cols=3.0)
    assert np.isfinite(float(loss)), f'FAIL: non-finite loss for a piece'
    (loss / 2).backward()
opt.step()
print('Combined tempo_1000 + tempo_750 training step: OK, both losses finite, backward succeeded', flush=True)

print('\nSMOKE TEST PASSED', flush=True)
