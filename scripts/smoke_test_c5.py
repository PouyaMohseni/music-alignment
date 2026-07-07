import sys, json, math
from pathlib import Path

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
cpjku_root = Path('third_party/cpjku_unet').resolve()
sys.path.insert(0, str(cpjku_root))

from mymodel.cpjku_adapter import madmom_compat
madmom_compat.patch()

import torch
from audio_conditioned_unet.network import ConditionalUNet
from mymodel.cpjku_adapter.eval_official import _patched_load_piece
from extensions.decode.test_time_calibration import calibrate_and_infer_piece

config = {
    'spectrogram_params': {'sample_rate': 22050, 'frame_size': 2048, 'fps': 20, 'pad': 40},
    'gt_width': 10, 'real_perf': True, 'tempo_factors': [1000], 'sf_path': '',
}

param_path = cpjku_root / 'models' / 'CB_TA' / 'best_model.pt'
config_path = cpjku_root / 'models' / 'CB_TA' / 'net_config.json'
net_config = json.load(open(config_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = ConditionalUNet(net_config)
network.load_state_dict(torch.load(param_path, map_location='cpu'))
network.to(device).eval()
n_frames = network.perf_encoder.n_input_frames
print('Loaded CB_TA:', sum(p.numel() for p in network.parameters()), 'params on', device, flush=True)

CPJKU_FMT = 'data/MSMD/cpjku_fmt'
# Run piece A, then piece B, then piece A AGAIN -- the 3rd run's calibration
# should start from (near-)identical initial loss to the 1st run's if
# conv_out is genuinely being reset between pieces and not accumulating.
pieces = ['AndreJ__O34__andre-sonatine', 'Anonymous__lanative__lanative',
          'AndreJ__O34__andre-sonatine']

import copy
orig_conv_out = copy.deepcopy(network.conv_out.state_dict())


def conv_out_matches(sd_a, sd_b):
    return all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


init_losses = []

for idx, piece_name in enumerate(pieces):
    print(f'\n--- run {idx}: {piece_name} ---', flush=True)
    assert conv_out_matches(network.conv_out.state_dict(), orig_conv_out), \
        'FAIL: conv_out did not reset to original weights before this piece!'
    print('  conv_out matches pre-calibration snapshot: OK', flush=True)

    params = {'i': idx, 'path': CPJKU_FMT, 'piece_name': piece_name,
              'spectrogram_params': config['spectrogram_params'], 'scale_factor': 3}
    _, score, _, perf_dict = _patched_load_piece(params)
    perf = perf_dict[list(perf_dict.keys())[0]]
    onsets_set = set(perf['onsets'].tolist())
    # perf['add_per_staff'] is [staff_coords, add_per_staff_array] -- must
    # unpack before use, matching eval_official.py's own convention (passing
    # the raw 2-tuple through made add_per_staff[0] resolve to staff_coords,
    # a list, not the scalar offset -- confirmed via TypeError on first run).
    _staff_coords, add_per_staff = perf['add_per_staff']

    # Small calib/eval windows here purely for smoke-test wall-clock speed on
    # CPU (full strip width x many frames is expensive) -- the real eval
    # script (eval_test_time_calibration.py) uses the intended defaults
    # (calib_seconds=8.0, num_steps=15, full-piece eval).
    diffs, calib_init_loss, calib_final_loss = calibrate_and_infer_piece(
        network, score, perf['spec'], perf['interpol_fnc'], perf['interpol_c2o'],
        add_per_staff, onsets_set,
        pad=config['spectrogram_params']['pad'], gt_width=config['gt_width'],
        n_frames=n_frames, device=device,
        calib_seconds=2.0, fps=20, num_steps=5, lr=1e-3, seq_len=8, threshold=0.5,
        max_eval_seconds=4.0)

    print(f'  calib_loss: {calib_init_loss:.4f} -> {calib_final_loss:.4f} '
          f'({"decreased" if calib_final_loss < calib_init_loss else "DID NOT DECREASE"})',
          flush=True)
    assert calib_final_loss < calib_init_loss, 'FAIL: calibration loss did not decrease'

    print(f'  eval-segment onset frames scored: {len(diffs)}', flush=True)
    assert len(diffs) > 0, 'FAIL: no eval-segment onsets scored'
    assert all(math.isfinite(d) for d in diffs), 'FAIL: non-finite frame diff'
    print(f'  eval-segment frame_diffs (frames): min={min(diffs):.2f} '
          f'max={max(diffs):.2f} mean={sum(diffs)/len(diffs):.2f}', flush=True)

    init_losses.append(calib_init_loss)

assert conv_out_matches(network.conv_out.state_dict(), orig_conv_out), \
    'FAIL: conv_out not reset after final piece'

rel_diff = abs(init_losses[0] - init_losses[2]) / max(init_losses[0], 1e-8)
print(f'\nRun 0 initial loss: {init_losses[0]:.4f}   Run 2 (same piece) initial loss: '
      f'{init_losses[2]:.4f}   relative diff: {rel_diff:.4f}', flush=True)
assert rel_diff < 1e-4, ('FAIL: re-running the SAME piece gave a different starting '
                         'calibration loss -- weights are leaking across pieces!')
print('Reset-between-pieces confirmed: identical piece gives identical starting loss.',
      flush=True)

print('\nSMOKE TEST PASSED', flush=True)
