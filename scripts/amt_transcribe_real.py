"""Transcribe the 25 MSMD real-performance pages with two piano-AMT checkpoints.

WHY THIS EXISTS
---------------
Our score-follower collapses from ~89 pct@0.5s on synthetic audio to ~46 on the
real room mic.  The AMT-bridge hypothesis is that automatic piano transcription
is domain-invariant enough to act as a stable intermediate representation:
audio -> notes -> score position, sidestepping the synth->real acoustic gap.
Literature reports ~88 note-F1 for a reverb-augmented AMT model on real
Disklavier room recordings, but we refuse to extrapolate from someone else's
test set.  This script measures it on OUR audio.

WHAT MAKES THIS MEASURABLE
--------------------------
third_party/cpjku_unet/data/msmd/msmd_real_performances/ holds, per page:
    performance/{piece}_room.wav      real Yamaha through a ROOM mic
    performance/{piece}_di-left.wav   SAME take through a DIRECT pickup
    performance/{piece}.mid           the MIDI that was played back  == ground truth
Because both wavs are the same physical take through two microphones, the
room-vs-di-left delta isolates *acoustics* with performance held exactly fixed.

MODELS
------
  kong_stock      Kong et al. 2021 high-resolution CRNN, no augmentation.
                  zenodo 4034264, checkpoint dict has {'note_model','pedal_model'}
                  -> model_type 'Note_pedal'.
  edwards_robust  Edwards et al., IEEE SPL 2024 (arXiv:2402.01424), the SAME
                  architecture retrained with reverb/noise/codec augmentation.
                  zenodo 10610212, flat state dict with NO pedal head
                  -> model_type 'Regress_onset_offset_frame_velocity_CRNN'.
The stock-vs-robust contrast is the direct measurement of what the reverb
augmentation buys on our audio.

OUTPUT
------
/scratch/pmohseni/amt_out/{model}/{tier}/{piece}.json
    {"onset": [...], "offset": [...], "pitch": [...], "velocity": [...]}
Times in seconds, pitch in MIDI note numbers.  Scoring lives in
scripts/amt_bridge_eval.py (separate venv: it needs real madmom to reproduce
the CPJKU note indexing bit-for-bit).

RUN
---
    sbatch amt_transcribe_gpu.sh          # or: python scripts/amt_transcribe_real.py --device cpu
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(REPO, 'third_party/cpjku_unet/data/msmd/msmd_real_performances')
CKPTS = {
    # model_key: (checkpoint path, piano_transcription_inference model_type)
    'kong_stock': ('/scratch/pmohseni/amt_ckpts/kong_stock.pth', 'Note_pedal'),
    'edwards_robust': ('/scratch/pmohseni/amt_ckpts/edwards_robust.pth',
                       'Regress_onset_offset_frame_velocity_CRNN'),
}
TIERS = ('room', 'di-left')


def piece_list():
    import yaml
    with open(os.path.join(REC, 'rp_split.yaml'), 'rb') as fp:
        return yaml.safe_load(fp)['files']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default='/scratch/pmohseni/amt_out')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--models', default='kong_stock,edwards_robust')
    ap.add_argument('--tiers', default='room,di-left')
    args = ap.parse_args()

    t0 = time.time()
    import torch
    print(f'[{time.time()-t0:.0f}s] torch {torch.__version__} imported', flush=True)
    import librosa
    from piano_transcription_inference import PianoTranscription, sample_rate
    print(f'[{time.time()-t0:.0f}s] piano_transcription_inference imported '
          f'(sample_rate={sample_rate})', flush=True)

    dev = args.device
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    print(f'device = {device}', flush=True)

    pieces = piece_list()
    print(f'{len(pieces)} pieces', flush=True)

    for model_key in args.models.split(','):
        ckpt, model_type = CKPTS[model_key]
        assert os.path.exists(ckpt), ckpt
        print(f'\n===== {model_key} ({model_type}) =====', flush=True)
        # NOTE: PianoTranscription re-downloads if the file is < 1.6e8 bytes, which
        # would clobber the (smaller, pedal-free) Edwards checkpoint with Kong's.
        # Guard by asserting the file we intend to use is the one that gets loaded.
        import piano_transcription_inference.inference as _inf
        _real_getsize = _inf.os.path.getsize
        size = _real_getsize(ckpt)

        def _guarded_getsize(p, _c=os.path.abspath(ckpt), _r=_real_getsize):
            # Only lie about OUR checkpoint, and only to clear their size gate.
            return 2e8 if os.path.abspath(p) == _c else _r(p)

        _inf.os.path.getsize = _guarded_getsize
        try:
            tr = PianoTranscription(model_type=model_type, checkpoint_path=ckpt,
                                    device=device)
        finally:
            _inf.os.path.getsize = _real_getsize
        assert _real_getsize(ckpt) == size, 'checkpoint was overwritten by autodownload!'
        # Sanity: their load_state_dict uses strict=False, so a wholesale key
        # mismatch would silently leave a RANDOMLY INITIALISED net and we would
        # report its garbage transcriptions as a measurement. Verify every model
        # tensor was actually populated from the checkpoint.
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)['model']
        if model_type == 'Note_pedal':
            ck = {f'{pre}.{k}': v for pre in ('note_model', 'pedal_model')
                  for k, v in ck[pre].items()}
        model = tr.model.module if hasattr(tr.model, 'module') else tr.model
        msd = model.state_dict()
        matched = [k for k in msd if k in ck and msd[k].shape == ck[k].shape]
        print(f'weights: model={len(msd)} ckpt={len(ck)} matched={len(matched)}',
              flush=True)
        assert len(matched) == len(msd), (
            f'{len(msd) - len(matched)} model tensors NOT loaded from {ckpt}')
        del ck, msd

        for tier in args.tiers.split(','):
            outdir = os.path.join(args.out_root, model_key, tier)
            os.makedirs(outdir, exist_ok=True)
            for i, piece in enumerate(pieces):
                out = os.path.join(outdir, piece + '.json')
                if os.path.exists(out):
                    print(f'  [{i+1}/{len(pieces)}] skip {piece}', flush=True)
                    continue
                wav = os.path.join(REC, 'performance', f'{piece}_{tier}.wav')
                audio, _ = librosa.load(wav, sr=sample_rate, mono=True)
                t1 = time.time()
                res = tr.transcribe(audio, None)
                ev = res['est_note_events']
                rec = {
                    'onset': [float(e['onset_time']) for e in ev],
                    'offset': [float(e['offset_time']) for e in ev],
                    'pitch': [int(e['midi_note']) for e in ev],
                    'velocity': [int(e['velocity']) for e in ev],
                    'audio_seconds': float(len(audio) / sample_rate),
                    'model': model_key, 'tier': tier, 'piece': piece,
                }
                with open(out, 'w') as f:
                    json.dump(rec, f)
                print(f'  [{i+1}/{len(pieces)}] {tier} {piece}: '
                      f'{len(ev)} notes, {len(audio)/sample_rate:.1f}s audio, '
                      f'{time.time()-t1:.1f}s wall', flush=True)

    print(f'\nDONE in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
