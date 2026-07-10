"""E4 -- render whole-piece MIDI (cpjku_fmt/performance/<piece>.mid, tempo_1000
by convention) at additional tempo factors, for D2 multi-tempo training data.

No whole-piece multi-tempo MIDI exists anywhere in this project (only the
per-PAGE multi-tempo MIDI under /scratch/pmohseni/msmd_train_full/performance,
a different dataset layout used by B-extensions/A0). Tempo-scaling convention
(verified in extensions/pretrain/tempo_contrastive.py against real per-page
multi-tempo renders): onset TIME scales by tempo_factor/1000 relative to the
tempo_1000 base (tempo_500 = all events at 0.5x the time = twice as fast).
Applied here identically to a whole-piece MIDI's note start/end times.

    python scripts/render_multitempo_wholepiece.py \
        --performance_dir data/MSMD/cpjku_fmt/performance \
        --out_dir /scratch/pmohseni/cpjku_fmt_multitempo \
        --tempo_factors 750 1250 \
        --pieces_file /tmp/pieces_subset.txt   # optional: limit to N pieces
        --sound_font third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2 \
        --fluidsynth /scratch/pmohseni/micromamba/envs/fluidsynth/bin/fluidsynth
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

import pretty_midi


def scale_midi(src_path: str, tempo_factor: int, out_path: str):
    """Writes a new MIDI with all note/CC times scaled by tempo_factor/1000."""
    m = pretty_midi.PrettyMIDI(str(src_path))
    ratio = tempo_factor / 1000.0
    out = pretty_midi.PrettyMIDI(initial_tempo=120.0 * (1000.0 / tempo_factor))
    for inst in m.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in inst.notes:
            new_inst.notes.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch, start=n.start * ratio, end=n.end * ratio))
        for cc in inst.control_changes:
            new_inst.control_changes.append(pretty_midi.ControlChange(
                number=cc.number, value=cc.value, time=cc.time * ratio))
        out.instruments.append(new_inst)
    out.write(str(out_path))


def render_wav(midi_path: str, sound_font: str, fluidsynth_bin: str, out_wav: str):
    cmd = [fluidsynth_bin, '-R', '0', '-C', '0', '-F', out_wav, '-O', 's16', '-T', 'wav',
           sound_font, midi_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--performance_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--tempo_factors', type=int, nargs='+', required=True)
    p.add_argument('--pieces_file', default=None, help='optional: text file, one piece name per line')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--sound_font', required=True)
    p.add_argument('--fluidsynth', required=True)
    a = p.parse_args()

    perf_dir = Path(a.performance_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.pieces_file:
        pieces = [l.strip() for l in open(a.pieces_file) if l.strip()]
    else:
        pieces = sorted(f.stem for f in perf_dir.glob('*.mid'))
    if a.limit:
        pieces = pieces[:a.limit]

    print(f'{len(pieces)} pieces x {len(a.tempo_factors)} tempo factors to render', flush=True)

    done = fail = 0
    for piece in pieces:
        src_midi = perf_dir / f'{piece}.mid'
        if not src_midi.exists():
            print(f'  SKIP {piece}: no source MIDI', flush=True)
            continue
        for tf in a.tempo_factors:
            out_wav = out_dir / f'{piece}_tempo_{tf}.wav'
            out_midi = out_dir / f'{piece}_tempo_{tf}.mid'
            if out_wav.exists():
                done += 1
                continue
            try:
                scale_midi(str(src_midi), tf, str(out_midi))
                render_wav(str(out_midi), a.sound_font, a.fluidsynth, str(out_wav))
                done += 1
            except Exception as e:
                print(f'  FAIL {piece} tempo={tf}: {e}', flush=True)
                fail += 1

    print(f'Finished: done={done} fail={fail}', flush=True)


if __name__ == '__main__':
    main()
