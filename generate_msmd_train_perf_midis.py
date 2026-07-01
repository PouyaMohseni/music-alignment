"""
Generate missing tempo-augmented performance MIDIs for all msmd_train score pages.

For each score/{piece}.mid and each tempo_factor in [500,750,950,1000,1050,1250,1500],
produce performance/{piece}_tempo_{factor}.mid by scaling every set_tempo event by
(factor / 1000).  Skips files that already exist.

Usage:
    python generate_msmd_train_perf_midis.py [--data_dir PATH]
"""
import argparse, glob, os
import mido

TEMPO_FACTORS = [500, 750, 950, 1000, 1050, 1250, 1500]
DEFAULT_TEMPO = 500000  # 120 BPM fallback when MIDI has no set_tempo events


def scale_midi_tempo(src_path: str, dst_path: str, factor: int) -> None:
    mid = mido.MidiFile(src_path)
    new_mid = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        has_tempo = any(msg.type == 'set_tempo' for msg in track)
        # If no tempo events in this track, inject scaled default at t=0
        if not has_tempo and track.name in ('', 'Tempo Track', 'tempo'):
            scaled = max(1, int(DEFAULT_TEMPO * factor / 1000))
            new_track.append(mido.MetaMessage('set_tempo', tempo=scaled, time=0))
        for msg in track:
            if msg.type == 'set_tempo':
                new_tempo = max(1, int(msg.tempo * factor / 1000))
                new_track.append(msg.copy(tempo=new_tempo))
            else:
                new_track.append(msg)
        new_mid.tracks.append(new_track)

    new_mid.save(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=
        '/project/def-ichiro/pmohseni/music-alignment/'
        'third_party/cpjku_unet/data/msmd/msmd_train')
    args = parser.parse_args()

    score_dir = os.path.join(args.data_dir, 'score')
    perf_dir  = os.path.join(args.data_dir, 'performance')
    os.makedirs(perf_dir, exist_ok=True)

    score_mids = sorted(glob.glob(os.path.join(score_dir, '*.mid')))
    print(f"Score MIDIs found: {len(score_mids)}")
    print(f"Tempo factors:     {TEMPO_FACTORS}")
    print(f"Max to generate:   {len(score_mids) * len(TEMPO_FACTORS)}")
    print()

    generated = skipped = errors = 0
    for score_path in score_mids:
        piece = os.path.basename(score_path).replace('.mid', '')
        for factor in TEMPO_FACTORS:
            dst = os.path.join(perf_dir, f'{piece}_tempo_{factor}.mid')
            if os.path.exists(dst):
                skipped += 1
                continue
            try:
                scale_midi_tempo(score_path, dst, factor)
                generated += 1
            except Exception as e:
                print(f"  ERROR {piece} tempo={factor}: {e}")
                errors += 1

    print(f"Done. Generated={generated}  Skipped(exist)={skipped}  Errors={errors}")
    total = generated + skipped
    print(f"Performance MIDIs now present: {total} / {len(score_mids) * len(TEMPO_FACTORS)}")


if __name__ == '__main__':
    main()
