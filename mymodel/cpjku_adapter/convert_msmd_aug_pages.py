"""Convert MSMD augmented data to CPJKU page format.

Produces per-page NPZ files and per-page tempo-scaled MIDI files that exactly
match the Zenodo CPJKU dataset format (Henkel et al. ISMIR 2020).

Input:
  data/MSMD/processed/<piece>/noteheads.npz  -- note coords + onset times
  data/MSMD/msmd_aug_v1-1_no-audio/<piece>/scores/<eng>/img/<N>.png  -- page imgs
  data/MSMD/msmd_aug_v1-1_no-audio/<piece>/scores/<eng>/coords/systems_<N>.npy
  data/MSMD/processed/<piece>/score.midi  -- piece MIDI (tempo=1000 base)

Output:
  data/MSMD/msmd_aug_cpjku_pages/score/<piece>_page_<N>.npz
  data/MSMD/msmd_aug_cpjku_pages/performance/<piece>_page_<N>_tempo_<T>.mid
"""

import argparse
import glob
import os
import sys

import numpy as np
import pretty_midi
from PIL import Image
from tqdm import tqdm

TEMPO_FACTORS = [500, 750, 1000, 1250, 1500, 1750, 2000]
PROCESSED_ROOT = "data/MSMD/processed"
AUG_ROOT = "data/MSMD/msmd_aug_v1-1_no-audio"
OUTPUT_ROOT = "data/MSMD/msmd_aug_cpjku_pages"


def get_staff_height(systems, note_y):
    """Return the height of the staff containing note_y."""
    best_dist = float("inf")
    best_h = 80.0
    for staff in systems:
        top_y = float(staff[0][0])
        bot_y = float(staff[2][0])
        h = bot_y - top_y
        # distance from note to staff centre
        centre = (top_y + bot_y) / 2.0
        dist = abs(note_y - centre)
        if dist < best_dist:
            best_dist = dist
            best_h = h
    return best_h


def build_coord2onset(page_onset_sec, midi_onset_sec):
    """Map coord_idx -> index in per-page MIDI notes array (sorted by onset).

    midi_onset_sec: 1-D array of onset times (seconds) for notes in the
    per-page MIDI, sorted ascending.
    """
    coord2onset = {}
    for coord_idx, t in enumerate(page_onset_sec):
        midi_idx = int(np.argmin(np.abs(midi_onset_sec - t)))
        coord2onset[coord_idx] = midi_idx
    return coord2onset


def make_page_midi(full_midi_path, page_onset_min, page_onset_max, tempo_factor):
    """Return a PrettyMIDI object containing only notes on this page,
    onset-shifted to t=0 and tempo-scaled by (1000/tempo_factor)."""
    pm = pretty_midi.PrettyMIDI(full_midi_path)
    scale = 1000.0 / tempo_factor

    new_pm = pretty_midi.PrettyMIDI()
    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(
            program=inst.program, is_drum=inst.is_drum, name=inst.name
        )
        for note in inst.notes:
            # include notes whose onset falls in this page's time window
            # add a tiny tolerance (0.05 s) to catch boundary notes
            if page_onset_min - 0.05 <= note.start <= page_onset_max + 0.05:
                shifted_start = max(0.0, (note.start - page_onset_min) * scale)
                shifted_end = max(shifted_start + 0.01,
                                  (note.end - page_onset_min) * scale)
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=shifted_start,
                    end=shifted_end,
                )
                new_inst.notes.append(new_note)
        if new_inst.notes:
            new_pm.instruments.append(new_inst)

    return new_pm


def convert_piece(piece_id, output_root,
                  processed_root=None, aug_root=None):
    processed_root = processed_root or PROCESSED_ROOT
    aug_root = aug_root or AUG_ROOT
    processed_dir = os.path.join(processed_root, piece_id)
    aug_dir = os.path.join(aug_root, piece_id)

    # --- locate score engraving dir ---
    score_engravings = [
        d for d in os.listdir(os.path.join(aug_dir, "scores"))
        if os.path.isdir(os.path.join(aug_dir, "scores", d))
    ]
    if not score_engravings:
        return 0
    eng_dir = os.path.join(aug_dir, "scores", score_engravings[0])

    # --- load noteheads (per-note page info + onset times) ---
    nh_path = os.path.join(processed_dir, "noteheads.npz")
    if not os.path.exists(nh_path):
        return 0
    nh = np.load(nh_path)
    page_idx_all = nh["page_idx"]
    page_x_all = nh["page_x"].astype(np.float32)
    page_y_all = nh["page_y"].astype(np.float32)
    onset_sec_all = nh["onset_sec"].astype(np.float64)

    # --- load full MIDI ---
    midi_path = os.path.join(processed_dir, "score.midi")
    if not os.path.exists(midi_path):
        return 0

    n_pages_created = 0
    for page_n in sorted(np.unique(page_idx_all)):
        page_n = int(page_n)
        img_fname = f"{page_n + 1:02d}.png"
        img_path = os.path.join(eng_dir, "img", img_fname)
        sys_path = os.path.join(eng_dir, "coords", f"systems_{page_n + 1:02d}.npy")

        if not os.path.exists(img_path):
            continue

        # --- load page image (greyscale, uint8) ---
        sheet = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        # Pad to standard height (1181) so all pages are uniform — allows batch_size>1.
        if sheet.shape[0] < 1181:
            pad = np.full((1181 - sheet.shape[0], sheet.shape[1]), 255, dtype=np.uint8)
            sheet = np.vstack([sheet, pad])

        # --- notes on this page ---
        mask = page_idx_all == page_n
        if mask.sum() == 0:
            continue
        px = page_x_all[mask]
        py = page_y_all[mask]
        onset_sec = onset_sec_all[mask]
        n_notes = len(px)

        # --- staff height per note ---
        if os.path.exists(sys_path):
            systems = np.load(sys_path)
            heights = np.array([get_staff_height(systems, float(y)) for y in py],
                               dtype=np.float32)
        else:
            heights = np.full(n_notes, 80.0, dtype=np.float32)

        # coords: [y, x, height] — in original pixel space (load_score divides by scale_factor)
        coords = np.column_stack([py, px, heights]).astype(np.float32)

        # --- page MIDI time window ---
        page_onset_min = float(onset_sec.min())
        page_onset_max = float(onset_sec.max())

        # --- build per-page MIDI (tempo=1000 base) ---
        page_pm = make_page_midi(midi_path, page_onset_min, page_onset_max, 1000)
        if not page_pm.instruments or not any(
            inst.notes for inst in page_pm.instruments
        ):
            continue

        # all notes in page MIDI, sorted by onset
        all_notes_sorted = sorted(
            [n for inst in page_pm.instruments for n in inst.notes],
            key=lambda n: n.start,
        )
        midi_onset_sec = np.array([n.start for n in all_notes_sorted], dtype=np.float64)

        # --- coord2onset ---
        # onset_sec are already at tempo_1000; midi onsets are also at tempo_1000 scale
        coord2onset = build_coord2onset(onset_sec, midi_onset_sec)

        # --- save NPZ ---
        npz_name = f"{piece_id}_page_{page_n}.npz"
        npz_path = os.path.join(output_root, "score", npz_name)
        # Save coord2onset as (N,2) int32 pairs [coord_idx, onset_idx] to avoid
        # numpy pickle version incompatibility (numpy 2.x vs 1.x).
        c2o_pairs = np.array([[k, v] for k, v in sorted(coord2onset.items())],
                             dtype=np.int32)
        np.savez(npz_path,
                 sheet=sheet,
                 coords=coords,
                 coord2onset_pairs=c2o_pairs)

        # --- save tempo-scaled MIDIs ---
        for tempo in TEMPO_FACTORS:
            tempo_pm = make_page_midi(midi_path, page_onset_min, page_onset_max, tempo)
            mid_name = f"{piece_id}_page_{page_n}_tempo_{tempo}.mid"
            mid_path = os.path.join(output_root, "performance", mid_name)
            tempo_pm.write(mid_path)

        n_pages_created += 1

    return n_pages_created


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", default=PROCESSED_ROOT)
    parser.add_argument("--aug_root", default=AUG_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split_file", default=None,
                        help="Path to splits.json; if set, only convert pieces in --split_key")
    parser.add_argument("--split_key", default="train",
                        help="Which split to use: train, val, or test")
    args = parser.parse_args()

    out_root = args.output_root
    proc_root = args.processed_root
    a_root = args.aug_root

    os.makedirs(os.path.join(out_root, "score"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "performance"), exist_ok=True)

    # pieces present in both processed/ and aug/
    processed_pieces = set(os.listdir(proc_root))
    aug_pieces = set(os.listdir(a_root))
    pieces = sorted(processed_pieces & aug_pieces)
    pieces = [p for p in pieces if os.path.isdir(os.path.join(proc_root, p))]

    # optionally restrict to a specific split
    if args.split_file is not None:
        import json
        with open(args.split_file) as f:
            splits = json.load(f)
        keep = set(splits[args.split_key])
        pieces = [p for p in pieces if p in keep]
        print(f"Restricting to split '{args.split_key}': {len(pieces)} pieces")

    print(f"Converting {len(pieces)} pieces to page format...")
    total_pages = 0

    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            results = list(tqdm(
                pool.starmap(convert_piece, [(p, out_root, proc_root, a_root) for p in pieces]),
                total=len(pieces)
            ))
        total_pages = sum(results)
    else:
        for p in tqdm(pieces):
            total_pages += convert_piece(p, out_root, proc_root, a_root)

    print(f"Done. Created {total_pages} pages across {len(pieces)} pieces.")
    print(f"Output: {out_root}")

    # write split file for training (all pages)
    score_files = sorted(os.listdir(os.path.join(out_root, "score")))
    piece_names = [f[:-4] for f in score_files if f.endswith(".npz")]

    import yaml
    splits_path = os.path.join(out_root, "split_train.yaml")
    with open(splits_path, "w") as f:
        yaml.dump({"files": piece_names}, f)
    print(f"Wrote {len(piece_names)} entries to {splits_path}")


if __name__ == "__main__":
    main()
