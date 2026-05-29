# MSMD Preprocessing for Audio-to-Score Alignment

This repository transforms the raw Multimodal Sheet Music Dataset (MSMD) into a processed form suitable for training the audio-to-score alignment models described in the main project repository. The pipeline produces, for each piece, an unrolled score strip image, a synthesised audio file, a small JSON metadata sidecar, and a columnar NPZ file holding the per-notehead alignment arrays. A small set of dataset-level files (splits, global statistics) sits alongside.

This README describes the preprocessing pipeline in enough detail to be implemented directly. The annotation file format is the contract between this repository and the model training code, so its specification is treated as authoritative.

## Inputs

The pipeline consumes MSMD in its public-release format. MSMD is available on GitHub from the CPJKU group at Johannes Kepler University Linz, with the data hosted on Zenodo at DOI 10.5281/zenodo.2597505. The no-audio archive is approximately 9.5 gigabytes and contains 497 solo piano pieces, each with LilyPond source files, rendered PDFs, MIDI files, and alignment annotations.

For each piece, the relevant files are organised as follows. The piece directory contains a scores subdirectory, which in turn contains directories for each engraved version of the score. Within each score directory, the imgs subdirectory holds the rendered page images as PNG files, the coords subdirectory holds numpy NPZ files with notehead and staff system pixel coordinates per page, and the score itself is in MIDI format alongside the LilyPond source.

Audio is not stored in MSMD as files; it is synthesised on demand from the MIDI using FluidSynth with a piano soundfont. The MSMD repository includes a synthesis script that handles this. The preprocessing pipeline triggers audio synthesis as part of its first stage.

## Outputs

The processed dataset has a flat top-level layout. Each piece occupies one directory containing four files. The strip.png file is the unrolled score image. The audio.wav file is the synthesised audio at 24 kHz mono (MERT's native sample rate). The annotations.json file is a small metadata sidecar described below. The noteheads.npz file is a columnar numpy archive holding all per-notehead arrays.

Three dataset-level files sit at the top of the processed directory. The splits.json file records which pieces belong to the train, validation, and test partitions, following the standard MSMD partition with the same exclusions for problematic pieces that prior work uses. The exclusions.json file records each excluded piece along with the reason. The optional noteheads.parquet file is a single Parquet table concatenating every notehead row across the dataset with a piece_id column. It is not used by the model training code, only by analysis scripts and notebooks that want fast dataset-wide statistics.

## Pipeline stages

The preprocessing pipeline has six sequential stages. Each stage is implemented as a standalone script that reads the output of the previous stage and writes its own output. This makes it easy to re-run a single stage when something changes without rebuilding everything from scratch.

### Stage 1 — Filter and split pieces

The first stage walks the MSMD directory tree, identifies all pieces, and filters out pieces that should be excluded. The standard exclusions follow Henkel 2019: pieces with Da Capo or active repeat barlines are removed because LilyPond's MIDI export handles them inconsistently, leading to alignment mismatches between the rendered score and the synthesised audio. A list of excluded pieces, each tagged with the reason for exclusion, is written to exclusions.json so that exclusions are reproducible and reviewable.

After filtering, the remaining pieces are partitioned into train, validation, and test splits matching the MSMD standard partition. The output is splits.json, which lists piece IDs per split.

### Stage 2 — Audio synthesis

For each piece in any split, the MIDI is rendered to a mono WAV file at 24 kHz using FluidSynth with the piano soundfont used by prior work. The rate matches MERT's native sample rate so no per-batch resampling is needed at training time. The audio is normalised to peak amplitude minus three decibels to avoid clipping. The output is one WAV file per piece, stored in the piece directory as audio.wav. The synthesis configuration (soundfont identifier, sample rate, peak target) is recorded in the metadata sidecar produced in Stage 5 so that audio and annotations can be verified against one another.

This stage runs in parallel across pieces, since each is independent. Total wall-clock time on a modest machine is approximately two hours for the full dataset.

### Stage 3 — Staff system extraction

For each piece, the rendered page images are loaded one at a time. For each page, the staff system bounding boxes are loaded from the corresponding NPZ file in the coords directory. Each bounding box is given in pixel coordinates as a tuple of x_min, y_min, x_max, y_max relative to the original page image.

This stage produces no output files on its own; it is a function called by Stage 4. It is documented separately because it is a logical unit that may be modified independently in the future (for example, to swap in a learned detector for non-MSMD data sources).

### Stage 4 — Strip construction and unrolling

The unrolled strip is built by iterating over all pages of a piece in order, and within each page over all staff systems in vertical order. Each system is cropped from the page image using its bounding box and placed horizontally adjacent to the previous system in the strip. The vertical alignment within the strip is set so that all systems share the same y-coordinate range, which means each system is padded vertically to a fixed strip height. The strip height is the maximum system height observed across the entire dataset, typically around 120 pixels.

A small horizontal gap (5 to 10 pixels) is inserted between systems to give the model a visual cue about system boundaries, though this is optional and can be set to zero. Whether or not a gap is used, the position of each system in the strip is recorded for the mapping table.

The output is the strip image saved as strip.png, plus an in-memory mapping table that lists, for each system in order, its start and end strip-x coordinates and the bounding box it came from on the original page. The mapping is passed to Stage 5 and is also embedded in the JSON metadata sidecar for self-containment, so the training code does not need to load a separate mapping file at runtime.

### Stage 5 — Annotation file construction

This is the most important stage. It produces the two annotation artifacts that the training pipeline consumes, the JSON metadata sidecar and the columnar NPZ archive. The full schema is given in the next section.

For each piece, the stage performs the following operations. The notehead annotation NPZ files for all pages are loaded. Each notehead entry contains a pixel coordinate (x, y) on its original page and an onset time in seconds relative to the start of the synthesised audio. The page-coordinate pixel positions are translated to strip-x coordinates using the mapping table from Stage 4. Notehead entries are then sorted by onset time. Each notehead is tagged with its system index in the strip and its measure index, derived from MIDI bar markers. Additional per-piece metadata (total duration, strip dimensions, beat and bar times, tempo events, soundfont identifier, audio and strip file hashes) is computed.

The resulting arrays are written to noteheads.npz and the scalar and list-valued metadata is written to annotations.json. Both files are written atomically (write-then-rename) so that an interrupted run never leaves a piece in a partially written state.

### Stage 6 — Global aggregate

For convenience of analysis, all per-piece notehead arrays are concatenated into a single Parquet file at the dataset root, with an additional piece_id column. This file is regenerated whenever any per-piece NPZ changes. It is not consumed by the training dataloader and may be skipped on disk-constrained machines.

## Annotation file format

The annotation files are the contract between this repository and the model training code. Their format is specified precisely so that downstream code can rely on it. The schema version is included in every metadata sidecar; downstream code should check it and refuse to load files from unsupported versions.

### Per-piece metadata sidecar (annotations.json)

The metadata sidecar is a small JSON object, typically a few kilobytes per piece. It is intended to be human-readable and to hold only scalar metadata, short lists, and the strip-to-page mapping table.

The top-level structure is as follows.

The schema_version field is a string in semantic-version form (currently "1.0"). Major-version bumps signal a breaking change to the format.

The piece_id field is a string matching the piece directory name in the processed dataset. It serves as a unique identifier.

The score_engraving_id field records which LilyPond engraving of the score (MSMD provides multiple per piece) was used. This is needed because the page images and notehead coordinates differ across engravings.

The audio sub-object contains five fields. The path field is the relative path to the WAV file, normally "audio.wav". The sha256 field is the SHA-256 hash of the WAV file, computed at the time of synthesis. The sample_rate_hz field is the audio sample rate (24000 for MSMD-synthesised audio; MERT-native). The duration_sec field is the total duration of the audio in seconds. The soundfont field is the identifier of the soundfont used for synthesis. The peak_db field is the target peak amplitude used by the synthesis normalisation step.

The strip sub-object contains five fields. The path field is the relative path to the PNG file, normally "strip.png". The sha256 field is the SHA-256 hash of the strip image. The width_px and height_px fields are the strip image dimensions. The system_gap_px field records the horizontal gap inserted between systems during strip construction.

The tempo_events field is a list of tempo change events. Each event is an object with a time_sec field and a bpm field. MSMD MIDI files almost always contain at least one tempo event, and may contain more. This replaces the scalar tempo field used in prior versions of this spec.

The beat_times_sec field is a list of floating-point numbers giving the times of beat boundaries in seconds, derived from the MIDI tempo events. This is useful for beat-aligned windowing during training.

The bar_times_sec field is a list of floating-point numbers giving the times of bar lines in seconds, derived from the MIDI time signature events. This is useful for measure-level evaluation.

The strip_to_page_mapping field is a list of mapping entries. Each entry is an object with strip_x_start, strip_x_end, page_idx, system_idx_in_strip, and page_bbox (a four-element list of pixel coordinates) fields. The mapping is used by the training code to translate predicted strip positions back to original pages for visualisation.

The notehead_count field is an integer giving the number of noteheads in the companion NPZ file. This is included so that downstream code can validate the NPZ has the expected length without loading it.

### Per-piece notehead arrays (noteheads.npz)

The notehead arrays are stored as a numpy NPZ archive. Loading a single piece takes a few milliseconds with `np.load`, and the resulting arrays can be consumed by the dataloader without further parsing.

The archive contains nine arrays, each of length N (the number of noteheads in the piece). Every array is indexed in the same order, sorted by onset_sec ascending.

The onset_sec array (float32) holds note onset times in seconds, relative to the start of the audio.

The midi_offset_sec array (float32) holds note-off times in seconds, derived from MIDI note durations. The name is deliberately distinct from offset_sec because, on the synthesised audio, sustain pedal events and the natural decay of the piano envelope extend the audible note beyond the MIDI note-off time; the field reflects MIDI structure, not audio energy.

The strip_x array (int32) holds the x-coordinate of each notehead on the unrolled strip, in pixels. The y-coordinate is not stored because all relevant alignment happens along the strip's horizontal axis.

The midi_pitch array (int8) holds the MIDI pitch number (60 is middle C).

The system_idx array (int16) holds the index of the system (within the strip) that contains the notehead. This is useful for per-system error analysis and for plotting predicted alignments back onto specific systems.

The measure_idx array (int16) holds the index of the bar that contains the notehead, derived from MIDI bar lines. This is useful for measure-level evaluation metrics that prior work reports.

The page_idx, page_x, and page_y arrays (int16) preserve the original page coordinates for debugging and visualisation, but are not used by the training code directly.

### Example metadata sidecar

A truncated example of the metadata sidecar follows.

```json
{
  "schema_version": "1.0",
  "piece_id": "BachCPE__cpe-bach-rondo",
  "score_engraving_id": "lilypond-2-18-2",
  "audio": {
    "path": "audio.wav",
    "sha256": "1a2b3c...",
    "sample_rate_hz": 24000,
    "duration_sec": 187.421,
    "soundfont": "FluidR3_GM.sf2",
    "peak_db": -3.0
  },
  "strip": {
    "path": "strip.png",
    "sha256": "9f8e7d...",
    "width_px": 18420,
    "height_px": 120,
    "system_gap_px": 8
  },
  "tempo_events": [{"time_sec": 0.0, "bpm": 120.0}],
  "beat_times_sec": [0.0, 0.5, 1.0, 1.5, 2.0],
  "bar_times_sec":  [0.0, 2.0, 4.0],
  "strip_to_page_mapping": [
    {
      "strip_x_start": 0,
      "strip_x_end": 1042,
      "page_idx": 1,
      "system_idx_in_strip": 0,
      "page_bbox": [80, 350, 1122, 480]
    }
  ],
  "notehead_count": 3421
}
```

### Dataset-level files

The splits.json file at the dataset root is a JSON object with three keys (train, val, test), each a list of piece_id strings.

The exclusions.json file is a list of objects, each with a piece_id field and a reason field (for example "da_capo_in_lilypond" or "missing_coords_npz").

The noteheads.parquet file, if present, has the union of all per-piece notehead columns plus a piece_id column. Schema is identical to the NPZ arrays.

## Why this format

Several design decisions shape this format and are worth understanding before modifying it.

The annotation file stores continuous timestamps and pixel coordinates, not pre-cut snippets. This is deliberate. Pre-cutting snippets at preprocessing time would fix the window size and stride, making it impossible to experiment with different settings later. By storing the raw alignment between time and position, the dataloader can sample arbitrary windows on the fly during training, supporting random temporal augmentation and flexible hyperparameter sweeps.

The split between a JSON metadata sidecar and a columnar NPZ archive separates two access patterns. The sidecar is read once when a piece is opened, occasionally inspected by humans, and small enough that JSON parse cost is negligible. The notehead arrays are accessed in the inner loop of training, possibly thousands of times per epoch across many dataloader workers, and benefit from numpy-native storage. JSON parsing of thousands of notehead dicts is one of the slower steps in a naive implementation and disappears entirely with NPZ.

The columnar layout matches how the dataloader accesses the data. The training code only ever needs aligned slices of onset_sec, strip_x, and a few other columns; it never needs all fields of one notehead in isolation. Columnar storage lets `np.searchsorted(onset_sec, t0)` locate the first notehead inside a window in microseconds and lets boolean masking pull out the rest in one pass.

The strip representation collapses the two-dimensional page layout into one dimension. This matches the natural structure of musical time: left-to-right on the strip corresponds to earlier-to-later in time, with no ambiguity about which system to look at next. Models that operate on the strip therefore predict alignment along a single axis rather than two, which simplifies both the architecture and the loss function.

The format excludes the audio waveform and the strip image to keep annotation files small. The waveform stays as audio.wav and the strip stays as strip.png. The sidecar links them together with timestamps, pixel coordinates, and content hashes but does not duplicate their content. A typical metadata sidecar is between two and five kilobytes, the notehead NPZ is between fifty and five hundred kilobytes depending on piece length, and the audio and image files together are tens of megabytes per piece.

Hashes of the audio and strip files are stored in the sidecar so that the annotation can be verified against its companion files. When the audio is re-synthesised with a different soundfont, or the strip is rebuilt with a different gap, the hashes change and stale annotations are detected immediately rather than silently pairing with the wrong files.

Schema versioning is included from the start because annotation formats inevitably change. Storing a version string in every sidecar lets future code recognise files from earlier formats and either migrate them or refuse to load them.

Repeats are handled at the filter stage rather than at the annotation stage. This means a notehead annotation always maps a single audio onset time to a single strip-x position, with no ambiguity. This simplification matches the standard practice in all prior end-to-end work on this dataset.

## How the training code uses the annotation file

For completeness, here is how the model training code consumes the annotation files.

When a piece is first opened, the training code reads its annotations.json, checks the schema version, optionally verifies the audio.wav and strip.png hashes, and loads the noteheads.npz into a dictionary of numpy arrays. These can be cached in memory if the dataset fits.

The dataloader picks a piece and a random window start time, typically uniform between zero and duration_sec minus the window length. It loads the audio waveform, slices a window of the configured length (five seconds by default, matching MERT's pretraining segment length), and feeds it to the audio encoder.

It then identifies the noteheads with onset_sec falling within the window by binary searching the (sorted) onset_sec array. The minimum and maximum strip_x of these noteheads define the strip region of interest, optionally expanded with a small margin. The strip is sliced into vertical column patches of the configured width and stride, and these are fed to the image encoder.

The ground-truth alignment for the window is constructed from the noteheads within it. For each notehead, the audio frame index corresponding to its onset_sec is computed using the audio frame rate, and the image column index corresponding to its strip_x is computed using the column stride. These produce the (audio_frame_idx, image_column_idx) pairs that define the diagonal of the similarity matrix. The SoftDTW loss is then computed over the similarity matrix and the ground-truth diagonal.

This sketch is implemented as a PyTorch Dataset and DataLoader in the training repository.

## Running the pipeline

The full pipeline is run end-to-end with a single command, typically `python -m msmd_prep.run_all --config configs/default.yaml`. Each stage can also be run individually for debugging. The pipeline is deterministic given a fixed configuration and the same input data; no randomness is introduced at preprocessing time, since all randomness (window sampling, augmentation) lives in the training code.

The pipeline uses local disk for intermediate outputs and is not designed to run on a compute cluster. Total wall-clock time for the full dataset on a modest workstation is approximately three to four hours, dominated by audio synthesis. Storage requirements for the processed dataset are around five to ten gigabytes, depending on audio compression.

## Repository layout

The repository follows a standard Python package layout. The main package directory contains modules for each pipeline stage: filter.py, synth.py, systems.py, strip.py, annotate.py, and aggregate.py. A schema.py module defines the schema version constant, the NPZ array names and dtypes, and the JSON sidecar key list; every other stage imports from this single source of truth. A run_all.py script orchestrates the stages in order. Configuration files in YAML format are stored in the configs directory and specify all hyperparameters of the preprocessing, including audio synthesis settings, strip dimensions, and exclusion lists. The tests directory contains unit tests for each stage and integration tests that verify the full pipeline produces correct outputs on a small subset of MSMD.

## Validation

Three validation steps are recommended after running the pipeline.

The first is to visually inspect a handful of strip images and verify that staff systems are correctly concatenated. Common issues include incorrect bounding boxes from the source data, vertical misalignment within the strip, and missing systems. A debug script in the repository produces side-by-side comparisons of original pages and constructed strips.

The second is to visually inspect alignment plots. For each piece, plot the notehead positions on the strip with their onset times annotated. A correct alignment shows a monotonically increasing relationship between time and strip-x. Outliers indicate problems with the synthesis or annotation. A debug script produces these plots.

The third is a set of automated sanity checks. For every processed piece, the pipeline verifies that onset_sec is monotonically non-decreasing, that strip_x lies within [0, strip.width_px], that the Spearman correlation between onset_sec and strip_x is above 0.99, that the audio hash in the sidecar matches the audio file on disk, that the strip hash matches the strip file on disk, and that the notehead_count field equals the length of every NPZ array. Any piece failing a check is logged and, if the failure cannot be auto-resolved, added to exclusions.json with the reason.

All three validation steps should pass on every piece in the train and validation splits before model training is attempted.

## Future extensions

Several extensions of the pipeline are anticipated but not yet implemented.

Support for MSMD-Rec, the real-audio extension dataset, requires extending Stage 2 to use the recorded WAV file instead of synthesising audio from MIDI. The remainder of the pipeline is unchanged because MSMD-Rec uses the same MIDI alignment as MSMD itself.

Support for the Magaloff and Zeilinger corpora requires more substantial changes because those datasets do not share the LilyPond source pipeline. A separate ingestion module will be developed when the time comes.

For cluster training, the per-piece directories may be repackaged into WebDataset tar shards, with each sample containing strip.png, audio.wav, annotations.json, and noteheads.npz. This is a packaging change, not a schema change, and is deferred until the schema is stable and the model training has moved to multi-node runs.

Support for non-Western notation, specifically Iranian classical music, will require either manual annotation of staff system coordinates (since no automatic detector exists for such notation) or a different segmentation strategy entirely. This is a research question to be addressed in the cross-tradition pilot phase of the main project.

## Licence

Code in this repository is released under the MIT licence. The MSMD source data is distributed under its own licence, which permits research use; see the MSMD repository for details. Synthesised audio is not redistributed by this repository.
