"""D2 eval == D1's eval, verbatim. This file's entire content is the point: D2's
training-only MIDI signals (repeat-aware soft labels, MIDI-encoder distillation)
never touch inference. D2's checkpoint stores a 'midi_encoder' key alongside
'model' (see train.py's save_ckpt), but D1Model.load_state_dict only ever reads
the 'model' key -- the MidiEncoder weights are saved for reproducibility and
otherwise inert at eval time. Re-exporting rather than duplicating so any future
fix to D1's eval.py (e.g. a decode change) applies to D2 automatically.
"""
from mymodel.d1_align_matrix.eval import main, eval_piece   # noqa: F401

if __name__ == '__main__':
    main()
