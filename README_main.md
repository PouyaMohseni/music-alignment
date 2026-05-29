# Audio-to-Score Alignment via Foundation Models

End-to-end audio-to-score alignment using a multi-modal system built on pretrained foundation models for music audio (MERT) and sheet music images (ViT), trained with a SoftDTW loss that exploits the monotonic temporal structure of musical alignment.

This work proposes a research thesis. The repository hosts code, configuration, and experimental records.

## Motivation

The task is to take a recording of a musical performance and find the matching position in the score at every moment in time. Existing approaches fall into three families. Audio-to-audio pipelines synthesise the score and compare it to the performance using DTW on chroma features; symbolic pipelines first transcribe the audio to MIDI and then align it to the score; end-to-end approaches learn the audio-to-score correspondence directly from paired data. The third family is the most flexible because it does not depend on optical music recognition or transcription, but existing end-to-end systems use small task-specific networks trained from scratch on synthetic piano data and generalise poorly to real audio and scanned scores.

This work replaces those small networks with two pretrained foundation models, connects them through learned projection heads into a shared embedding space, and trains the system end-to-end with a differentiable variant of dynamic time warping. The hypothesis is that the broad pretraining of MERT and a vision transformer transfers prior knowledge about music and document images that small networks trained on MSMD alone cannot acquire.

A secondary motivation is that the approach makes no assumption about symbolic encodings. Musical traditions that lack machine-readable formats, such as Iranian classical music, are inaccessible to pipeline-based systems but in principle accessible to ours.

## Research questions

The work is structured around three questions.

Do pretrained foundation model encoders, connected by lightweight learned projection heads, yield more accurate audio-to-score alignments than the task-specific convolutional networks used in prior end-to-end work?

Does a temporally aware training loss based on SoftDTW outperform standard contrastive objectives such as InfoNCE for this task?

How well do the learned representations generalise beyond their training distribution, specifically from synthetic to real audio and across notation systems?

## High-level system design

The system processes two inputs. The audio is a music recording at 24 kHz (MERT's native sample rate), processed in five-second sliding windows with a beat-aligned stride. Five seconds matches MERT-v1-95M's pretraining segment length, which keeps the encoder in-distribution and yields the strongest features. The image is a sheet music page, preprocessed by detecting staff systems, cropping them, and concatenating them horizontally into a one-dimensional strip; the strip is sliced into vertical column patches each covering approximately one beat of music.

Each audio window passes through MERT-v1-95M, producing a sequence of frame embeddings at 75 Hz (375 frames per five-second window). Each image column passes through a vision transformer, producing one vector per column. Projection heads map both into a shared embedding space of dimension d. The pairwise cosine similarities between every image column and every audio frame form a similarity matrix on which SoftDTW is computed as the training loss. The MERT frame embeddings are kept dense rather than mean-pooled, so the time dimension of the matrix is preserved for the alignment loss.

At inference time, standard DTW is run on the predicted similarity matrix to produce the alignment path. The path is converted back to time using the notehead-onset annotations from the dataset, and the predicted alignment time for each ground-truth note onset is compared to its actual time to compute tracking error.

## Repository structure

The project is split across multiple repositories.

The current repository contains the model architecture, training and evaluation scripts, and configuration files for all experimental variants.

A separate dataset repository handles all data preparation, from MSMD download through staff system unrolling to the production of per-piece annotation files. See the dataset README for details.

A future repository will contain the trained model checkpoints and inference utilities.

## Architecture variants

Four variants are trained and compared. The comparison isolates the effect of the encoder adaptation strategy on alignment quality while keeping the SoftDTW loss and preprocessing pipeline fixed.

Variant A freezes both foundation models and trains only the projection heads. This is the cheapest variant and the strongest test of whether the pretrained representations are useful out-of-the-box.

Variant B applies a staged training procedure in which each encoder is first adapted to its domain through unimodal self-supervised pretraining with LoRA, after which the encoders are frozen and the projection heads are trained jointly. This allows domain adaptation without requiring paired data for the adaptation step.

Variant C unfreezes all parameters and trains the entire system end-to-end at a low encoder learning rate. This has the highest performance ceiling but the highest risk of overfitting.

Variant D replaces the mean pooling in the audio branch with a Q-Former, a small cross-attention adapter that produces a fixed-size embedding from a variable-length sequence of MERT frame tokens.

## Evaluation

Three test tiers of increasing difficulty are used. Tier one is the synthetic MSMD test split, which uses LilyPond-rendered images and FluidSynth-synthesised audio. Tier two is MSMD-Rec, which pairs real piano performances recorded on a Yamaha hybrid piano with the same typeset MSMD scores. Tier three combines the Magaloff corpus and the Zeilinger dataset, both of which provide real concert recordings on Bösendorfer instruments aligned to scanned score images.

The primary metric is the cumulative tracking error curve, which reports the percentage of note onsets aligned within each of a set of tolerance thresholds, typically 0.05, 0.1, 0.5, 1, and 5 seconds. Mean absolute tracking error serves as a single-number summary. For comparison with the cross-modal retrieval literature, mean reciprocal rank and recall at k are also reported on the snippet retrieval task.

A small qualitative experiment on Iranian classical music tests whether the learned representations generalise across notation systems. As note-level ground truth is not available for this material, the output is qualitative alignment plots rather than tracking error statistics.

## Execution roadmap

The project unfolds over twelve weeks across five phases.

Week 1 is environment setup. Install a local conda environment with PyTorch, HuggingFace transformers, peft, librosa, and pysdtw. Request a Compute Canada allocation and verify that simple GPU jobs run successfully. Download MSMD and explore its structure.

Weeks 2 and 3 reproduce the Henkel 2021 baseline using the public cyolo_score_following codebase. Verify that the published tracking error numbers can be reproduced within a reasonable margin. Document the data flow from MSMD raw files through tensors to model output to evaluation metrics.

Weeks 4 and 5 build the custom data pipeline: staff system unrolling, image column slicing, audio sliding windowing, and ground-truth alignment generation. This is described in detail in the dataset README. The deliverable is a notebook that takes any MSMD piece and produces visually verified paired sequences.

Week 6 builds a minimal end-to-end model using small CNN encoders and the SoftDTW loss. The goal is to overfit a single piece to prove that the loss and training loop work, then train on the full MSMD split. This gives a baseline number that does not yet use foundation models, isolating the effect of the loss function.

Weeks 7 through 9 introduce the foundation models incrementally. First swap the audio encoder for MERT, then the image encoder for ViT, then add LoRA adapters. Each variant is trained on the full MSMD split and evaluated against the baseline.

Weeks 10 through 12 cover evaluation on the real-audio and scanned-score tiers, the Iranian music pilot, and the writing of the thesis.

## Local versus cluster work

Development happens locally with tiny subsets of the data. The dataloader, model, loss, and evaluation must all run on five pieces on a laptop CPU or modest GPU before any cluster job is submitted. Full training runs and evaluation on the complete test set go to Compute Canada.

Visualisation is local. The cluster is a bad place to look at plots of similarity matrices, alignment paths, or training curves. Use Weights and Biases to stream training metrics back to your laptop so you can monitor cluster jobs interactively.

A typical training run for variant A takes around twelve hours on a single A100. Full ablation including all four variants and three test tiers requires roughly two to three hundred GPU hours, which is well within a standard masters-level allocation.

## Dependencies

The project requires Python 3.10 or later. The major Python packages are PyTorch for deep learning, HuggingFace transformers for the foundation models, HuggingFace peft for LoRA adapters, librosa and torchaudio for audio processing, pysdtw or soft-dtw-cuda for the differentiable DTW loss, pretty-midi for reading MIDI files, Pillow and torchvision for image handling, and matplotlib and seaborn for visualisation. Weights and Biases is used for experiment tracking. A full requirements.txt is provided.

## Outputs and reproducibility

All experiments are configured via YAML files in the configs directory. Each run is logged with its config, git commit hash, and random seed so that any result can be reproduced exactly. Trained model checkpoints are saved at each evaluation milestone and shared via the model checkpoints repository.

Results tables are produced as CSV files and converted to LaTeX for inclusion in the thesis. Figures are produced as PDFs from matplotlib for the same purpose.

## Related work

The closest prior systems are Henkel and Widmer 2021 for the alignment task and Carvalho, Washüttl, and Widmer 2023 for the cross-modal retrieval task. Both are publicly available on GitHub from the CPJKU group at Johannes Kepler University Linz. The dataset is MSMD, introduced by Dorfer and colleagues in 2018, with the MSMD-Rec extension introduced by Henkel and colleagues in 2019. The SoftDTW loss is from Cuturi and Blondel 2017. The audio foundation model is MERT from the Music and Audio Processing group, 2023.

## Status

This is an active research project. The current state is the planning and proposal phase. Implementation begins after the data pipeline (described in the dataset README) is complete.
