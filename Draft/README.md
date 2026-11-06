# Draft notebooks — experiment notes

These are **earlier experiments**, not the final working TTS system. Some cells are exploratory, repeated, or depend on previous Kaggle state.

## Draft 1 — Initial setup and recovery checks

**Goal:** Connect to Hugging Face, test access to the English and Telugu speech datasets, and look for earlier FastSpeech 2 backups in Kaggle.

**What we tried:** Used Kaggle Secrets for Hugging Face access, installed dataset tools, loaded datasets, and searched Kaggle folders for saved project files.

**Result:** Early setup and recovery checks only; no complete training pipeline or speech synthesis.

## Draft 2 — Dataset preparation, training, and synthesis experiments

**Goal:** Train separate English and Telugu FastSpeech 2-style text-to-speech models.

**What we tried:**

1. Loaded English Svarah and Telugu IndicTTS and inspected audio metadata.
2. Prepared audio and train/validation manifests targeting about five hours of training and one hour of validation per language.
3. Tested CTC forced alignment, including Telugu romanization and alignment checks.
4. Extracted mel-spectrogram, duration, pitch, and energy features.
5. Built separate FastSpeech 2-style acoustic models, trained them, and checked losses and checkpoints.
6. Tried revised model versions, feature normalization, and speech synthesis.
7. Used plots, mel reconstruction, and short-sample experiments to investigate poor audio quality.

**Result:** Substantial experiments and saved plots, but the custom FastSpeech 2-style synthesis was not intelligible. This notebook is not a verified clean end-to-end run.

## Saved notebook images

Draft 2 contains seven embedded PNG outputs, from cells 29, 60, 64 (two images), 72, 73, and 82. The original notebook outputs are preserved in the downloadable archive. For GitHub, the notebook has been saved without large outputs, and all seven PNG figures have been uploaded separately under `Draft/images/`.

## Feature checkpoint backup

The `features_checkpoint.tar.gz` backup contains preprocessed acoustic features. It is not a complete trained model or a replacement for the audio dataset.

## Final notebook

The later Final notebook was affected by a Kaggle crash. Its recovered code and backup results are documented separately. The Draft notebooks should not be mistaken for that Final work.
