# English–Telugu Text-to-Speech: FastSpeech 2 Experiments

This repository documents English and Telugu text-to-speech experiments based on the FastSpeech 2 architecture, together with an independently pretrained MMS-VITS inference baseline.

## Project status

**Experimental FastSpeech 2:** Separate English and Telugu acoustic models were trained and evaluated. Losses decreased and checkpoints were saved, but synthesized speech remained unintelligible/noisy. These models **should not be described as successful speech synthesizers**.

**Pretrained MMS-VITS baseline:** The separately loaded `facebook/mms-tts-eng` and `facebook/mms-tts-tel` checkpoints generated audible speech for five test sentences in each language. These models were **not trained on the assignment datasets** and **are not FastSpeech 2**.

## Data and preprocessing

- English source: `ai4bharat/Svarah`
- Telugu source: `SPRINGLab/IndicTTS_Telugu`
- Audio resampled to 22,050 Hz; manifests targeting approximately 5 hours of training and 1 hour of validation per language
- IPA phonemization with eSpeak NG, including Telugu phoneme conversion
- CTC word-level forced alignment, followed by estimated within-word phoneme durations (not phoneme-level forced alignment)
- 80-bin log-mel spectrograms, pitch and energy features

## Repository layout

- `Final/`: acoustic model checkpoints, figures, manifests, phoneme frontend and audio results
- `Final/Results/<language>/test_generated_audios/`: FastSpeech 2 experimental synthesis (unintelligible)
- `Final/Results/<language>/calibrated_audio/`: additional calibrated-model experiments (unintelligible)
- `Final/Results/<language>/pretrained_baseline/`: audible **MMS-VITS** baseline outputs
- `Draft/`: earlier exploratory notebooks and experiments (to be added later)

## Architecture and limitations

The custom implementation was a simplified FastSpeech 2-style acoustic model with duration, pitch, and energy predictors; it is not a faithful reproduction of all components in the upstream `ming024/FastSpeech2` implementation. The reference implementation is available at https://github.com/ming024/FastSpeech2 and should be credited separately.

Acoustic feature extraction and model training were completed for a subset of aligned examples. The reduced aligned subset, approximate phoneme timing, and inference/training mismatch are limitations; their relative contributions to poor synthesis quality were not isolated.

## Reproducibility

Training requires the exported audio, aligned acoustic features, and the custom model definitions from the training notebook. Model checkpoints alone are insufficient to reproduce training or inference. The final training notebook/source will be added separately when recovered. Pretrained MMS-VITS weights and original datasets are excluded from version control.

## Acknowledgments

- FastSpeech 2: Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*.
- Reference code: https://github.com/ming024/FastSpeech2
- Meta MMS-TTS pretrained checkpoints: https://huggingface.co/facebook/mms-tts-eng and https://huggingface.co/facebook/mms-tts-tel

## Notebook recovery

Due to an unexpected Kaggle notebook crash, some of the original code and execution history were lost. Luckily, the project files, trained models, graphs, and audio results were saved in a backup ZIP. Most of the missing code was also recovered from the notebook history. The final notebook is being rebuilt using this recovered code, so some parts may differ from the original version.
