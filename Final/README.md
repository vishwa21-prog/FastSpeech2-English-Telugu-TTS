# Final implementation and results

The Final notebook is a recovered copy from Kaggle. Some earlier training cells are missing from that notebook after a session crash. The complete available execution-history code is preserved in [`recovery/Recovered_Notebook_Cells.py`](recovery/Recovered_Notebook_Cells.py), including the custom FastSpeech 2-style architecture and training routines.

The custom trained acoustic models did **not** produce intelligible synthesized speech. The audible English and Telugu samples in the backed-up `Results/English/pretrained_baseline/` and `Results/Telugu/pretrained_baseline/` folders were produced using **pretrained Meta MMS-VITS**, not FastSpeech 2.

The original backup ZIP includes checkpoints, figures, and WAV samples. Large checkpoints and the original dataset recordings are not included in this GitHub repository. The notebook uploaded here has outputs cleared to keep the repository manageable.
