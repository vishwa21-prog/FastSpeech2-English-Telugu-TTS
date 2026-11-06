

# ===== CELL 1 =====

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
ROOT.mkdir(parents=True, exist_ok=True)

repo = ROOT / "FastSpeech2"

if not (repo / "train.py").exists():
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "https://github.com/ming024/FastSpeech2.git",
            str(repo)
        ],
        check=True
    )

os.chdir(repo)

import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE")
print("Repository:", repo)
print("Git commit:", subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    text=True
).strip())

print("\nKey dependencies:")
for name in ["librosa", "numpy", "scipy", "yaml", "g2p_en", "pyworld", "tgt", "phonemizer"]:
    print(name, "AVAILABLE" if importlib.util.find_spec(name) else "MISSING")

print("\nExisting Kaggle datasets:")
data_root = Path("/kaggle/input")
for p in sorted(data_root.iterdir()) if data_root.exists() else []:
    print(" -", p.name)

# ===== CELL 2 =====

import sys
import subprocess
import importlib.util

if importlib.util.find_spec("datasets") is None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "datasets"
    ])

from huggingface_hub import HfApi

api = HfApi()

datasets_to_check = {
    "English": "ai4bharat/Svarah",
    "Telugu": "SPRINGLab/IndicTTS_Telugu"
}

for lang, repo_id in datasets_to_check.items():
    try:
        info = api.dataset_info(repo_id)
        print(f"{lang}: ACCESSIBLE")
        print("Dataset:", repo_id)
        print("Private:", info.private)
        print("Files:", len(info.siblings))
    except Exception as e:
        print(f"{lang}: ERROR")
        print(type(e).__name__, str(e)[:300])

# ===== CELL 3 =====

from huggingface_hub import HfApi
import json

api = HfApi()

repos = {
    "English": "ai4bharat/Svarah",
    "Telugu": "SPRINGLab/IndicTTS_Telugu"
}

for lang, repo in repos.items():
    print(f"\n{'='*50}")
    print(lang, repo)

    try:
        info = api.dataset_info(repo, files_metadata=True)

        for f in info.siblings:
            size = getattr(f, "size", None)
            print(
                f.rfilename,
                f"{size / 1024**2:.1f} MB" if size is not None else ""
            )

        try:
            splits = api.dataset_info(repo).card_data
            print("Dataset card:", str(splits)[:500])
        except Exception:
            pass

    except Exception as e:
        print(type(e).__name__, str(e)[:400])

# ===== CELL 4 =====

from datasets import load_dataset

repos = {
    "English": ("ai4bharat/Svarah", "test"),
    "Telugu": ("SPRINGLab/IndicTTS_Telugu", "train")
}

for lang, (repo, split) in repos.items():
    print(f"\n{'='*50}")
    print(lang)

    ds = load_dataset(repo, split=split, streaming=True)

    print("Columns:", ds.column_names)

    sample = next(iter(ds))

    for key, value in sample.items():
        if key in ("audio", "audio_filepath"):
            print(key, "Audio field present")
        else:
            print(key, str(value)[:150])

# ===== CELL 5 =====

from huggingface_hub import login, whoami
from getpass import getpass

login(token=getpass("Hugging Face token: "), add_to_git_credential=False)

print("Authenticated as:", whoami()["name"])

# ===== CELL 6 =====

from huggingface_hub import login, whoami
from getpass import getpass

login(token=getpass("Hugging Face token: "), add_to_git_credential=False)

print("Authenticated as:", whoami()["name"])

# ===== CELL 7 =====

from datasets import load_dataset

repos = {
    "English": ("ai4bharat/Svarah", "test"),
    "Telugu": ("SPRINGLab/IndicTTS_Telugu", "train")
}

for lang, (repo, split) in repos.items():
    print(f"\n{'='*50}")
    print(lang)

    ds = load_dataset(repo, split=split, streaming=True)

    print("Columns:", ds.column_names)

    sample = next(iter(ds))

    for key, value in sample.items():
        if key in ("audio", "audio_filepath"):
            print(key, "Audio field present")
        else:
            print(key, str(value)[:150])

# ===== CELL 8 =====

from huggingface_hub import login, whoami
from getpass import getpass

login(token=getpass("Hugging Face token: "), add_to_git_credential=False)

print("Authenticated as:", whoami()["name"])

# ===== CELL 9 =====

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami

secrets = UserSecretsClient()
token = secrets.get_secret("HF_TOKEN")

login(token=token, add_to_git_credential=False)

print("Authenticated as:", whoami()["name"])

# ===== CELL 10 =====

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami

secrets = UserSecretsClient()
token = secrets.get_secret("HF_TOKEN")

login(token=token, add_to_git_credential=False)

print("Authenticated as:", whoami()["name"])

# ===== CELL 11 =====

from datasets import load_dataset

repos = {
    "English": ("ai4bharat/Svarah", "test"),
    "Telugu": ("SPRINGLab/IndicTTS_Telugu", "train")
}

for lang, (repo, split) in repos.items():
    print(f"\n{'='*50}")
    print(lang)

    ds = load_dataset(repo, split=split, streaming=True)

    print("Columns:", ds.column_names)

    sample = next(iter(ds))

    for key, value in sample.items():
        if key in ("audio", "audio_filepath"):
            print(key, "Audio field present")
        else:
            print(key, str(value)[:150])

# ===== CELL 12 =====

import os, csv, json, random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
SR = 22050
SEED = 2026
TARGET = {"train": 5 * 3600, "val": 3600}

sources = {
    "English": ("ai4bharat/Svarah", "test", "audio_filepath"),
    "Telugu": ("SPRINGLab/IndicTTS_Telugu", "train", "audio")
}

summary = {}

for lang, (repo, split, audio_col) in sources.items():
    print(f"\nProcessing {lang}...", flush=True)

    ds = load_dataset(repo, split=split, streaming=True, token=True)
    ds = ds.cast_column(audio_col, Audio(decode=False))

    out = ROOT / lang
    for part in TARGET:
        (out / "wavs" / part).mkdir(parents=True, exist_ok=True)

    rows = {"train": [], "val": []}
    seconds = {"train": 0.0, "val": 0.0}
    rng = random.Random(SEED)
    count = 0

    for example in ds:
        if all(seconds[p] >= TARGET[p] for p in TARGET):
            break

        text = str(example.get("text", "")).strip()
        if not text:
            continue

        audio = example[audio_col]

        try:
            if audio.get("bytes") is not None:
                import io
                wav, sr = sf.read(io.BytesIO(audio["bytes"]))
            elif audio.get("path"):
                wav, sr = sf.read(audio["path"])
            else:
                continue

            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            duration = len(wav) / sr
            if not 1.0 <= duration <= 30.0:
                continue

            wav = librosa.resample(
                wav.astype(np.float32),
                orig_sr=sr,
                target_sr=SR
            )

            available = [p for p in TARGET if seconds[p] < TARGET[p]]
            if not available:
                break

            part = rng.choice(available)
            filename = f"{lang.lower()}_{count:06d}.wav"
            path = out / "wavs" / part / filename

            sf.write(path, wav, SR)

            rows[part].append({
                "audio": str(path),
                "text": text,
                "duration": round(len(wav) / SR, 4)
            })

            seconds[part] += len(wav) / SR
            count += 1

            if count % 250 == 0:
                print(
                    f"{lang}: {count} files | "
                    f"train {seconds['train']/3600:.2f}h | "
                    f"val {seconds['val']/3600:.2f}h",
                    flush=True
                )

        except Exception as e:
            if count < 5:
                print("Skipped:", type(e).__name__, str(e)[:100])
            continue

    for part in TARGET:
        manifest = out / f"{part}_manifest.csv"
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["audio", "text", "duration"])
            writer.writeheader()
            writer.writerows(rows[part])

    summary[lang] = {
        part: {
            "samples": len(rows[part]),
            "hours": round(seconds[part] / 3600, 4)
        }
        for part in TARGET
    }

    print(lang, summary[lang], flush=True)

with open(ROOT / "data_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nDATA SUMMARY")
print(json.dumps(summary, indent=2))

# ===== CELL 13 =====

import shutil
import subprocess
import importlib.util

for cmd in ["espeak-ng", "espeak", "mfa", "ffmpeg"]:
    print(f"{cmd}: {shutil.which(cmd) or 'NOT FOUND'}")

for pkg in ["torch", "torchaudio", "transformers", "ctc_forced_aligner",
            "phonemizer", "pyworld", "tgt", "g2p_en"]:
    try:
        found = importlib.util.find_spec(pkg) is not None
        print(f"{pkg}: {'AVAILABLE' if found else 'MISSING'}")
    except Exception as e:
        print(pkg, type(e).__name__)

print("\nAvailable disk space:")
subprocess.run(["df", "-h", "/kaggle/working"], check=True)

# ===== CELL 14 =====

import sys
import subprocess

packages = [
    "ctc-forced-aligner==1.0.2",
    "g2p-en",
    "pyworld",
    "tgt",
    "phonemizer",
    "Unidecode",
    "uroman"
]

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", *packages],
    check=True
)

import importlib.util

for name in ["ctc_forced_aligner", "g2p_en", "pyworld", "tgt", "phonemizer", "uroman"]:
    print(name, "AVAILABLE" if importlib.util.find_spec(name) else "MISSING")

# ===== CELL 15 =====

import csv
import inspect
from pathlib import Path

import torch
import soundfile as sf
import librosa

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

print("CUDA:", torch.cuda.is_available())

from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results
)

print("\nAlignment function signatures:")
for fn in [
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results
]:
    print(fn.__name__, inspect.signature(fn))

samples = {}

for lang in ["English", "Telugu"]:
    manifest = ROOT / lang / "train_manifest.csv"

    with open(manifest, encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    wav, sr = sf.read(row["audio"])

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    samples[lang] = {
        "audio": row["audio"],
        "text": row["text"],
        "duration": len(wav) / sr
    }

    print(f"\n{lang}")
    print("Text:", row["text"])
    print("Duration:", round(len(wav) / sr, 2), "seconds")
    print("Sample rate:", sr)

print("\nLoading multilingual alignment model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

model, tokenizer = load_alignment_model(
    device,
    dtype=torch.float32
)

print("Alignment model loaded on:", device)

# ===== CELL 16 =====

import ctc_forced_aligner
import importlib
import inspect
import pkgutil
from pathlib import Path

pkg = ctc_forced_aligner

print("Package:", pkg.__file__)

print("\nTop-level exports:")
print([x for x in dir(pkg) if not x.startswith("_")])

print("\nAvailable modules:")
for m in pkgutil.iter_modules(pkg.__path__):
    print(" -", m.name)

print("\nAlignment-related functions:")
for m in pkgutil.iter_modules(pkg.__path__):
    if m.name.startswith("_"):
        continue

    try:
        module = importlib.import_module(
            f"ctc_forced_aligner.{m.name}"
        )

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if any(k in name.lower() for k in [
                "align", "load", "emission", "token", "span", "text"
            ]):
                try:
                    print(f"{m.name}.{name}{inspect.signature(obj)}")
                except Exception:
                    print(f"{m.name}.{name}")
    except Exception as e:
        print(m.name, ":", type(e).__name__, str(e)[:150])

print("\nDataset files:")
ROOT = Path("/kaggle/working/FastSpeech2_Submission")

for lang in ["English", "Telugu"]:
    manifest = ROOT / lang / "train_manifest.csv"
    print(lang, "manifest exists:", manifest.exists())

# ===== CELL 17 =====

import inspect
import ctc_forced_aligner as ctc

names = [
    "Alignment",
    "AlignmentTorch",
    "forced_align",
    "generate_emissions",
    "preprocess_text",
    "get_alignments",
    "get_spans",
    "postprocess_results",
    "get_uroman_tokens"
]

for name in names:
    obj = getattr(ctc, name, None)
    if obj is None:
        continue

    print(f"\n{'=' * 55}")
    print(name)

    try:
        print("Signature:", inspect.signature(obj))
    except Exception as e:
        print("Signature unavailable:", str(e)[:120])

    if name in ["Alignment", "AlignmentTorch", "forced_align"]:
        try:
            print("Documentation:", inspect.getdoc(obj)[:1500])
        except Exception:
            pass

print("\nAlignment methods:")
for cls_name in ["Alignment", "AlignmentTorch"]:
    cls = getattr(ctc, cls_name)
    for name, obj in inspect.getmembers(cls):
        if not name.startswith("_") and callable(obj):
            try:
                print(cls_name, name, inspect.signature(obj))
            except Exception:
                print(cls_name, name)

# ===== CELL 18 =====

import csv
import inspect
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import ctc_forced_aligner as ctc

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

print("Alignment package:", ctc.__file__)

print("\nModel and tokenizer API:")
for name in ["ensure_onnx_model", "Tokenizer", "preprocess_text"]:
    obj = getattr(ctc, name)
    print(name, inspect.signature(obj))

print("\nAlignment model constants:")
print("Model URL:", ctc.MODEL_URL)
print("Sampling frequency:", ctc.SAMPLING_FREQ)

print("\nSample recordings:")

for lang in ["English", "Telugu"]:
    manifest = ROOT / lang / "train_manifest.csv"

    with open(manifest, encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    audio, sr = sf.read(row["audio"])

    print(f"\n{lang}")
    print("Text:", row["text"])
    print("Duration:", round(len(audio) / sr, 2), "seconds")
    print("Sample rate:", sr)
    print("Audio shape:", audio.shape)

# ===== CELL 19 =====

import csv
import tempfile
from pathlib import Path

import ctc_forced_aligner as ctc

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
ALIGN_DIR = ROOT / "alignment_test"
ALIGN_DIR.mkdir(exist_ok=True)

aligner = ctc.Alignment(
    str(ALIGN_DIR / "ctc_model.onnx")
)

for lang, code in [("English", "eng"), ("Telugu", "tel")]:
    manifest = ROOT / lang / "train_manifest.csv"

    with open(manifest, encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    text_file = ALIGN_DIR / f"{lang.lower()}_text.txt"
    output_file = ALIGN_DIR / f"{lang.lower()}_alignment.srt"

    text_file.write_text(row["text"], encoding="utf-8")

    print(f"\nAligning {lang}...", flush=True)

    try:
        aligner.generate_srt(
            input_audio_path=row["audio"],
            input_text_path=str(text_file),
            output_srt_path=str(output_file),
            language=code,
            batch_size=1
        )

        if output_file.exists():
            print(output_file.read_text(encoding="utf-8")[:1500])
        else:
            print("No alignment file generated.")

    except Exception as e:
        print(type(e).__name__, str(e)[:1000])

# ===== CELL 20 =====

import csv
import inspect
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import ctc_forced_aligner as ctc

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

print("Alignment package:", ctc.__file__)

print("\nModel and tokenizer API:")
for name in ["ensure_onnx_model", "Tokenizer", "preprocess_text"]:
    obj = getattr(ctc, name)
    print(name, inspect.signature(obj))

print("\nAlignment model constants:")
print("Model URL:", ctc.MODEL_URL)
print("Sampling frequency:", ctc.SAMPLING_FREQ)

print("\nSample recordings:")

for lang in ["English", "Telugu"]:
    manifest = ROOT / lang / "train_manifest.csv"

    with open(manifest, encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    audio, sr = sf.read(row["audio"])

    print(f"\n{lang}")
    print("Text:", row["text"])
    print("Duration:", round(len(audio) / sr, 2), "seconds")
    print("Sample rate:", sr)
    print("Audio shape:", audio.shape)

# ===== CELL 21 =====

import csv
import tempfile
from pathlib import Path

import ctc_forced_aligner as ctc

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
ALIGN_DIR = ROOT / "alignment_test"
ALIGN_DIR.mkdir(exist_ok=True)

aligner = ctc.Alignment(
    str(ALIGN_DIR / "ctc_model.onnx")
)

for lang, code in [("English", "eng"), ("Telugu", "tel")]:
    manifest = ROOT / lang / "train_manifest.csv"

    with open(manifest, encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    text_file = ALIGN_DIR / f"{lang.lower()}_text.txt"
    output_file = ALIGN_DIR / f"{lang.lower()}_alignment.srt"

    text_file.write_text(row["text"], encoding="utf-8")

    print(f"\nAligning {lang}...", flush=True)

    try:
        aligner.generate_srt(
            input_audio_path=row["audio"],
            input_text_path=str(text_file),
            output_srt_path=str(output_file),
            language=code,
            batch_size=1
        )

        if output_file.exists():
            print(output_file.read_text(encoding="utf-8")[:1500])
        else:
            print("No alignment file generated.")

    except Exception as e:
        print(type(e).__name__, str(e)[:1000])

# ===== CELL 22 =====

import csv
from pathlib import Path

import ctc_forced_aligner as ctc

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
ALIGN_DIR = ROOT / "alignment_test"
ALIGN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ALIGN_DIR / "ctc_model.onnx"

# Download the model only if it is missing.
if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
    print("Downloading CTC alignment model...", flush=True)
    ctc.ensure_onnx_model(str(MODEL_PATH), ctc.MODEL_URL)

print("Model size:", round(MODEL_PATH.stat().st_size / 1024**2, 1), "MB")

aligner = ctc.Alignment(str(MODEL_PATH))

for lang, code in [("English", "eng"), ("Telugu", "tel")]:
    manifest = ROOT / lang / "train_manifest.csv"

    with open(manifest, encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    text_file = ALIGN_DIR / f"{lang.lower()}_text.txt"
    output_file = ALIGN_DIR / f"{lang.lower()}_alignment.srt"

    text_file.write_text(row["text"], encoding="utf-8")

    print(f"\nAligning {lang}...", flush=True)

    try:
        aligner.generate_srt(
            input_audio_path=row["audio"],
            input_text_path=str(text_file),
            output_srt_path=str(output_file),
            language=code,
            batch_size=1
        )

        if output_file.exists():
            print(output_file.read_text(encoding="utf-8")[:1500])
        else:
            print("No alignment file generated.")

    except Exception as e:
        print(type(e).__name__, str(e)[:1000])

# ===== CELL 23 =====

import csv
from pathlib import Path
import ctc_forced_aligner as ctc

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
ALIGN_DIR = ROOT / "alignment_test"
MODEL_PATH = ALIGN_DIR / "ctc_model.onnx"

assert MODEL_PATH.exists(), "Previously downloaded ONNX model is missing"

aligner = ctc.Alignment(str(MODEL_PATH))

for lang, code in [("English", "eng"), ("Telugu", "tel")]:
    with open(ROOT / lang / "train_manifest.csv", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    text_path = ALIGN_DIR / f"{lang.lower()}_text.txt"
    srt_path = ALIGN_DIR / f"{lang.lower()}_alignment.srt"

    text_path.write_text(row["text"], encoding="utf-8")

    print(f"\nAligning {lang}...", flush=True)

    try:
        aligner.generate_srt(
            row["audio"],
            str(text_path),
            str(srt_path),
            code
        )

        if srt_path.exists():
            print(srt_path.read_text(encoding="utf-8")[:1500])
        else:
            print("No SRT file generated")

    except Exception as e:
        print(type(e).__name__, str(e)[:1000])

# ===== CELL 24 =====

import inspect
import ctc_forced_aligner as ctc

print("Alignment.generate_srt implementation:")
print(inspect.getsource(ctc.Alignment.generate_srt))

print("\nAlignment constructor:")
print(inspect.getsource(ctc.Alignment.__init__))

print("\nAlignmentTorch.generate_srt implementation:")
print(inspect.getsource(ctc.AlignmentTorch.generate_srt))

# ===== CELL 25 =====

import inspect
import ctc_forced_aligner as ctc

for name in ["_generate_subtitle", "align", "get_alignments"]:
    fn = getattr(ctc, name, None)

    print(f"\n{'=' * 50}")
    print(name)

    if fn is None:
        print("Not available")
        continue

    print("Signature:", inspect.signature(fn))
    print(inspect.getsource(fn)[:9000])

# ===== CELL 26 =====

import csv
from pathlib import Path

import numpy as np
import ctc_forced_aligner as ctc

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
MODEL_PATH = ROOT / "alignment_test" / "ctc_model.onnx"

assert MODEL_PATH.exists(), "CTC model is missing"

aligner = ctc.Alignment(str(MODEL_PATH))

results = {}

for lang, code in [("English", "eng"), ("Telugu", "tel")]:
    with open(ROOT / lang / "train_manifest.csv", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    print(f"\n{'=' * 50}")
    print("Language:", lang)
    print("Text:", row["text"])

    try:
        audio = ctc.load_audio(row["audio"])

        emissions, stride = ctc.generate_emissions(
            aligner.model,
            audio,
            batch_size=1
        )

        tokens, text_starred = ctc.preprocess_text(
            row["text"],
            romanize=True,
            language=code
        )

        segments, scores, blank = ctc.get_alignments(
            emissions,
            tokens,
            aligner.tokenizer
        )

        spans = ctc.get_spans(tokens, segments, blank)

        timestamps = ctc.postprocess_results(
            text_starred,
            spans,
            stride,
            scores
        )

        results[lang] = timestamps

        print("Emission shape:", np.shape(emissions))
        print("Number of aligned segments:", len(timestamps))

        for item in timestamps[:12]:
            print(item)

    except Exception as e:
        print("Alignment error:", type(e).__name__, str(e)[:500])
        raise

# ===== CELL 27 =====

import csv
from pathlib import Path
import ctc_forced_aligner as ctc

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
MODEL_PATH = ROOT / "alignment_test" / "ctc_model.onnx"

aligner = ctc.Alignment(str(MODEL_PATH))

with open(ROOT / "Telugu" / "train_manifest.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

row = next(
    r for r in rows
    if 5 <= len(r["text"].split()) <= 15
    and 3 <= float(r["duration"]) <= 15
)

print("Text:", row["text"])
print("Audio duration:", row["duration"], "seconds")

audio = ctc.load_audio(row["audio"])

emissions, stride = ctc.generate_emissions(
    aligner.model, audio, batch_size=1
)

tokens, text_starred = ctc.preprocess_text(
    row["text"], romanize=True, language="tel"
)

segments, scores, blank = ctc.get_alignments(
    emissions, tokens, aligner.tokenizer
)

spans = ctc.get_spans(tokens, segments, blank)

timestamps = ctc.postprocess_results(
    text_starred, spans, stride, scores
)

print("\nAligned words:", len(timestamps))

for item in timestamps:
    print(item)

# ===== CELL 28 =====

import csv
from pathlib import Path

import numpy as np
import ctc_forced_aligner as ctc

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
MODEL_PATH = ROOT / "alignment_test" / "ctc_model.onnx"

assert MODEL_PATH.exists(), "CTC model is missing"

aligner = ctc.Alignment(str(MODEL_PATH))

results = {}

for lang, code in [("English", "eng"), ("Telugu", "tel")]:    with open(ROOT / lang / "train_manifest.csv", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    print(f"\n{'=' * 50}")
    print("Language:", lang)
    print("Text:", row["text"])

    try:
        audio = ctc.load_audio(row["audio"])

        emissions, stride = ctc.generate_emissions(
            aligner.model,
            audio,
            batch_size=1
        )

        tokens, text_starred = ctc.preprocess_text(
            row["text"],
            romanize=True,
            language=code
        )

        segments, scores, blank = ctc.get_alignments(
            emissions,
            tokens,
            aligner.tokenizer
        )

        spans = ctc.get_spans(tokens, segments, blank)

        timestamps = ctc.postprocess_results(
            text_starred,
            spans,
            stride,
            scores
        )

        results[lang] = timestamps

        print("Emission shape:", np.shape(emissions))
        print("Number of aligned segments:", len(timestamps))

        for item in timestamps[:12]:
            print(item)

    except Exception as e:
        print("Alignment error:", type(e).__name__, str(e)[:500])
        raise

# ===== CELL 29 =====

import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

if shutil.which("espeak-ng") is None:
    subprocess.run(
        ["apt-get", "update", "-qq"],
        check=True,
        stdout=subprocess.DEVNULL
    )
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "espeak-ng"],
        check=True,
        stdout=subprocess.DEVNULL
    )

frontend_code = r'''
import re
import unicodedata
from functools import lru_cache
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

LANGUAGES = {
    "English": "en-us",
    "Telugu": "te",
}

@lru_cache(maxsize=2)
def backend(language):
    return EspeakBackend(
        language=LANGUAGES[language],
        preserve_punctuation=False,
        with_stress=False,
        language_switch="remove-flags"
    )

def phonemize_words(text, language):
    text = unicodedata.normalize("NFC", text.strip())
    text = re.sub(r"\s+", " ", text)

    if not text:
        return []

    words = text.split(" ")
    result = []

    sep = Separator(phone="|", word=" ", syllable="")

    for word in words:
        output = backend(language).phonemize(
            [word],
            separator=sep,
            strip=True,
            njobs=1
        )[0]

        phones = [p for p in output.split("|") if p.strip()]
        phones = [p.strip() for p in phones]

        if phones:
            result.append(phones)

    return result

def phonemize_text(text, language):
    words = phonemize_words(text, language)
    return [phone for word in words for phone in word]
'''

frontend_path = ROOT / "phoneme_frontend.py"
frontend_path.write_text(frontend_code, encoding="utf-8")

sys.path.insert(0, str(ROOT))

from phoneme_frontend import phonemize_words, phonemize_text

examples = {
    "English": "The orchestra changed its rhythm unexpectedly.",
    "Telugu": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది."
}

for language, sentence in examples.items():
    print(f"\n{language}")
    print("Original:", sentence)

    try:
        words = phonemize_words(sentence, language)
        phones = phonemize_text(sentence, language)

        print("Phonemes by word:", words)
        print("Total phonemes:", len(phones))
        print("Flattened:", phones[:60])

        assert len(words) == len(sentence.split())
        assert len(phones) > 0

    except Exception as e:
        print("ERROR:", type(e).__name__, str(e)[:500])
        raise

print("\nFrontend saved:", frontend_path)

# ===== CELL 30 =====

import csv
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
import librosa
import pyworld as pw
import ctc_forced_aligner as ctc

from phoneme_frontend import phonemize_words

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
MODEL = ROOT / "alignment_test" / "ctc_model.onnx"
OUT = ROOT / "acoustic_features"

SR = 22050
HOP = 256
FFT = 1024
MELS = 80

assert MODEL.exists(), "Previously downloaded CTC model is missing"

aligner = ctc.Alignment(str(MODEL))

def word_to_phone_durations(timestamps, words, frames, seconds):
    if len(timestamps) != len(words):
        raise ValueError(
            f"Word count mismatch: {len(timestamps)} vs {len(words)}"
        )

    if not words or frames <= 0:
        raise ValueError("Empty phonemes or mel frames")

    boundaries = [0.0]

    for a, b in zip(timestamps[:-1], timestamps[1:]):
        midpoint = (
            float(a["end"]) + float(b["start"])
        ) / 2
        boundaries.append(midpoint)

    boundaries.append(seconds)

    boundaries = np.clip(boundaries, 0, seconds)
    boundaries = np.maximum.accumulate(boundaries)

    frame_boundaries = np.rint(
        np.asarray(boundaries) / seconds * frames
    ).astype(int)

    frame_boundaries[0] = 0
    frame_boundaries[-1] = frames

    durations = []

    for i, phones in enumerate(words):
        count = frame_boundaries[i + 1] - frame_boundaries[i]
        n = len(phones)

        if count < n:
            raise ValueError("Insufficient frames for phonemes")

        q, r = divmod(count, n)
        durations.extend(
            [q + 1] * r + [q] * (n - r)
        )

    durations = np.asarray(durations, dtype=np.int32)

    if durations.sum() != frames:
        raise ValueError("Duration and mel lengths differ")

    return durations


def acoustic_features(wav):
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SR,
        n_fft=FFT,
        hop_length=HOP,
        win_length=FFT,
        n_mels=MELS,
        fmin=0,
        fmax=8000,
        center=True,
        power=2.0
    )

    mel = np.log(np.maximum(mel, 1e-8)).T.astype(np.float32)
    frames = len(mel)

    stft = librosa.stft(
        wav,
        n_fft=FFT,
        hop_length=HOP,
        center=True
    )

    energy = np.sqrt(
        np.sum(np.abs(stft) ** 2, axis=0)
    ).astype(np.float32)

    audio64 = np.ascontiguousarray(wav, dtype=np.float64)

    f0, times = pw.dio(
        audio64,
        SR,
        frame_period=HOP / SR * 1000
    )

    f0 = pw.stonemask(audio64, f0, times, SR)

    if len(f0) != frames:
        f0 = np.interp(
            np.linspace(0, 1, frames),
            np.linspace(0, 1, len(f0)),
            f0
        )

    if len(energy) != frames:
        energy = np.interp(
            np.linspace(0, 1, frames),
            np.linspace(0, 1, len(energy)),
            energy
        )

    return (
        mel,
        np.asarray(f0, dtype=np.float32),
        np.asarray(energy, dtype=np.float32)
    )


def process_sample(row, lang, code, output):
    wav, sr = librosa.load(
        row["audio"],
        sr=SR,
        mono=True
    )

    text = row["text"].strip()
    words = phonemize_words(text, lang)
    phones = [p for word in words for p in word]

    if not phones:
        raise ValueError("No phonemes")

    audio16 = ctc.load_audio(row["audio"])

    emissions, stride = ctc.generate_emissions(
        aligner.model,
        audio16,
        batch_size=1
    )

    tokens, starred = ctc.preprocess_text(
        text,
        romanize=True,
        language=code
    )

    segments, scores, blank = ctc.get_alignments(
        emissions,
        tokens,
        aligner.tokenizer
    )

    spans = ctc.get_spans(tokens, segments, blank)

    timestamps = ctc.postprocess_results(
        starred,
        spans,
        stride,
        scores
    )

    mel, pitch, energy = acoustic_features(wav)

    duration = word_to_phone_durations(
        timestamps,
        words,
        len(mel),
        len(wav) / SR
    )

    if len(duration) != len(phones):
        raise ValueError("Phoneme-duration length mismatch")

    if not np.isfinite(mel).all():
        raise ValueError("Invalid mel values")

    if not np.isfinite(pitch).all():
        raise ValueError("Invalid pitch values")

    if not np.isfinite(energy).all():
        raise ValueError("Invalid energy values")

    np.savez(
        output.with_suffix(".npz"),
        mel=mel,
        pitch=pitch,
        energy=energy,
        duration=duration
    )

    metadata = {
        "audio": row["audio"],
        "text": text,
        "phones": phones,
        "frames": len(mel),
        "duration_seconds": len(wav) / SR,
        "alignment_type": "CTC word-level; estimated within-word phoneme durations"
    }

    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8"
    )

    return len(phones), len(mel)


start = time.time()

for lang, code in [("English", "eng"), ("Telugu", "tel")]:
    print(f"\nProcessing {lang}", flush=True)

    for split in ["train", "val"]:
        folder = OUT / lang / split
        folder.mkdir(parents=True, exist_ok=True)

        manifest = ROOT / lang / f"{split}_manifest.csv"

        with open(manifest, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        completed = 0
        failed = 0

        for i, row in enumerate(rows):
            stem = Path(row["audio"]).stem
            target = folder / stem

            if (
                target.with_suffix(".npz").exists()
                and target.with_suffix(".json").exists()
            ):
                completed += 1
                continue

            try:
                nphones, nframes = process_sample(
                    row, lang, code, target
                )

                completed += 1

                if completed <= 3:
                    print(
                        stem,
                        "phonemes:", nphones,
                        "frames:", nframes,
                        flush=True
                    )

            except Exception as e:
                failed += 1

                if failed <= 5:
                    print(
                        "Skipped", stem,
                        type(e).__name__,
                        str(e)[:160],
                        flush=True
                    )

            if (i + 1) % 25 == 0:
                print(
                    f"{lang}/{split}: "
                    f"{i+1}/{len(rows)} examined, "
                    f"{completed} saved, "
                    f"{failed} skipped, "
                    f"{(time.time()-start)/60:.1f} min",
                    flush=True
                )

        print(
            f"{lang}/{split}: {completed} saved, "
            f"{failed} skipped",
            flush=True
        )

    counter = Counter()

    for meta_file in (OUT / lang).rglob("*.json"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        counter.update(meta["phones"])

    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }

    for phone in sorted(counter):
        if phone not in vocab:
            vocab[phone] = len(vocab)

    (OUT / lang / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"{lang} vocabulary: {len(vocab)} tokens", flush=True)

print(
    "\nFeature extraction finished in",
    round((time.time() - start) / 60, 1),
    "minutes"
)

# ===== CELL 31 =====

from pathlib import Path
import json
import numpy as np

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
OUT = ROOT / "acoustic_features"

for lang in ["English", "Telugu"]:
    print(f"\n{lang}")

    for split in ["train", "val"]:
        folder = OUT / lang / split
        files = sorted(folder.glob("*.npz"))

        seconds = 0
        valid = 0

        for path in files:
            meta_path = path.with_suffix(".json")
            if not meta_path.exists():
                continue

            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))

                with np.load(path) as data:
                    mel = data["mel"]
                    duration = data["duration"]

                    if (
                        len(duration) == len(meta["phones"])
                        and int(duration.sum()) == len(mel)
                    ):
                        valid += 1
                        seconds += meta["duration_seconds"]

            except Exception:
                continue

        print(
            f"{split}: {valid} valid files, "
            f"{seconds / 3600:.3f} hours"
        )

# ===== CELL 32 =====

import csv, json, time
from pathlib import Path
from collections import Counter

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
OUT = ROOT / "acoustic_features"

LIMITS = {
    "English": {"train": 0, "val": 30},
    "Telugu": {"train": 120, "val": 30}
}

start = time.time()

for lang, code in [("English", "eng"), ("Telugu", "tel")]:
    for split, limit in LIMITS[lang].items():
        folder = OUT / lang / split
        folder.mkdir(parents=True, exist_ok=True)

        manifest = ROOT / lang / f"{split}_manifest.csv"

        with open(manifest, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        existing = len(list(folder.glob("*.npz")))
        saved = 0
        failed = 0

        print(f"\n{lang}/{split}: {existing} existing", flush=True)

        if limit == 0:
            continue

        for row in rows:
            if saved >= limit:
                break

            stem = Path(row["audio"]).stem
            target = folder / stem

            if (
                target.with_suffix(".npz").exists()
                and target.with_suffix(".json").exists()
            ):
                continue

            try:
                process_sample(row, lang, code, target)
                saved += 1
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print("Skipped:", stem, str(e)[:120], flush=True)

            if saved > 0 and saved % 10 == 0:
                print(
                    f"{lang}/{split}: {saved}/{limit} new files | "
                    f"{(time.time()-start)/60:.1f} min",
                    flush=True
                )

        print(
            f"{lang}/{split}: {saved} new files, "
            f"{failed} skipped",
            flush=True
        )

for lang in ["English", "Telugu"]:
    counter = Counter()

    for path in (OUT / lang).rglob("*.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        counter.update(meta["phones"])

    vocab = {"<pad>": 0, "<unk>": 1}

    for phone in sorted(counter):
        if phone not in vocab:
            vocab[phone] = len(vocab)

    path = OUT / lang / "vocab.json"
    path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(lang, "vocabulary size:", len(vocab))

print("Total elapsed minutes:", round((time.time()-start)/60, 1))

# ===== CELL 33 =====

import json, random, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
FEATURES = ROOT / "acoustic_features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(2026)
random.seed(2026)
np.random.seed(2026)

class TTSDataset(Dataset):
    def __init__(self, lang, split):
        self.folder = FEATURES / lang / split
        self.vocab = json.loads(
            (FEATURES / lang / "vocab.json").read_text(encoding="utf-8")
        )
        self.files = [
            p for p in sorted(self.folder.glob("*.npz"))
            if p.with_suffix(".json").exists()
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))

        with np.load(p) as z:
            mel = z["mel"].astype("float32")
            dur = z["duration"].astype("int64")
            pitch = z["pitch"].astype("float32").reshape(-1)
            energy = z["energy"].astype("float32").reshape(-1)

        if mel.ndim != 2:
            raise ValueError(f"Invalid mel shape: {p} {mel.shape}")

        if mel.shape[0] == 80 and mel.shape[1] != 80:
            mel = mel.T

        if mel.shape[1] != 80:
            raise ValueError(f"Expected 80 mel bins: {p} {mel.shape}")

        n = len(mel)

        def resize(x):
            if len(x) == n:
                return x
            if len(x) == 0:
                return np.zeros(n, dtype="float32")
            return np.interp(
                np.linspace(0, len(x) - 1, n),
                np.arange(len(x)),
                x
            ).astype("float32")

        pitch = resize(pitch)
        energy = resize(energy)

        phones = [
            self.vocab.get(x, 1)
            for x in meta["phones"]
        ]

        if len(phones) != len(dur) or int(dur.sum()) != n:
            raise ValueError(f"Duration mismatch: {p}")

        return (
            torch.tensor(phones, dtype=torch.long),
            torch.tensor(dur, dtype=torch.long),
            torch.from_numpy(mel),
            torch.from_numpy(pitch),
            torch.from_numpy(energy)
        )

def collate(batch):
    bs = len(batch)
    max_text = max(len(x[0]) for x in batch)
    max_mel = max(len(x[2]) for x in batch)

    text = torch.zeros(bs, max_text, dtype=torch.long)
    dur = torch.zeros(bs, max_text, dtype=torch.long)
    mel = torch.zeros(bs, max_mel, 80)
    pitch = torch.zeros(bs, max_mel)
    energy = torch.zeros(bs, max_mel)
    text_len = []
    mel_len = []

    for i, (t, d, m, p, e) in enumerate(batch):
        text[i, :len(t)] = t
        dur[i, :len(d)] = d
        mel[i, :len(m)] = m
        pitch[i, :len(p)] = p
        energy[i, :len(e)] = e
        text_len.append(len(t))
        mel_len.append(len(m))

    return (
        text, dur, mel, pitch, energy,
        torch.tensor(text_len),
        torch.tensor(mel_len)
    )

class Predictor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        # LayerNorm is applied separately in forward.
        self.conv1 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        y = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        y = self.norm1(F.relu(y))
        y = self.conv2(y.transpose(1, 2)).transpose(1, 2)
        y = self.norm2(F.relu(y))
        return self.out(y).squeeze(-1)

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=0)

        layer = lambda: nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(layer(), num_layers=2)
        self.duration_predictor = Predictor(hidden)
        self.pitch_predictor = Predictor(hidden)
        self.energy_predictor = Predictor(hidden)

        self.pitch_embed = nn.Linear(1, hidden)
        self.energy_embed = nn.Linear(1, hidden)

        self.decoder = nn.TransformerEncoder(layer(), num_layers=2)
        self.mel_projection = nn.Linear(hidden, 80)

    def position(self, length, device):
        pos = torch.arange(length, device=device).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.hidden, 2, device=device).float()
            * (-math.log(10000.0) / self.hidden)
        )
        pe = torch.zeros(length, self.hidden, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, text, durations, pitch, energy, text_len, mel_len):
        bs, nt = text.shape
        text_mask = torch.arange(nt, device=text.device)[None, :] >= text_len[:, None]

        x = self.embed(text) + self.position(nt, text.device)
        x = self.encoder(x, src_key_padding_mask=text_mask)

        log_dur = self.duration_predictor(x).masked_fill(text_mask, 0)

        expanded = []
        for i in range(bs):
            length = int(text_len[i])
            z = torch.repeat_interleave(
                x[i, :length],
                durations[i, :length].clamp(min=0),
                dim=0
            )
            expanded.append(z)

        max_len = int(mel_len.max())
        y = x.new_zeros(bs, max_len, self.hidden)

        for i, z in enumerate(expanded):
            n = min(len(z), max_len)
            y[i, :n] = z[:n]

        mel_mask = (
            torch.arange(max_len, device=text.device)[None, :]
            >= mel_len[:, None]
        )

        pred_pitch = self.pitch_predictor(y)
        pred_energy = self.energy_predictor(y)

        y = (
            y
            + self.pitch_embed(pitch.unsqueeze(-1))
            + self.energy_embed(energy.unsqueeze(-1))
            + self.position(max_len, text.device)
        )

        y = self.decoder(y, src_key_padding_mask=mel_mask)
        pred_mel = self.mel_projection(y)

        return pred_mel, log_dur, pred_pitch, pred_energy

def compute_loss(outputs, batch):
    pred_mel, log_dur, pred_pitch, pred_energy = outputs
    text, dur, mel, pitch, energy, text_len, mel_len = batch

    text_mask = (
        torch.arange(text.shape[1], device=DEVICE)[None, :]
        < text_len[:, None]
    )

    mel_mask = (
        torch.arange(mel.shape[1], device=DEVICE)[None, :]
        < mel_len[:, None]
    )

    mel_loss = (
        (pred_mel - mel).abs() * mel_mask.unsqueeze(-1)
    ).sum() / (mel_mask.sum() * 80).clamp(min=1)

    dur_loss = (
        ((log_dur - torch.log1p(dur.float())) ** 2) * text_mask
    ).sum() / text_mask.sum().clamp(min=1)

    pitch_loss = (
        (pred_pitch - pitch).abs() * mel_mask
    ).sum() / mel_mask.sum().clamp(min=1)

    energy_loss = (
        (pred_energy - energy).abs() * mel_mask
    ).sum() / mel_mask.sum().clamp(min=1)

    # Raw pitch and energy magnitudes vary considerably.
    total = (
        mel_loss
        + 0.1 * dur_loss
        + 0.001 * pitch_loss
        + 0.001 * energy_loss
    )

    return total, [
        mel_loss.item(),
        dur_loss.item(),
        pitch_loss.item(),
        energy_loss.item()
    ]

def train_language(lang, epochs=5, batch_size=4):
    train_ds = TTSDataset(lang, "train")
    val_ds = TTSDataset(lang, "val")

    print(f"\n{lang}: {len(train_ds)} train, {len(val_ds)} val", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, collate_fn=collate, num_workers=0
    )

    model = FastSpeech2(len(train_ds.vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    output = ROOT / lang / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)

    results = ROOT / "Results" / lang
    results.mkdir(parents=True, exist_ok=True)

    history = {"train": [], "val": []}
    best = float("inf")

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = tuple(x.to(DEVICE) for x in batch)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                batch[0], batch[1], batch[3],
                batch[4], batch[5], batch[6]
            )
            loss, parts = compute_loss(outputs, batch)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(x.to(DEVICE) for x in batch)
                outputs = model(
                    batch[0], batch[1], batch[3],
                    batch[4], batch[5], batch[6]
                )
                loss, _ = compute_loss(outputs, batch)
                if torch.isfinite(loss):
                    val_losses.append(loss.item())

        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses))

        history["train"].append(tr)
        history["val"].append(va)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "vocab": train_ds.vocab,
            "train_loss": tr,
            "val_loss": va,
            "architecture": "compact_fastspeech2",
            "hidden": 128
        }

        torch.save(checkpoint, output / "last.pt")

        if va < best:
            best = va
            torch.save(checkpoint, output / "best.pt")

        print(
            f"{lang} epoch {epoch}/{epochs} | "
            f"train={tr:.4f} | val={va:.4f} | "
            f"{time.time()-start:.1f}s",
            flush=True
        )

    (results / "history.json").write_text(
        json.dumps(history, indent=2)
    )

    for key, filename in [
        ("train", "training_loss.png"),
        ("val", "validation_loss.png")
    ]:
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, epochs + 1), history[key], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{lang} {key.capitalize()} Loss")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(results / filename, dpi=160)
        plt.close()

    print(f"{lang} checkpoint: {output / 'best.pt'}", flush=True)
    return history

print("Device:", DEVICE)
english_history = train_language("English", epochs=5)
telugu_history = train_language("Telugu", epochs=5)

print("\nBoth training runs finished.")

# ===== CELL 34 =====

import sys, json, math, re
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
sys.path.insert(0, str(ROOT))

from phoneme_frontend import phonemize_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TESTS = {
    "English": [
        "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "The orchestra's conduct during the ceremony was praised by the audience.",
        "The researchers will conduct an experiment to evaluate the proposed method.",
        "The rhythm of the orchestra changed unexpectedly.",
        "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."
    ],
    "Telugu": [
        "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
        "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."
    ]
}

@torch.no_grad()
def synthesize(model, text, vocab, lang):
    phones = phonemize_text(text, lang)
    ids = [vocab.get(p, 1) for p in phones]

    if not ids:
        raise ValueError("No phonemes generated")

    tokens = torch.tensor([ids], device=DEVICE)
    n = tokens.shape[1]

    x = model.embed(tokens) + model.position(n, DEVICE)
    x = model.encoder(x)

    log_duration = model.duration_predictor(x)[0]
    durations = torch.clamp(
        torch.round(torch.expm1(log_duration)).long(),
        min=1,
        max=30
    )

    expanded = torch.repeat_interleave(
        x[0], durations, dim=0
    ).unsqueeze(0)

    frames = expanded.shape[1]

    pred_pitch = model.pitch_predictor(expanded)
    pred_energy = model.energy_predictor(expanded)

    y = (
        expanded
        + model.pitch_embed(pred_pitch.unsqueeze(-1))
        + model.energy_embed(pred_energy.unsqueeze(-1))
        + model.position(frames, DEVICE)
    )

    y = model.decoder(y)
    mel = model.mel_projection(y)[0].cpu().numpy()

    return (
        mel,
        durations.cpu().numpy(),
        pred_pitch[0].cpu().numpy(),
        pred_energy[0].cpu().numpy(),
        phones
    )

def mel_to_audio(mel):
    # Preprocessing used log power-mel spectrograms.
    power_mel = np.exp(
        np.clip(mel.T, -20, 12)
    )

    audio = librosa.feature.inverse.mel_to_audio(
        power_mel,
        sr=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        fmin=0,
        fmax=8000,
        power=2.0,
        n_iter=32
    )

    audio = np.nan_to_num(audio)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = 0.95 * audio / peak

    return audio.astype(np.float32)

for lang, sentences in TESTS.items():
    ckpt = torch.load(
        ROOT / lang / "checkpoints" / "best.pt",
        map_location=DEVICE,
        weights_only=False
    )

    vocab = ckpt["vocab"]
    model = FastSpeech2(
        len(vocab),
        hidden=ckpt["hidden"]
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()

    base = ROOT / "Results" / lang
    audio_dir = base / "test_generated_audios"
    plot_dir = base / "variance_adaptor_plots"

    audio_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {lang}...", flush=True)

    for i, sentence in enumerate(sentences, 1):
        mel, duration, pitch, energy, phones = synthesize(
            model, sentence, vocab, lang
        )

        audio = mel_to_audio(mel)

        sf.write(
            audio_dir / f"test_{i:02d}.wav",
            audio,
            22050
        )

        for name, values in [
            ("duration", duration),
            ("pitch", pitch),
            ("energy", energy)
        ]:
            plt.figure(figsize=(10, 3))
            plt.plot(values)
            plt.title(f"{lang} Test {i}: {name.capitalize()}")
            plt.xlabel(
                "Phoneme index" if name == "duration"
                else "Mel frame"
            )
            plt.ylabel(name.capitalize())
            plt.tight_layout()
            plt.savefig(
                plot_dir / f"{name}_test_{i:02d}.png",
                dpi=150
            )
            plt.close()

        print(
            f"{lang} test {i}/5 | "
            f"{len(phones)} phones | "
            f"{len(audio)/22050:.2f}s | "
            f"saved",
            flush=True
        )

print("\nGenerated 10 test WAV files and variance-adaptor plots.")

# ===== CELL 35 =====

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
RESULTS = ROOT / "Results"

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
colors = {"English": "#3478C0", "Telugu": "#E18B39"}

for lang in ["English", "Telugu"]:
    history = json.loads(
        (RESULTS / lang / "history.json").read_text()
    )

    epochs = np.arange(1, len(history["train"]) + 1)

    ax[0, 0].plot(
        epochs, history["train"],
        marker="o", linewidth=2,
        color=colors[lang], label=lang
    )

    ax[0, 1].plot(
        epochs, history["val"],
        marker="o", linewidth=2,
        color=colors[lang], label=lang
    )

ax[0, 0].set_title("Training Loss")
ax[0, 1].set_title("Validation Loss")

for a in ax[0]:
    a.set_xlabel("Epoch")
    a.set_ylabel("Total Loss")
    a.legend()
    a.grid(alpha=0.2)

# Use one actual preprocessed English sample
feature_dir = ROOT / "acoustic_features" / "English" / "train"
sample_path = sorted(feature_dir.glob("*.npz"))[0]

with np.load(sample_path) as data:
    mel = data["mel"]
    duration = data["duration"]

if mel.shape[0] == 80 and mel.shape[1] != 80:
    mel = mel.T

ax[1, 0].imshow(
    mel.T,
    origin="lower",
    aspect="auto",
    cmap="magma"
)
ax[1, 0].set_title("English Sample: Log-Mel Spectrogram")
ax[1, 0].set_xlabel("Time Frame")
ax[1, 0].set_ylabel("Mel Bin")

ax[1, 1].bar(
    np.arange(len(duration)),
    duration,
    color="#4E9B88",
    width=0.85
)
ax[1, 1].set_title("English Sample: Phoneme Durations")
ax[1, 1].set_xlabel("Phoneme Index")
ax[1, 1].set_ylabel("Duration (Frames)")

fig.suptitle(
    "FastSpeech 2 — Training and Acoustic Feature Analysis",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

output = RESULTS / "FastSpeech2_Analysis_Dashboard.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.show()

print("Saved:", output)

# ===== CELL 36 =====

from pathlib import Path
from IPython.display import Audio, display

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

for lang in ["English", "Telugu"]:
    print(f"\n{lang} — Test Sentence 2")

    path = (
        ROOT / "Results" / lang
        / "test_generated_audios" / "test_02.wav"
    )

    display(Audio(filename=str(path)))

# ===== CELL 37 =====

from pathlib import Path
from IPython.display import Audio, display
import numpy as np
import librosa
import soundfile as sf

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

for lang in ["English", "Telugu"]:
    folder = ROOT / "acoustic_features" / lang / "train"
    path = sorted(folder.glob("*.npz"))[0]

    with np.load(path) as data:
        mel = data["mel"].astype(np.float32)

    if mel.shape[0] == 80 and mel.shape[1] != 80:
        mel = mel.T

    power_mel = np.exp(np.clip(mel.T, -20, 12))

    reconstructed = librosa.feature.inverse.mel_to_audio(
        power_mel,
        sr=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        fmin=0,
        fmax=8000,
        power=2.0,
        n_iter=64
    )

    reconstructed = np.nan_to_num(reconstructed)
    peak = np.max(np.abs(reconstructed))

    if peak > 0:
        reconstructed = 0.95 * reconstructed / peak

    output = ROOT / "Results" / lang / "ground_truth_reconstruction.wav"
    sf.write(output, reconstructed, 22050)

    print(f"\n{lang} — Ground-truth mel reconstruction")
    display(Audio(reconstructed, rate=22050))

# ===== CELL 38 =====

import json
import numpy as np
import librosa
from pathlib import Path
from IPython.display import Audio, display

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
folder = ROOT / "acoustic_features" / "Telugu" / "train"

candidates = []

for path in folder.glob("*.json"):
    meta = json.loads(path.read_text(encoding="utf-8"))
    if len(meta.get("text", "").split()) >= 7:
        candidates.append((path, meta))

if not candidates:
    raise RuntimeError("No Telugu recordings with 7+ words found")

path, meta = candidates[0]

with np.load(path.with_suffix(".npz")) as z:
    mel = z["mel"].astype(np.float32)

if mel.shape[0] == 80 and mel.shape[1] != 80:
    mel = mel.T

audio = librosa.feature.inverse.mel_to_audio(
    np.exp(np.clip(mel.T, -20, 12)),
    sr=22050,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    fmin=0,
    fmax=8000,
    power=2.0,
    n_iter=64
)

print("Text:", meta["text"])
print("Audio duration:", round(len(audio) / 22050, 2), "seconds")

print("\nOriginal Telugu:")
display(Audio(filename=str(meta["audio"])))

print("\nReconstructed Telugu:")
display(Audio(audio, rate=22050))

# ===== CELL 39 =====

import json, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

def resume_training(lang, extra_epochs=50):
    train_ds = TTSDataset(lang, "train")
    val_ds = TTSDataset(lang, "val")

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        collate_fn=collate, num_workers=0
    )

    ckpt_dir = ROOT / lang / "checkpoints"
    ckpt = torch.load(
        ckpt_dir / "last.pt",
        map_location=DEVICE,
        weights_only=False
    )

    model = FastSpeech2(
        len(train_ds.vocab),
        hidden=ckpt["hidden"]
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(ckpt["optimizer"])

    for group in optimizer.param_groups:
        group["lr"] = 1e-4

    results = ROOT / "Results" / lang
    history_path = results / "history.json"

    history = json.loads(history_path.read_text())
    best = min(history["val"])

    start_epoch = ckpt["epoch"]

    for epoch in range(start_epoch + 1, start_epoch + extra_epochs + 1):
        start = time.time()
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = tuple(x.to(DEVICE) for x in batch)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                batch[0], batch[1], batch[3],
                batch[4], batch[5], batch[6]
            )

            loss, _ = compute_loss(outputs, batch)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(x.to(DEVICE) for x in batch)

                outputs = model(
                    batch[0], batch[1], batch[3],
                    batch[4], batch[5], batch[6]
                )

                loss, _ = compute_loss(outputs, batch)

                if torch.isfinite(loss):
                    val_losses.append(loss.item())

        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses))

        history["train"].append(tr)
        history["val"].append(va)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "vocab": train_ds.vocab,
            "train_loss": tr,
            "val_loss": va,
            "architecture": "compact_fastspeech2",
            "hidden": 128
        }

        torch.save(state, ckpt_dir / "last.pt")

        if va < best:
            best = va
            torch.save(state, ckpt_dir / "best.pt")

        history_path.write_text(json.dumps(history, indent=2))

        if epoch % 10 == 0 or epoch == start_epoch + 1:
            print(
                f"{lang} epoch {epoch} | "
                f"train={tr:.4f} | val={va:.4f} | "
                f"{time.time()-start:.1f}s",
                flush=True
            )

    for key, filename in [
        ("train", "training_loss.png"),
        ("val", "validation_loss.png")
    ]:
        plt.figure(figsize=(7, 4))
        plt.plot(history[key], linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{lang} {key.capitalize()} Loss")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(results / filename, dpi=160)
        plt.close()

    print(f"{lang}: finished at epoch {start_epoch + extra_epochs}")

resume_training("English", extra_epochs=50)
resume_training("Telugu", extra_epochs=80)

# ===== CELL 40 =====

from IPython.display import Audio, display
import torch
import soundfile as sf

for lang in ["English", "Telugu"]:
    ckpt = torch.load(
        ROOT / lang / "checkpoints" / "best.pt",
        map_location=DEVICE,
        weights_only=False
    )

    model = FastSpeech2(
        len(ckpt["vocab"]),
        hidden=ckpt["hidden"]
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()

    out = ROOT / "Results" / lang / "test_generated_audios"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{lang} — best checkpoint: epoch {ckpt['epoch']}")

    for i, sentence in enumerate(TESTS[lang], 1):
        mel, duration, pitch, energy, phones = synthesize(
            model, sentence, ckpt["vocab"], lang
        )

        audio = mel_to_audio(mel)
        sf.write(out / f"test_{i:02d}.wav", audio, 22050)

        print(f"Test {i}: {len(audio)/22050:.2f}s")

        if i == 2:
            display(Audio(audio, rate=22050))

print("\nAll 10 test recordings updated.")

# ===== CELL 41 =====

import json
import numpy as np
import torch
from pathlib import Path
from IPython.display import Audio, display

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

for lang in ["English", "Telugu"]:
    ds = TTSDataset(lang, "train")

    ckpt = torch.load(
        ROOT / lang / "checkpoints" / "best.pt",
        map_location=DEVICE,
        weights_only=False
    )

    model = FastSpeech2(
        len(ckpt["vocab"]),
        hidden=ckpt["hidden"]
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()

    idx = min(10, len(ds) - 1)
    batch = collate([ds[idx]])
    batch = tuple(x.to(DEVICE) for x in batch)

    with torch.no_grad():
        outputs = model(
            batch[0],
            batch[1],
            batch[3],
            batch[4],
            batch[5],
            batch[6]
        )

    predicted_mel = outputs[0][0, :int(batch[6][0])].cpu().numpy()
    reference_mel = batch[2][0, :int(batch[6][0])].cpu().numpy()

    predicted_audio = mel_to_audio(predicted_mel)
    reference_audio = mel_to_audio(reference_mel)

    print(f"\n{lang} — using ground-truth durations, pitch and energy")

    print("Reference mel reconstruction:")
    display(Audio(reference_audio, rate=22050))

    print("Model-predicted mel reconstruction:")
    display(Audio(predicted_audio, rate=22050))

    print(
        "Mel MAE:",
        round(float(np.mean(np.abs(predicted_mel - reference_mel))), 4)
    )

# ===== CELL 42 =====

import json, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

class CalibratedFS2(nn.Module):
    def __init__(self, base, mean, std):
        super().__init__()
        self.base = base
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.calibration = nn.Sequential(
            nn.Linear(80, 80),
            nn.Tanh(),
            nn.Linear(80, 80)
        )
        nn.init.zeros_(self.calibration[-1].weight)
        nn.init.zeros_(self.calibration[-1].bias)

    def forward(self, *args):
        mel, dur, pitch, energy = self.base(*args)
        normalized = (mel - self.mean) / self.std
        normalized = normalized + self.calibration(normalized)
        return normalized, dur, pitch, energy

def get_mel_stats(ds):
    total = np.zeros(80, dtype=np.float64)
    total_sq = np.zeros(80, dtype=np.float64)
    count = 0

    for i in range(len(ds)):
        mel = ds[i][2].numpy().astype(np.float64)
        total += mel.sum(axis=0)
        total_sq += (mel ** 2).sum(axis=0)
        count += len(mel)

    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-4))

    return (
        torch.tensor(mean, dtype=torch.float32, device=DEVICE),
        torch.tensor(std, dtype=torch.float32, device=DEVICE)
    )

def finetune(lang, epochs=25):
    train_ds = TTSDataset(lang, "train")
    val_ds = TTSDataset(lang, "val")

    ckpt_dir = ROOT / lang / "checkpoints"
    ckpt = torch.load(
        ckpt_dir / "best.pt",
        map_location=DEVICE,
        weights_only=False
    )

    base = FastSpeech2(
        len(ckpt["vocab"]),
        hidden=ckpt["hidden"]
    ).to(DEVICE)

    base.load_state_dict(ckpt["model"])

    mean, std = get_mel_stats(train_ds)
    model = CalibratedFS2(base, mean, std).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, collate_fn=collate
    )

    best = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            batch = tuple(x.to(DEVICE) for x in batch)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                batch[0], batch[1], batch[3],
                batch[4], batch[5], batch[6]
            )

            target = (batch[2] - mean) / std
            mask = (
                torch.arange(target.shape[1], device=DEVICE)[None, :]
                < batch[6][:, None]
            )

            mel_loss = (
                (outputs[0] - target).abs()
                * mask.unsqueeze(-1)
            ).sum() / (mask.sum() * 80).clamp(min=1)

            dur_mask = (
                torch.arange(batch[0].shape[1], device=DEVICE)[None, :]
                < batch[5][:, None]
            )

            dur_loss = (
                ((outputs[1] - torch.log1p(batch[1].float())) ** 2)
                * dur_mask
            ).sum() / dur_mask.sum().clamp(min=1)

            loss = mel_loss + 0.1 * dur_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        vals = []

        with torch.no_grad():
            for batch in val_loader:
                batch = tuple(x.to(DEVICE) for x in batch)
                outputs = model(
                    batch[0], batch[1], batch[3],
                    batch[4], batch[5], batch[6]
                )

                target = (batch[2] - mean) / std
                mask = (
                    torch.arange(target.shape[1], device=DEVICE)[None, :]
                    < batch[6][:, None]
                )

                err = (
                    (outputs[0] - target).abs()
                    * mask.unsqueeze(-1)
                ).sum() / (mask.sum() * 80).clamp(min=1)

                vals.append(err.item())

        va = float(np.mean(vals))

        if va < best:
            best = va
            torch.save({
                "model": model.state_dict(),
                "mean": mean.cpu(),
                "std": std.cpu(),
                "vocab": ckpt["vocab"],
                "hidden": ckpt["hidden"],
                "epoch": epoch,
                "val_normalized_mae": va
            }, ckpt_dir / "normalized_best.pt")

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"{lang} calibration epoch {epoch}/{epochs} | "
                f"train={np.mean(losses):.4f} | val={va:.4f}",
                flush=True
            )

    print(lang, "best normalized val MAE:", round(best, 4))

finetune("English")
finetune("Telugu")

# ===== CELL 43 =====

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from IPython.display import Audio, display

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

for lang in ["English", "Telugu"]:
    ckpt = torch.load(
        ROOT / lang / "checkpoints" / "normalized_best.pt",
        map_location=DEVICE,
        weights_only=False
    )

    base = FastSpeech2(
        len(ckpt["vocab"]),
        hidden=ckpt["hidden"]
    ).to(DEVICE)

    model = CalibratedFS2(
        base,
        ckpt["mean"].to(DEVICE),
        ckpt["std"].to(DEVICE)
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()

    output = ROOT / "Results" / lang / "calibrated_audio"
    output.mkdir(parents=True, exist_ok=True)

    print(f"\n{lang} — calibrated checkpoint")

    for i, sentence in enumerate(TESTS[lang], 1):
        # synthesize() expects the original FastSpeech2 architecture.
        # First generate using the calibrated model's base.
        mel, duration, pitch, energy, phones = synthesize(
            model.base, sentence, ckpt["vocab"], lang
        )

        with torch.no_grad():
            mel_tensor = torch.tensor(
                mel, dtype=torch.float32, device=DEVICE
            )

            normalized = (
                mel_tensor - model.mean
            ) / model.std

            calibrated = normalized + model.calibration(normalized)

            mel = (
                calibrated * model.std + model.mean
            ).cpu().numpy()

        audio = mel_to_audio(mel)

        path = output / f"test_{i:02d}.wav"
        sf.write(path, audio, 22050)

        print(
            f"Test {i}: {len(audio)/22050:.2f}s",
            flush=True
        )

        if i == 2:
            display(Audio(audio, rate=22050))

print("\nCalibrated audio generation complete.")

# ===== CELL 44 =====

y = (
    y
    + self.pitch_embed(pitch.unsqueeze(-1))
    + self.energy_embed(energy.unsqueeze(-1))
)

# ===== CELL 45 =====

import sys
import subprocess
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from IPython.display import Audio, display

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.44.2",
    "sentencepiece",
    "soundfile"
])

from transformers import VitsModel, AutoTokenizer

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "English": "facebook/mms-tts-eng",
    "Telugu": "facebook/mms-tts-tel"
}

SENTENCES = {
    "English": [
        "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "The orchestra's conduct during the ceremony was praised by the audience.",
        "The researchers will conduct an experiment to evaluate the proposed method.",
        "The rhythm of the orchestra changed unexpectedly.",
        "The meeting will start at ten forty five in the morning on November twelfth."
    ],
    "Telugu": [
        "నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
        "ఈ సమావేశం నవంబర్ పన్నెండవ తేదీ ఉదయం ప్రారంభమవుతుంది."
    ]
}

for lang, model_id in MODELS.items():
    print(f"\nLoading {lang}: {model_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(DEVICE)
    model.eval()

    out_dir = ROOT / "Results" / lang / "pretrained_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    sr = model.config.sampling_rate

    for i, sentence in enumerate(SENTENCES[lang], 1):
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            waveform = model(**inputs).waveform[0].cpu().numpy()

        waveform = np.nan_to_num(waveform).astype(np.float32)

        path = out_dir / f"test_{i:02d}.wav"
        sf.write(path, waveform, sr)

        print(
            f"{lang} {i}/5 | "
            f"{len(waveform)/sr:.2f}s | saved",
            flush=True
        )

        if i == 2:
            display(Audio(waveform, rate=sr))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nPretrained baseline generation complete.")

# ===== CELL 46 =====

import sys
import subprocess

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "-q", "-U",
    "transformers>=4.51,<5",
    "tokenizers>=0.21",
    "sentencepiece"
])

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from IPython.display import Audio, display
from transformers import VitsModel, AutoTokenizer

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "English": "facebook/mms-tts-eng",
    "Telugu": "facebook/mms-tts-tel"
}

SENTENCES = {
    "English": [
        "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "The orchestra's conduct during the ceremony was praised by the audience.",
        "The researchers will conduct an experiment to evaluate the proposed method.",
        "The rhythm of the orchestra changed unexpectedly.",
        "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."
    ],    "Telugu": [
        "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
        "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."
    ]
}

for lang, model_id in MODELS.items():
    print(f"\nLoading {lang}: {model_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(DEVICE)
    model.eval()

    output = ROOT / "Results" / lang / "pretrained_baseline"
    output.mkdir(parents=True, exist_ok=True)

    sr = model.config.sampling_rate

    for i, sentence in enumerate(SENTENCES[lang], 1):
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            audio = model(**inputs).waveform[0].cpu().numpy()

        audio = np.nan_to_num(audio).astype(np.float32)

        sf.write(output / f"test_{i:02d}.wav", audio, sr)

        print(f"{lang} {i}/5 saved", flush=True)

        if i == 2:
            display(Audio(audio, rate=sr))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nAll 10 baseline recordings generated.")

# ===== CELL 47 =====

import sys
import subprocess

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "-q", "-U",
    "transformers>=4.51,<5",
    "tokenizers>=0.21",
    "sentencepiece"
])

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from IPython.display import Audio, display
from transformers import VitsModel, AutoTokenizer

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "English": "facebook/mms-tts-eng",
    "Telugu": "facebook/mms-tts-tel"
}

SENTENCES = {
    "English": [
        "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "The orchestra's conduct during the ceremony was praised by the audience.",
        "The researchers will conduct an experiment to evaluate the proposed method.",
        "The rhythm of the orchestra changed unexpectedly.",
        "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."
    ],
    "Telugu": [
        "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
        "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."
    ]
}

for lang, model_id in MODELS.items():
    print(f"\nLoading {lang}: {model_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(DEVICE)
    model.eval()

    output = ROOT / "Results" / lang / "pretrained_baseline"
    output.mkdir(parents=True, exist_ok=True)

    sr = model.config.sampling_rate

    for i, sentence in enumerate(SENTENCES[lang], 1):
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            audio = model(**inputs).waveform[0].cpu().numpy()

        audio = np.nan_to_num(audio).astype(np.float32)

        sf.write(output / f"test_{i:02d}.wav", audio, sr)

        print(f"{lang} {i}/5 saved", flush=True)

        if i == 2:
            display(Audio(audio, rate=sr))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nAll 10 baseline recordings generated.")

# ===== CELL 48 =====

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from IPython.display import Audio, display
from transformers import VitsModel, VitsTokenizer

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "English": "facebook/mms-tts-eng",
    "Telugu": "facebook/mms-tts-tel"
}

SENTENCES = {
    "English": [
        "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "The orchestra's conduct during the ceremony was praised by the audience.",
        "The researchers will conduct an experiment to evaluate the proposed method.",
        "The rhythm of the orchestra changed unexpectedly.",
        "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."
    ],
    "Telugu": [
        "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
        "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
        "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."
    ]
}

for lang, model_id in MODELS.items():
    print(f"\nLoading {lang}: {model_id}", flush=True)

    tokenizer = VitsTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(DEVICE)
    model.eval()

    output = ROOT / "Results" / lang / "pretrained_baseline"
    output.mkdir(parents=True, exist_ok=True)

    sr = model.config.sampling_rate

    for i, sentence in enumerate(SENTENCES[lang], 1):
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            waveform = model(**inputs).waveform[0].cpu().numpy()

        waveform = np.nan_to_num(waveform).astype(np.float32)

        path = output / f"test_{i:02d}.wav"
        sf.write(path, waveform, sr)

        print(
            f"{lang} test {i}/5 | "
            f"{len(waveform)/sr:.2f}s | saved",
            flush=True
        )

        if i == 2:
            display(Audio(waveform, rate=sr))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nAll 10 baseline recordings generated.")

# ===== CELL 49 =====

---------------------------------------------------------------------------
HTTPStatusError                           Traceback (most recent call last)
/usr/local/lib/python3.13/dist-packages/huggingface_hub/utils/_http.py in hf_raise_for_status(response, endpoint_name)

/usr/local/lib/python3.13/dist-packages/httpx/_models.py in raise_for_status(self)
    828         message = message.format(self, error_type=error_type)
--> 829         raise HTTPStatusError(message, request=request, response=self)
    830 

HTTPStatusError: Client error '404 Not Found' for url 'https://huggingface.co/api/models/facebook/mms-tts-eng/tree/main/additional_chat_templates?recursive=false&expand=false'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404

The above exception was the direct cause of the following exception:

RemoteEntryNotFoundError                  Traceback (most recent call last)
/tmp/ipykernel_50/3305132755.py in <cell line: 0>()
     34     print(f"\nLoading {lang}: {model_id}", flush=True)
     35 
---> 36     tokenizer = VitsTokenizer.from_pretrained(model_id)
     37     model = VitsModel.from_pretrained(model_id).to(DEVICE)
     38     model.eval()

/usr/local/lib/python3.13/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   2037                                 )
   2038                     else:
-> 2039                         for template in list_repo_templates(
   2040                             pretrained_model_name_or_path,
   2041                             local_files_only=local_files_only,

/usr/local/lib/python3.13/dist-packages/transformers/utils/hub.py in list_repo_templates(repo_id, local_files_only, revision, cache_dir, token)
    167             return [
    168                 entry.path.removeprefix(f"{CHAT_TEMPLATE_DIR}/")
--> 169                 for entry in list_repo_tree(
    170                     repo_id=repo_id,
    171                     revision=revision,

/usr/local/lib/python3.13/dist-packages/huggingface_hub/hf_api.py in list_repo_tree(self, repo_id, path_in_repo, recursive, expand, revision, repo_type, token)
   4101                     - [`~hf_api.CommitOperationDelete`] to delete a file
   4102                     - [`~hf_api.CommitOperationCopy`] to copy a file
-> 4103 
   4104                 Operation objects will be mutated to include information relative to the upload. Do not reuse the
   4105                 same objects for multiple commits.

/usr/local/lib/python3.13/dist-packages/huggingface_hub/utils/_pagination.py in paginate(path, params, headers, timeout)
     35     session = get_session()
     36     r = session.get(path, params=params, headers=headers)
---> 37     hf_raise_for_status(r)
     38     yield from r.json()
     39 

/usr/local/lib/python3.13/dist-packages/huggingface_hub/utils/_http.py in hf_raise_for_status(response, endpoint_name)

RemoteEntryNotFoundError: 404 Client Error. (Request ID: Root=1-6aee0102-6beae86e5c629cbd39b9155a;271bff03-28e6-4284-8a8c-4a48a25aed97)

Entry Not Found for url: https://huggingface.co/api/models/facebook/mms-tts-eng/tree/main/additional_chat_templates?recursive=false&expand=false.
additional_chat_templates does not exist on "main"

# ===== CELL 50 =====

from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import VitsTokenizer, VitsModel
from IPython.display import Audio, display
import torch
import numpy as np
import soundfile as sf

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "English": "facebook/mms-tts-eng",
    "Telugu": "facebook/mms-tts-tel"
}

# Reuse SENTENCES from the previous cell.
assert "SENTENCES" in globals(), "Run the previous definitions cell first."

for lang, repo in MODELS.items():
    print(f"\nLoading {lang}...", flush=True)

    local = ROOT / "pretrained_models" / lang
    local.mkdir(parents=True, exist_ok=True)

    # Download only the files needed for tokenizer initialization.
    for filename in ["vocab.json", "tokenizer_config.json", "config.json"]:
        try:
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(local)
            )
        except Exception as e:
            if filename == "vocab.json":
                raise
            print(f"Optional {filename}: {type(e).__name__}")

    # Local directory avoids the additional_chat_templates API request.
    tokenizer = VitsTokenizer.from_pretrained(
        str(local),
        local_files_only=True
    )

    model = VitsModel.from_pretrained(repo).to(DEVICE)
    model.eval()

    output = ROOT / "Results" / lang / "pretrained_baseline"
    output.mkdir(parents=True, exist_ok=True)

    sr = model.config.sampling_rate

    for i, sentence in enumerate(SENTENCES[lang], 1):
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            audio = model(**inputs).waveform[0].cpu().numpy()

        audio = np.nan_to_num(audio).astype(np.float32)
        sf.write(output / f"test_{i:02d}.wav", audio, sr)

        print(f"{lang}: {i}/5 saved", flush=True)

        if i == 2:
            display(Audio(audio, rate=sr))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nBaseline generation finished.")

# ===== CELL 51 =====

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import VitsTokenizer, VitsModel
from IPython.display import Audio, display

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "English": "facebook/mms-tts-eng",
    "Telugu": "facebook/mms-tts-tel"
}

for lang, repo in MODELS.items():
    print(f"\nLoading {lang}...", flush=True)

    local = ROOT / "pretrained_models" / lang
    local.mkdir(parents=True, exist_ok=True)

    for filename in [
        "config.json",
        "vocab.json",
        "tokenizer_config.json",
        "pytorch_model.bin"
    ]:
        try:
            path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(local)
            )
            print("Downloaded:", filename, flush=True)
        except Exception as e:
            if filename == "tokenizer_config.json":
                continue
            raise RuntimeError(
                f"Failed downloading {filename}: {e}"
            ) from e

    tokenizer = VitsTokenizer.from_pretrained(
        str(local),
        local_files_only=True
    )

    model = VitsModel.from_pretrained(
        str(local),
        local_files_only=True,
        use_safetensors=False
    ).to(DEVICE)

    model.eval()

    output = ROOT / "Results" / lang / "pretrained_baseline"
    output.mkdir(parents=True, exist_ok=True)

    sr = model.config.sampling_rate

    for i, sentence in enumerate(SENTENCES[lang], 1):
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            audio = model(**inputs).waveform[0].cpu().numpy()

        audio = np.nan_to_num(audio).astype(np.float32)

        sf.write(output / f"test_{i:02d}.wav", audio, sr)

        print(f"{lang} test {i}/5 saved", flush=True)

        if i == 2:
            display(Audio(audio, rate=sr))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nBaseline generation finished.")

# ===== CELL 52 =====

import os
import requests
from pathlib import Path

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

MODELS = {
    "English": "facebook/mms-tts-eng",
    "Telugu": "facebook/mms-tts-tel"
}

for lang, repo in MODELS.items():
    folder = ROOT / "pretrained_models" / lang
    folder.mkdir(parents=True, exist_ok=True)

    target = folder / "pytorch_model.bin"

    if target.exists() and target.stat().st_size > 10_000_000:
        print(f"{lang}: weights already downloaded")
        continue

    url = f"https://huggingface.co/{repo}/resolve/main/pytorch_model.bin?download=true"

    print(f"\nDownloading {lang} weights...", flush=True)

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, 60),
            allow_redirects=True
        ) as response:
            response.raise_for_status()

            with open(str(target) + ".part", "wb") as f:
                downloaded = 0

                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if downloaded % (50 * 1024 * 1024) < len(chunk):
                            print(
                                f"{lang}: {downloaded / 1e6:.0f} MB",
                                flush=True
                            )

        os.replace(str(target) + ".part", target)

        print(
            f"{lang}: downloaded {target.stat().st_size / 1e6:.1f} MB",
            flush=True
        )

    except Exception as e:
        print(f"{lang} download failed: {e}", flush=True)
        raise

print("\nWeight downloads complete.")

# ===== CELL 53 =====

from transformers import VitsTokenizer, VitsModel

# ===== CELL 54 =====

from transformers import VitsTokenizer, VitsModel

# ===== CELL 55 =====

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import VitsTokenizer, VitsModel
from IPython.display import Audio, display

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENTENCES = {
    "English": [
        "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "The orchestra's conduct during the ceremony was praised by the audience.",
        "The researchers will conduct an experiment to evaluate the proposed method.",
        "The rhythm of the orchestra changed unexpectedly.",
        "The meeting will start at ten forty five in the morning on November twelfth."
    ],
    "Telugu": [
        "నూతన ఔషధాల ప్రభావాన్ని పరిశోధకులు విశ్లేషించారు.",
        "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
        "ఈ సమావేశం నవంబర్ పన్నెండవ తేదీన ఉదయం ప్రారంభమవుతుంది."
    ]
}

for lang in ["English", "Telugu"]:
    local = ROOT / "pretrained_models" / lang

    print(f"\nGenerating {lang} speech...", flush=True)

    tokenizer = VitsTokenizer.from_pretrained(
        str(local),
        local_files_only=True
    )

    model = VitsModel.from_pretrained(
        str(local),
        local_files_only=True,
        use_safetensors=False
    ).to(DEVICE)

    model.eval()

    output = ROOT / "Results" / lang / "pretrained_baseline"
    output.mkdir(parents=True, exist_ok=True)

    sr = model.config.sampling_rate

    for i, sentence in enumerate(SENTENCES[lang], 1):
        inputs = tokenizer(
            sentence,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            audio = model(**inputs).waveform[0].cpu().numpy()

        audio = np.nan_to_num(audio).astype(np.float32)

        path = output / f"test_{i:02d}.wav"
        sf.write(path, audio, sr)

        print(f"{lang} {i}/5 saved")

        if i == 2:
            display(Audio(audio, rate=sr))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nFinished generating all 10 WAV files.")

# ===== CELL 56 =====

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import VitsTokenizer, VitsModel
from IPython.display import Audio, display

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

local = ROOT / "pretrained_models" / "Telugu"
repo = "facebook/mms-tts-tel"

local.mkdir(parents=True, exist_ok=True)

# Download missing tokenizer files only
for filename in ["vocab.json", "tokenizer_config.json", "config.json"]:
    if not (local / filename).exists():
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(local)
        )

print("Telugu files:", [p.name for p in local.iterdir()])

tokenizer = VitsTokenizer.from_pretrained(
    str(local),
    local_files_only=True
)

model = VitsModel.from_pretrained(
    str(local),
    local_files_only=True,
    use_safetensors=False
).to(DEVICE)

model.eval()

sentences = [
    "నూతన ఔషధాల ప్రభావాన్ని పరిశోధకులు విశ్లేషించారు.",
    "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
    "ఈ సమావేశం నవంబర్ పన్నెండవ తేదీన ఉదయం ప్రారంభమవుతుంది."
]

output = ROOT / "Results" / "Telugu" / "pretrained_baseline"
output.mkdir(parents=True, exist_ok=True)

sr = model.config.sampling_rate

for i, sentence in enumerate(sentences, 1):
    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        audio = model(**inputs).waveform[0].cpu().numpy()

    audio = np.nan_to_num(audio).astype(np.float32)

    sf.write(output / f"test_{i:02d}.wav", audio, sr)

    print(f"Telugu {i}/5 saved")

    if i == 2:
        display(Audio(audio, rate=sr))

print("All 5 Telugu baseline recordings saved.")

# ===== CELL 57 =====

from pathlib import Path
import zipfile

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
OUT = Path("/kaggle/working/FastSpeech2_Submission_Backup.zip")

exclude = {
    "pytorch_model.bin",
    "ctc_model.onnx",
}

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.name in exclude:
            continue
        if ".git" in p.parts:
            continue
        z.write(p, p.relative_to(ROOT.parent))

print("ZIP:", OUT)
print("Size:", round(OUT.stat().st_size / 1024**2, 1), "MB")
print("English baseline WAVs:",
      len(list((ROOT / "Results/English/pretrained_baseline").glob("*.wav"))))
print("Telugu baseline WAVs:",
      len(list((ROOT / "Results/Telugu/pretrained_baseline").glob("*.wav"))))

# ===== CELL 58 =====

from pathlib import Path
import zipfile

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
OUT = Path("/kaggle/working/Final_Submission.zip")

exclude_dirs = {
    ".git",
    "wavs",
    "acoustic_features",
    "alignment_test",
    "pretrained_models",
    "__pycache__",
}

exclude_files = {
    "pytorch_model.bin",
    "ctc_model.onnx",
}

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue

        rel = p.relative_to(ROOT)

        if any(part in exclude_dirs for part in rel.parts):
            continue

        if p.name in exclude_files:
            continue

        z.write(p, rel)

print("Created:", OUT)
print("Size:", round(OUT.stat().st_size / 1024**2, 1), "MB")

# ===== CELL 59 =====

from pathlib import Path
import zipfile

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
OUT = Path("/kaggle/working/Final_Submission.zip")

exclude_dirs = {
    ".git",
    "wavs",
    "acoustic_features",
    "alignment_test",
    "pretrained_models",
    "__pycache__",
}

exclude_files = {
    "pytorch_model.bin",
    "ctc_model.onnx",
}

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue

        rel = p.relative_to(ROOT)

        if any(part in exclude_dirs for part in rel.parts):
            continue

        if p.name in exclude_files:
            continue

        z.write(p, rel)

print("Created:", OUT)
print("Size:", round(OUT.stat().st_size / 1024**2, 1), "MB")

# ===== CELL 60 =====

from pathlib import Path

ROOT = Path("/kaggle/working/FastSpeech2_Submission")

for lang in ["English", "Telugu"]:
    print(f"\n{'='*15} {lang} {'='*15}")

    paths = {
        "Training features": ROOT / "acoustic_features" / lang / "train",
        "Validation features": ROOT / "acoustic_features" / lang / "val",
        "Vocabulary": ROOT / "acoustic_features" / lang / "vocab.json",
        "Best checkpoint": ROOT / lang / "checkpoints" / "best.pt",
        "Last checkpoint": ROOT / lang / "checkpoints" / "last.pt",
        "Normalized checkpoint": ROOT / lang / "checkpoints" / "normalized_best.pt",
        "Training history": ROOT / "Results" / lang / "history.json",
        "FastSpeech2 audio": ROOT / "Results" / lang / "test_generated_audios",
        "Working VITS audio": ROOT / "Results" / lang / "pretrained_baseline",
    }

    for name, path in paths.items():
        if path.is_dir():
            count = len(list(path.glob("*.npz"))) if "features" in name.lower() else len(list(path.glob("*.wav")))
            print(f"{name}: EXISTS ({count} files)")
        else:
            print(f"{name}: {'EXISTS' if path.exists() else 'MISSING'}")

print("\nSaved notebooks:")
for p in ROOT.rglob("*.ipynb"):
    print(p)

# ===== CELL 61 =====

import json
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
FEATURES = ROOT / "acoustic_features"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TTSDataset(Dataset):
    def __init__(self, lang, split):
        self.folder = FEATURES / lang / split

        with open(FEATURES / lang / "vocab.json", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.files = sorted(self.folder.glob("*.npz"))

        self.files = [
            p for p in self.files
            if p.with_suffix(".json").exists()
        ]

        print(f"{lang}/{split}: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        with np.load(path) as data:
            mel = data["mel"].copy()
            pitch = data["pitch"].copy()
            energy = data["energy"].copy()
            duration = data["duration"].copy()

        meta = json.loads(
            path.with_suffix(".json").read_text(encoding="utf-8")
        )

        phones = [
            self.vocab.get(p, 1)
            for p in meta["phones"]
        ]

        return {
            "phones": torch.tensor(phones, dtype=torch.long),
            "duration": torch.tensor(duration, dtype=torch.long),
            "mel": torch.tensor(mel, dtype=torch.float32),
            "pitch": torch.tensor(pitch, dtype=torch.float32),
            "energy": torch.tensor(energy, dtype=torch.float32),
            "text": meta["text"]
        }


def collate(batch):
    from torch.nn.utils.rnn import pad_sequence

    result = {}

    for key in ["phones", "duration", "mel", "pitch", "energy"]:
        result[key] = pad_sequence(
            [item[key] for item in batch],
            batch_first=True,
            padding_value=0
        )

    result["phone_lengths"] = torch.tensor(
        [len(item["phones"]) for item in batch]
    )

    result["mel_lengths"] = torch.tensor(
        [len(item["mel"]) for item in batch]
    )

    result["text"] = [item["text"] for item in batch]

    return result


for lang in ["English", "Telugu"]:
    ds = TTSDataset(lang, "train")

    if len(ds):
        sample = ds[0]
        print(
            lang,
            "Phones:", sample["phones"].shape,
            "Mel:", sample["mel"].shape,
            "Duration sum:", sample["duration"].sum().item()
        )

# ===== CELL 62 =====

import torch
import numpy as np
import soundfile as sf

from pathlib import Path
from transformers import VitsTokenizer, VitsModel
from IPython.display import Audio, display

ROOT = Path("/kaggle/working/FastSpeech2_Submission")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENTENCES = {
    "English": [
        "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        "The orchestra's conduct during the ceremony was praised by the audience.",
        "The researchers will conduct an experiment to evaluate the proposed method.",
        "The rhythm of the orchestra changed unexpectedly.",
        "The meeting will start at ten forty five in the morning on November twelfth."
    ],
    "Telugu": [
        "నూతన ఔషధాల ప్రభావాన్ని పరిశోధకులు విశ్లేషించారు.",
        "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
        "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
        "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
        "ఈ సమావేశం నవంబర్ పన్నెండవ తేదీన ఉదయం ప్రారంభమవుతుంది."
    ]
}

for lang in ["English", "Telugu"]:
    local = ROOT / "pretrained_models" / lang
    output = ROOT / "Results" / lang / "pretrained_baseline"
    output.mkdir(parents=True, exist_ok=True)

    existing = list(output.glob("test_*.wav"))

    if len(existing) == 5:
        print(f"{lang}: All 5 WAV files already exist")
        display(Audio(filename=str(output / "test_02.wav")))
        continue

    tokenizer = VitsTokenizer.from_pretrained(
        str(local),
        local_files_only=True
    )

    model = VitsModel.from_pretrained(
        str(local),
        local_files_only=True,
        use_safetensors=False
    ).to(DEVICE)

    model.eval()
    sr = model.config.sampling_rate

    for i, sentence in enumerate(SENTENCES[lang], 1):
        path = output / f"test_{i:02d}.wav"

        if path.exists():
            continue

        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            audio = model(**inputs).waveform[0].cpu().numpy()

        audio = np.nan_to_num(audio).astype(np.float32)
        sf.write(path, audio, sr)

        print(f"{lang}: {i}/5 saved")

    display(Audio(filename=str(output / "test_02.wav")))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ===== CELL 63 =====

from pathlib import Path
from IPython import get_ipython

OUT = Path("/kaggle/working/Recovered_Notebook_Cells.py")

ip = get_ipython()

if ip is None:
    print("IPython session unavailable")
else:
    history = ip.history_manager.get_range(
        session=0,
        start=1,
        stop=None,
        raw=True,
        output=False
    )

    cells = []

    for session, line, source in history:
        if not source.strip():
            continue

        cells.append(
            f"\n\n# ===== CELL {line} =====\n\n{source}"
        )

    OUT.write_text(
        "".join(cells),
        encoding="utf-8"
    )

    print("Recovered cells:", len(cells))
    print("Saved to:", OUT)