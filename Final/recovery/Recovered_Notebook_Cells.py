

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

for lang, code in [("English", "eng"), ("Telugu", "tel")]: