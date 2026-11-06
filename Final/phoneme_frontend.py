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
