"""Language normalization and lightweight subtitle-script detection."""

from __future__ import annotations

import re
from typing import Iterable


_ALIASES = {
    "zh": "zh-Hans", "zh-cn": "zh-Hans", "zh-sg": "zh-Hans", "zh-hans": "zh-Hans",
    "chi": "zh-Hans", "zho": "zh-Hans", "cmn": "zh-Hans", "chs": "zh-Hans", "cn": "zh-Hans",
    "zh-tw": "zh-Hant", "zh-hk": "zh-Hant", "zh-mo": "zh-Hant", "zh-hant": "zh-Hant", "cht": "zh-Hant",
    "en": "en", "eng": "en", "ja": "ja", "jpn": "ja", "ko": "ko", "kor": "ko",
    "fr": "fr", "fra": "fr", "fre": "fr", "de": "de", "deu": "de", "ger": "de",
    "es": "es", "spa": "es", "it": "it", "ita": "it", "pt": "pt", "por": "pt",
    "ru": "ru", "rus": "ru", "ar": "ar", "ara": "ar",
}

_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af]")
_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_LATIN = re.compile(r"[A-Za-z]")


def canonical_language(value: str) -> str:
    """Map common ISO-639/BCP-47 spellings to stable plugin language IDs."""
    raw = str(value or "").strip().replace("_", "-")
    if not raw:
        return "source"
    lowered = raw.lower()
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    base = lowered.split("-", 1)[0]
    if base in _ALIASES:
        return _ALIASES[base]
    return raw


def infer_source_language(hint: str, texts: Iterable[str]) -> str:
    """Normalize track metadata, correcting it when subtitle scripts clearly disagree."""
    normalized = canonical_language(hint)
    sample = "\n".join(str(text or "") for text in texts)
    sample = sample[:30000]
    han = len(_HAN.findall(sample))
    kana = len(_KANA.findall(sample))
    hangul = len(_HANGUL.findall(sample))
    cyrillic = len(_CYRILLIC.findall(sample))
    latin = len(_LATIN.findall(sample))
    script_chars = han + kana + hangul + cyrillic + latin
    if not script_chars:
        return normalized

    # Script evidence can correct a bad/missing media-stream language tag.
    if hangul >= 3 and hangul / script_chars >= 0.2:
        return "ko"
    if kana >= 2 and kana / script_chars >= 0.12:
        return "ja"
    if han >= 3 and han / script_chars >= 0.2:
        return "zh-Hans" if normalized not in {"zh-Hant"} else "zh-Hant"
    if cyrillic >= 3 and cyrillic / script_chars >= 0.2:
        return "ru"
    if latin >= 12 and latin / script_chars >= 0.8 and normalized in {"source", "zh-Hans", "zh-Hant", "ja", "ko"}:
        return "en"
    return normalized
