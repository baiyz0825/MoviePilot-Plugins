"""Language normalization, subtitle-track detection, and translation validation."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Cue, CueGraph


_ALIASES = {
    "zh": "zh-Hans", "zh-cn": "zh-Hans", "zh-sg": "zh-Hans", "zh-hans": "zh-Hans",
    "chi": "zh-Hans", "zho": "zh-Hans", "cmn": "zh-Hans", "chs": "zh-Hans", "cn": "zh-Hans",
    "sc": "zh-Hans", "gb": "zh-Hans", "simplified": "zh-Hans", "simplified-chinese": "zh-Hans",
    "chinese-simplified": "zh-Hans", "简中": "zh-Hans", "简体": "zh-Hans",
    "zh-tw": "zh-Hant", "zh-hk": "zh-Hant", "zh-mo": "zh-Hant", "zh-hant": "zh-Hant",
    "cht": "zh-Hant", "tc": "zh-Hant", "big5": "zh-Hant", "traditional": "zh-Hant",
    "traditional-chinese": "zh-Hant", "chinese-traditional": "zh-Hant", "繁中": "zh-Hant", "繁体": "zh-Hant",
    "en": "en", "eng": "en", "english": "en",
    "ja": "ja", "jpn": "ja", "jp": "ja", "japanese": "ja",
    "ko": "ko", "kor": "ko", "kr": "ko", "korean": "ko",
    "fr": "fr", "fra": "fr", "fre": "fr", "french": "fr",
    "de": "de", "deu": "de", "ger": "de", "german": "de",
    "es": "es", "spa": "es", "spanish": "es",
    "it": "it", "ita": "it", "italian": "it",
    "pt": "pt", "por": "pt", "portuguese": "pt",
    "ru": "ru", "rus": "ru", "russian": "ru",
    "ar": "ar", "ara": "ar", "arabic": "ar",
    "hi": "hi", "hin": "hi", "hindi": "hi",
    "bn": "bn", "ben": "bn", "bengali": "bn",
    "ur": "ur", "urd": "ur", "urdu": "ur",
    "id": "id", "ind": "id", "indonesian": "id",
    "pcm": "pcm", "nigerian-pidgin": "pcm",
    "mr": "mr", "mar": "mr", "marathi": "mr",
    "te": "te", "tel": "te", "telugu": "te",
    "tr": "tr", "tur": "tr", "turkish": "tr",
    "ta": "ta", "tam": "ta", "tamil": "ta",
    "vi": "vi", "vie": "vi", "vietnamese": "vi",
    "tl": "tl", "tgl": "tl", "fil": "tl", "tagalog": "tl", "filipino": "tl",
    "fa": "fa", "fas": "fa", "per": "fa", "persian": "fa",
    "jv": "jv", "jav": "jv", "javanese": "jv",
    "ha": "ha", "hau": "ha", "hausa": "ha",
    "sw": "sw", "swa": "sw", "swahili": "sw",
}
_UNKNOWN = {
    "", "source", "src", "original", "orig", "unknown", "und", "unk", "auto", "none", "null",
    "mul", "multi", "multilingual", "bilingual", "dual", "双语", "多语",
}
_LATIN_LANGS = {"en", "fr", "de", "es", "it", "pt", "id", "pcm", "tr", "vi", "tl", "ha", "sw", "jv"}

_HAN = re.compile(r"[㐀-䶿一-鿿]")
_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž]")
_URL_OR_EMAIL = re.compile(r"^(?:https?://|www\.|mailto:|[^\s@]+@[^\s@]+\.)\S+$", re.IGNORECASE)
_MODEL_TOKEN = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d][A-Za-z\d._/+:#-]*$")
_ACRONYM = re.compile(r"^[A-Z][A-Z\d._/+:#-]{1,11}s?$")
_CAMEL_OR_BRAND = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])[A-Za-z][A-Za-z._/+:#-]{1,30}$")
_NAME_TOKEN = re.compile(r"^(?:[A-Z][a-z]{1,20}|[A-Z]\.)$")
_BILINGUAL_HINT = re.compile(r"双语|中英|英中|bilingual|dual[ ._-]*(?:sub|audio|lang)", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^\s*\[(?:翻译失败|translation failed)\]\s*$", re.IGNORECASE)

# A small set is enough to choose Hans/Hant when the track metadata is absent.
_SIMPLIFIED_ONLY = set("这为国发后里时会说们个来对从还开关门见听爱边过点体东车书云万与专业乐")
_TRADITIONAL_ONLY = set("這為國發後裡時會說們個來對從還開關門見聽愛邊過點體東車書雲萬與專業樂")
_COMMON_TRANSLATABLE_TITLE_WORDS = {
    "hello", "world", "welcome", "thanks", "thank", "yes", "no", "okay", "ok", "please",
    "good", "morning", "night", "sorry", "help", "stop", "wait", "go", "come", "love",
}


def canonical_language(value: str) -> str:
    """Map common ISO-639/BCP-47 spellings to stable plugin language IDs."""
    raw = str(value or "").strip().replace("_", "-")
    lowered = raw.lower()
    if lowered in _UNKNOWN:
        return "source"
    if lowered.startswith("zh-hant"):
        return "zh-Hant"
    if lowered.startswith("zh-hans"):
        return "zh-Hans"
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    base = lowered.split("-", 1)[0]
    if base in _ALIASES:
        return _ALIASES[base]
    return raw or "source"


def is_bilingual_hint(value: str) -> bool:
    raw = str(value or "").strip()
    return raw.lower() in {"bilingual", "dual", "mul", "multi", "双语", "多语"} or bool(_BILINGUAL_HINT.search(raw))


def _counts(text: str) -> Dict[str, int]:
    body = str(text or "")
    return {
        "han": len(_HAN.findall(body)),
        "kana": len(_KANA.findall(body)),
        "hangul": len(_HANGUL.findall(body)),
        "cyrillic": len(_CYRILLIC.findall(body)),
        "arabic": len(_ARABIC.findall(body)),
        "latin": len(_LATIN.findall(body)),
    }


def _chinese_variant(text: str, hint: str = "source") -> str:
    simplified = sum(char in _SIMPLIFIED_ONLY for char in str(text or ""))
    traditional = sum(char in _TRADITIONAL_ONLY for char in str(text or ""))
    if traditional > simplified:
        return "zh-Hant"
    if simplified > traditional:
        return "zh-Hans"
    return "zh-Hant" if canonical_language(hint) == "zh-Hant" else "zh-Hans"


def text_languages(text: str, hint: str = "source") -> Set[str]:
    """Return scripts that have meaningful evidence in one subtitle string.

    The result is deliberately conservative. Mixed-script lines are returned as
    multiple languages so callers can keep them intact instead of guessing how
    to split a sentence in the middle.
    """
    counts = _counts(text)
    east = counts["han"] + counts["kana"] + counts["hangul"]
    languages: Set[str] = set()
    if counts["hangul"]:
        languages.add("ko")
    elif counts["kana"]:
        languages.add("ja")
    elif counts["han"]:
        languages.add(_chinese_variant(text, hint))
    if counts["cyrillic"] >= 2:
        languages.add("ru")
    if counts["arabic"] >= 2:
        languages.add("ar")
    if counts["latin"] >= 2 and (not east or counts["latin"] >= max(4, east // 2)):
        normalized = canonical_language(hint)
        languages.add(normalized if normalized in _LATIN_LANGS else "en")
    return languages


def detect_text_language(text: str, hint: str = "source") -> str:
    """Detect one language only when the text provides unambiguous script evidence."""
    languages = text_languages(text, hint)
    if len(languages) == 1:
        return next(iter(languages))
    if not languages:
        return "source"
    counts = _counts(text)
    ranked = []
    if counts["hangul"]:
        ranked.append((counts["hangul"] + counts["han"], "ko"))
    elif counts["kana"]:
        ranked.append((counts["kana"] + counts["han"], "ja"))
    elif counts["han"]:
        ranked.append((counts["han"], _chinese_variant(text, hint)))
    if counts["cyrillic"]:
        ranked.append((counts["cyrillic"], "ru"))
    if counts["arabic"]:
        ranked.append((counts["arabic"], "ar"))
    if counts["latin"]:
        normalized = canonical_language(hint)
        ranked.append((counts["latin"], normalized if normalized in _LATIN_LANGS else "en"))
    ranked.sort(reverse=True)
    if len(ranked) == 1 or ranked[0][0] >= max(4, ranked[1][0] * 3):
        return ranked[0][1]
    return "source"


def infer_source_language(hint: str, texts: Iterable[str]) -> str:
    """Normalize track metadata and correct it when aggregate text clearly disagrees."""
    normalized = canonical_language(hint)
    sample = "\n".join(str(text or "") for text in texts)[:30000]
    counts = _counts(sample)
    script_chars = sum(counts.values())
    if not script_chars:
        return normalized
    if counts["hangul"] >= 3 and counts["hangul"] / script_chars >= 0.16:
        return "ko"
    if counts["kana"] >= 2 and counts["kana"] / script_chars >= 0.08:
        return "ja"
    if counts["han"] >= 3 and counts["han"] / script_chars >= 0.16:
        return _chinese_variant(sample, normalized)
    if counts["cyrillic"] >= 3 and counts["cyrillic"] / script_chars >= 0.2:
        return "ru"
    if counts["arabic"] >= 3 and counts["arabic"] / script_chars >= 0.2:
        return "ar"
    if counts["latin"] >= 8 and counts["latin"] / script_chars >= 0.72:
        return normalized if normalized in _LATIN_LANGS else "en"
    return normalized


def split_bilingual_lines(text: str, hint: str = "source") -> Tuple[Dict[str, str], bool]:
    """Split line-aligned bilingual text; return (tracks, looked_mixed_but_uncertain)."""
    lines = [line.strip() for line in str(text or "").replace("\r", "").split("\n") if line.strip()]
    mixed_hint = is_bilingual_hint(hint)
    if len(lines) < 2:
        languages = text_languages(text, hint)
        if len(languages) == 1:
            return {}, False
        return {}, mixed_hint or len(languages) > 1
    grouped: Dict[str, List[str]] = {}
    uncertain = False
    for line in lines:
        languages = text_languages(line, hint)
        if len(languages) != 1:
            # A punctuation/model-number line cannot be assigned safely; a line
            # containing two scripts is not line-aligned bilingual data.
            uncertain = True
            continue
        language = next(iter(languages))
        grouped.setdefault(language, []).append(line)
    if not uncertain and len(grouped) == 2:
        return {language: "\n".join(parts) for language, parts in grouped.items()}, False
    all_languages = text_languages(text, hint)
    looks_mixed = uncertain or len(all_languages) > 1 or len(grouped) > 1 or (mixed_hint and not grouped)
    return {}, bool(looks_mixed)


def _comparison_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().casefold()


def can_remain_unchanged(text: str) -> bool:
    """Whether an untranslated token can legitimately be identical in another language."""
    body = str(text or "").strip()
    if not body:
        return False
    if not any((_HAN.search(body), _KANA.search(body), _HANGUL.search(body), _CYRILLIC.search(body), _ARABIC.search(body), _LATIN.search(body))):
        return True
    compact = body.strip(" \t\r\n.,!?;:，。！？；：()[]{}<>《》‘’“”\"'")
    if _URL_OR_EMAIL.fullmatch(compact):
        return True
    tokens = [token for token in re.split(r"\s+", compact) if token]
    if tokens and all(
        _MODEL_TOKEN.fullmatch(token) or _ACRONYM.fullmatch(token) or _CAMEL_OR_BRAND.fullmatch(token) or token.isdigit()
        for token in tokens
    ):
        return True
    if 1 <= len(tokens) <= 3 and all(_NAME_TOKEN.fullmatch(token) for token in tokens):
        lowered = {token.rstrip(".").lower() for token in tokens}
        return not bool(lowered & _COMMON_TRANSLATABLE_TITLE_WORDS)
    return False


def _target_script_missing(source_counts: Dict[str, int], target_counts: Dict[str, int], target_lang: str) -> bool:
    source_letters = sum(source_counts.values())
    if not source_letters:
        return False
    if target_lang in {"zh-Hans", "zh-Hant"}:
        return target_counts["han"] == 0 and (
            source_counts["latin"] >= 6
            or source_counts["kana"] + source_counts["hangul"] + source_counts["cyrillic"] >= 3
        )
    if target_lang == "en":
        non_latin_source = (
            source_counts["han"] + source_counts["kana"] + source_counts["hangul"]
            + source_counts["cyrillic"] + source_counts["arabic"]
        )
        return target_counts["latin"] < 2 and non_latin_source >= 3
    if target_lang == "ja":
        return target_counts["kana"] + target_counts["han"] == 0 and source_letters >= 4
    if target_lang == "ko":
        return target_counts["hangul"] == 0 and source_letters >= 4
    if target_lang == "ru":
        return target_counts["cyrillic"] < 2 and source_letters >= 4
    if target_lang == "ar":
        return target_counts["arabic"] < 2 and source_letters >= 4
    if target_lang in _LATIN_LANGS:
        return target_counts["latin"] < 2 and source_letters >= 4
    return False


def translation_validation_reason(source: str, translated: str, target: str, source_lang: str = "source") -> str:
    """Return an empty string for a plausible translation, otherwise a stable reason code."""
    original = str(source or "").strip()
    candidate = str(translated or "").strip()
    target_lang = canonical_language(target)
    if not candidate or _PLACEHOLDER.match(candidate):
        return "empty"
    same = _comparison_text(original) == _comparison_text(candidate)
    if same:
        return "" if can_remain_unchanged(original) else "unchanged"
    if original and len(_comparison_text(original)) >= 8:
        source_flat = _comparison_text(original)
        candidate_flat = _comparison_text(candidate)
        if source_flat in candidate_flat and len(candidate_flat) > len(source_flat) * 1.25:
            return "source_mixed_into_translation"
    source_counts = _counts(original)
    target_counts = _counts(candidate)
    if _target_script_missing(source_counts, target_counts, target_lang):
        return "target_script_missing"
    return ""


def translation_is_valid(source: str, translated: str, target: str, source_lang: str = "source") -> bool:
    return not translation_validation_reason(source, translated, target, source_lang)


def _add_issue(cue: "Cue", code: str, message: str) -> None:
    from .models import Issue

    if any(item.code == code and item.message == message for item in cue.issues):
        return
    cue.issues.append(Issue(code=code, message=message, cue_id=cue.cue_id))


def _canonicalize_cue_keys(cue: "Cue") -> None:
    normalized: Dict[str, str] = {}
    for raw_key, raw_text in list(cue.texts.items()):
        key = canonical_language(raw_key)
        text = str(raw_text or "")
        if key not in normalized or not normalized[key]:
            normalized[key] = text
            continue
        if _comparison_text(normalized[key]) == _comparison_text(text):
            continue
        # Do not discard conflicting user data merely because two aliases map
        # to the same canonical key. Keep the raw key and make it visible.
        preserved_key = str(raw_key or "source")
        if preserved_key in normalized:
            preserved_key = f"{preserved_key}-conflict"
        normalized[preserved_key] = text
        _add_issue(cue, "language_key_conflict", f"语种别名 {raw_key} 归一化后与已有文本冲突，已保留待确认")
    cue.texts = normalized


def _remove_mislabeled_source_duplicates(graph: "CueGraph", source_lang: str) -> None:
    source_key = canonical_language(source_lang)
    if source_key == "source":
        return
    for cue in graph.all_cues():
        source_text = cue.text(source_key)
        if not source_text or can_remain_unchanged(source_text):
            continue
        source_counts = _counts(source_text)
        for raw_key, text in list(cue.texts.items()):
            target_key = canonical_language(raw_key)
            if target_key in {"source", source_key}:
                continue
            if _comparison_text(text) != _comparison_text(source_text):
                continue
            if not _target_script_missing(source_counts, _counts(text), target_key):
                continue
            cue.texts.pop(raw_key, None)
            _add_issue(
                cue,
                "mislabeled_source_duplicate",
                f"已移除误写入 {target_key} 的重复 {source_key} 源文本",
            )


def normalize_graph_source_language(graph: "CueGraph") -> "CueGraph":
    """Canonicalize keys and migrate a wrongly-labelled single source track.

    A confirmed translation under another key is never overwritten. If the
    source cannot be inferred reliably, data stays under ``source`` and a Cue
    issue makes the uncertainty visible to the workbench.
    """
    original_hint = canonical_language(graph.source_lang)
    for cue in graph.all_cues():
        _canonicalize_cue_keys(cue)

    dialogue = graph.sorted_cues()
    source_texts: List[str] = []
    for cue in dialogue:
        if cue.text(original_hint):
            source_texts.append(cue.text(original_hint))
        elif cue.text("source"):
            source_texts.append(cue.text("source"))
        elif len(cue.texts) == 1:
            source_texts.append(cue.text())
    source_sample = "\n".join(source_texts)
    mixed_unknown_source = original_hint == "source" and len(text_languages(source_sample, original_hint)) > 1
    inferred = "source" if mixed_unknown_source else infer_source_language(original_hint, source_texts)

    # Correct the declared source before deciding that duplicate keys form a
    # real bilingual track. This removes {zh-Hans: English, en: English} safely.
    if inferred != original_hint and inferred != "source":
        for cue in graph.all_cues():
            old_text = cue.texts.get(original_hint)
            if not old_text:
                continue
            existing = cue.texts.get(inferred)
            cue.texts.pop(original_hint, None)
            if not existing:
                cue.texts[inferred] = old_text
            elif _comparison_text(existing) != _comparison_text(old_text):
                _add_issue(cue, "source_text_conflict", f"错误的 {original_hint} 源轨与已有 {inferred} 文本冲突；保留已有正确语种文本")
        graph.source_lang = inferred
        _remove_mislabeled_source_duplicates(graph, inferred)
        return graph

    known_keys = {
        canonical_language(key)
        for cue in dialogue
        for key, text in cue.texts.items()
        if text and canonical_language(key) != "source"
    }
    if inferred == "source" and len(known_keys) == 1 and not any(cue.text("source") for cue in dialogue):
        # ASS style/script parsing may already have supplied a trustworthy key
        # even when a very short cue is too small for aggregate inference.
        graph.source_lang = next(iter(known_keys))
        _remove_mislabeled_source_duplicates(graph, graph.source_lang)
        return graph

    multi_track = len(known_keys) >= 2 and any(len([text for text in cue.texts.values() if text]) >= 2 for cue in dialogue)
    if multi_track:
        if original_hint in known_keys:
            inferred = original_hint
        elif "en" in known_keys:
            inferred = "en"
        else:
            inferred = sorted(known_keys)[0]
        if original_hint == "source" and dialogue:
            _add_issue(dialogue[0], "source_language_ambiguous", f"双语字幕未声明原始语种，暂以 {inferred} 作为翻译源")
        graph.source_lang = inferred
        _remove_mislabeled_source_duplicates(graph, inferred)
        return graph

    graph.source_lang = inferred
    if inferred == "source":
        for cue in dialogue:
            if cue.text():
                _add_issue(cue, "source_language_uncertain", "无法可靠判断字幕源语言，已保留原文等待确认")
    elif inferred == original_hint and original_hint != "source":
        observed = detect_text_language("\n".join(source_texts), original_hint)
        if observed not in {"source", original_hint}:
            for cue in dialogue:
                if cue.text(original_hint):
                    _add_issue(cue, "source_language_conflict_uncertain", f"正文脚本疑似 {observed}，但证据不足以覆盖 {original_hint} 元数据")
    _remove_mislabeled_source_duplicates(graph, graph.source_lang)
    return graph
