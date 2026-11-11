"""媒体服务器与播放客户端的字幕命名规则。

命名配置集中在同目录的 ``naming_presets.json``，前端预览导入同一份数据。
字幕格式单独选择；叠行字幕的文件名使用主语种码，双语文本保留在字幕内容中。
MoviePilot 兼容格式、旧版 web 配置也保留为独立 profile。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .models import FORMATS, LAYOUTS
from .style import style_from_config, style_sizes

LANG_ALIASES = {
    "zh": "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-hans": "zh-Hans",
    "chi": "zh-Hans",
    "chs": "zh-Hans",
    "cn": "zh-Hans",
    "zh-tw": "zh-Hant",
    "zh-hant": "zh-Hant",
    "zh-hk": "zh-Hant",
    "cht": "zh-Hant",
    "eng": "en",
    "en": "en",
    "jpn": "ja",
    "ja": "ja",
    "kor": "ko",
    "ko": "ko",
}

LANG_CATALOG = {
    "zh-Hans": {"label": "简中"},
    "zh-Hant": {"label": "繁中"},
    "en": {"label": "英文"},
    "ja": {"label": "日文"},
    "ko": {"label": "韩文"},
}

with Path(__file__).with_name("naming_presets.json").open("r", encoding="utf-8") as _preset_file:
    _PRESET_ITEMS = json.load(_preset_file)["profiles"]
PRESETS = {item["id"]: item for item in _PRESET_ITEMS}
PRESET_IDS = tuple(PRESETS)


def canonical_lang(lang_id: str) -> str:
    text = str(lang_id or "").strip()
    return LANG_ALIASES.get(text.lower(), text) if text else "zh-Hans"


def lang_code(lang_id: str, preset: str) -> str:
    lang = canonical_lang(lang_id)
    profile = PRESETS.get(preset) or PRESETS["fnos"]
    codes = profile.get("language_codes") or {}
    return str(codes.get(lang) or codes.get("zh-Hans") or lang)


def _stem_separator(preset: str) -> str:
    profile = PRESETS.get(preset) or PRESETS["fnos"]
    return str(profile.get("stem_separator") or ".")


def font_size_for_rank(rank: int, config: Dict[str, Any] | None = None) -> int:
    sizes = style_sizes(style_from_config(config), 3)
    if rank < 0:
        return sizes[0]
    if rank >= len(sizes):
        return sizes[-1]
    return sizes[rank]


def sanitize_stem(path: str) -> str:
    stem = Path(path).stem if path else "media"
    return stem or "media"


def build_filename(
    stem: str,
    langs: Sequence[str],
    ext: str,
    *,
    title: str = "",
    flags: Sequence[str] = (),
    flag_first: bool = False,
    stem_separator: str = ".",
) -> str:
    lang_part = ".".join(lang for lang in langs if lang)
    flag_parts = [flag for flag in flags if flag]
    suffix_parts: List[str] = []
    if flag_first:
        suffix_parts.extend(flag_parts)
        if lang_part:
            suffix_parts.append(lang_part)
        if title:
            suffix_parts.append(title)
    else:
        if lang_part:
            suffix_parts.append(lang_part)
        if title:
            suffix_parts.append(title)
        suffix_parts.extend(flag_parts)
    suffix = ".".join(suffix_parts)
    prefix = f"{stem}{stem_separator}{suffix}" if suffix else stem
    return f"{prefix}.{ext.lstrip('.')}"


def _flag_first(preset: str) -> bool:
    return bool((PRESETS.get(preset) or {}).get("flag_first"))


def _default_flags(mark_default: bool, lang_id: str, preset: str) -> List[str]:
    profile = PRESETS.get(preset) or PRESETS["fnos"]
    if not mark_default or not profile.get("supports_default"):
        return []
    if profile.get("default_language") == "zh" and not canonical_lang(lang_id).startswith("zh"):
        return []
    return ["default"]


def _stacked_filename(stem: str, codes: Sequence[str], fmt: str, *, preset: str, flag_first: bool) -> str:
    # Stacked text still has one track language. Keep the filename on the primary
    # language so media libraries never parse a made-up combined language code.
    primary = codes[0] if codes else lang_code("zh-Hans", preset)
    return build_filename(
        stem,
        [primary],
        fmt,
        flag_first=flag_first,
        stem_separator=_stem_separator(preset),
    )


def export_plan(config: Dict[str, Any], stem: str) -> List[Dict[str, Any]]:
    """语种 × 版式 × 格式 的笛卡尔积。特效 / SDH / ASR 由调用方另加。"""
    preset = str(config.get("export_preset") or "fnos")
    preset = preset if preset in PRESETS else "fnos"
    languages = list(config.get("target_languages") or ["zh-Hans"])[:3]
    layouts = [item for item in (config.get("export_layouts") or ["mono"]) if item in LAYOUTS]
    formats = [item for item in (config.get("export_formats") or ["srt"]) if item in FORMATS]
    profile = PRESETS[preset]
    mark_default = bool(config.get("mark_default")) and bool(profile.get("supports_default"))
    stack = str(config.get("lang_stack") or "main_bottom")
    flag_first = _flag_first(preset)
    stem_separator = _stem_separator(preset)
    files: List[Dict[str, Any]] = []
    if not languages or not layouts or not formats:
        return files

    # 只要大小关系且勾了叠行，SRT 无法表达字号，默认再补一份 ASS。
    if "stacked" in layouts and len(languages) > 1 and "ass" not in formats:
        formats = list(formats) + ["ass"]

    for layout in layouts:
        for fmt in formats:
            if layout == "mono":
                primary = languages[0]
                flags = _default_flags(mark_default, primary, preset)
                files.append({
                    "kind": "dialogue",
                    "layout": layout,
                    "format": fmt,
                    "langs": [primary],
                    "codes": [lang_code(primary, preset)],
                    "filename": build_filename(stem, [lang_code(primary, preset)], fmt, flags=flags, flag_first=flag_first, stem_separator=stem_separator),
                    "mark_default": bool(flags),
                    "stack": stack,
                    "sizes": [font_size_for_rank(0, config)],
                })
            elif layout == "stacked":
                if len(languages) < 2:
                    continue
                codes = [lang_code(item, preset) for item in languages]
                files.append({
                    "kind": "dialogue",
                    "layout": layout,
                    "format": fmt,
                    "langs": list(languages),
                    "codes": codes,
                    "filename": _stacked_filename(stem, codes, fmt, preset=preset, flag_first=flag_first),
                    "mark_default": False,
                    "stack": stack,
                    "sizes": [font_size_for_rank(index, config) for index in range(len(languages))],
                })
            elif layout == "split":
                for index, lang in enumerate(languages):
                    flags = _default_flags(mark_default, lang, preset) if index == 0 else []
                    files.append({
                        "kind": "dialogue",
                        "layout": layout,
                        "format": fmt,
                        "langs": [lang],
                        "codes": [lang_code(lang, preset)],
                        "filename": build_filename(stem, [lang_code(lang, preset)], fmt, flags=flags, flag_first=flag_first, stem_separator=stem_separator),
                        "mark_default": bool(flags),
                        "stack": stack,
                        "sizes": [font_size_for_rank(0, config)],
                    })
    return files


def extra_tracks(config: Dict[str, Any], stem: str, *, asr_ran: bool = False, source_lang: str = "en") -> List[Dict[str, Any]]:
    """特效 / SDH / ASR 原轨。不进乘法，各有开关。"""
    preset = str(config.get("export_preset") or "fnos")
    preset = preset if preset in PRESETS else "fnos"
    files: List[Dict[str, Any]] = []
    primary = (config.get("target_languages") or ["zh-Hans"])[0]
    flag_first = _flag_first(preset)
    stem_separator = _stem_separator(preset)
    primary_code = lang_code(primary, preset)
    spec = style_from_config(config)
    if config.get("effects_enabled"):
        files.append({
            "kind": "notes",
            "layout": "notes",
            "format": "ass",
            "langs": [primary],
            "codes": [primary_code],
            "filename": build_filename(stem, [primary_code], "ass", title="notes", flag_first=flag_first, stem_separator=stem_separator),
            "mark_default": False,
            "stack": "",
            "sizes": [spec["note_size"]],
        })
    if config.get("enable_sdh"):
        files.append({
            "kind": "sdh",
            "layout": "sdh",
            "format": "srt",
            "langs": [primary],
            "codes": [primary_code],
            "filename": build_filename(stem, [primary_code], "srt", title="sdh", flag_first=flag_first, stem_separator=stem_separator),
            "mark_default": False,
            "stack": "",
            "sizes": [font_size_for_rank(0, config)],
        })
    if config.get("save_asr_track") and asr_ran:
        asr_code = lang_code(source_lang, preset)
        files.append({
            "kind": "asr",
            "layout": "asr",
            "format": "srt",
            "langs": [canonical_lang(source_lang)],
            "codes": [asr_code],
            "filename": build_filename(stem, [asr_code], "srt", title="asr", flag_first=flag_first, stem_separator=stem_separator),
            "mark_default": False,
            "stack": "",
            "sizes": [font_size_for_rank(0, config)],
        })
    return files


def apply_preset(config: Dict[str, Any], preset: str) -> Dict[str, Any]:
    """Apply a profile's supported default-marker behavior without changing formats."""
    next_config = dict(config)
    spec = PRESETS.get(preset) or PRESETS["fnos"]
    next_config["export_preset"] = preset if preset in PRESETS else "fnos"
    next_config["mark_default"] = bool(spec.get("supports_default") and spec.get("mark_default"))
    return next_config


def validate_target_languages(value: Any) -> List[str]:
    """最少 1 个，最多 3 个，保序。最后一个不能被拿空。"""
    items: Iterable[Any]
    if isinstance(value, str):
        items = [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
    else:
        items = value or []
    langs: List[str] = []
    for item in items:
        lang = canonical_lang(item)
        if lang in LANG_CATALOG and lang not in langs:
            langs.append(lang)
        if len(langs) >= 3:
            break
    return langs or ["zh-Hans"]
