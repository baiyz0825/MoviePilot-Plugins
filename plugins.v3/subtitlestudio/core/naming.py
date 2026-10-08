"""导出文件名与语种码。

合同：`{stem}.{lang}[.{title}][.{flags}].{ext}`
搜索偏好 ≠ 导出包。预设只改默认勾选和语言码，不是最终文件清单。
历史版本记在插件库，不再用 `.ai` / `.aiasr` 当文件名。
ASR 原轨只有本次真的跑了 Whisper 才写 `Movie.en.asr.srt`，不标 default。
特效 notes.ass、SDH、ASR 不进「语种 × 版式 × 格式」乘法。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .models import FORMATS, LANG_FONT_SIZES, LAYOUTS

LANG_CATALOG = {
    "zh-Hans": {"label": "简中", "codes": {"library_zh": "zh-Hans", "plex": "chi", "web": "zh-Hans", "legacy": "chi"}},
    "zh-Hant": {"label": "繁中", "codes": {"library_zh": "zh-Hant", "plex": "cht", "web": "zh-Hant", "legacy": "cht"}},
    "en": {"label": "英文", "codes": {"library_zh": "en", "plex": "eng", "web": "en", "legacy": "eng"}},
    "ja": {"label": "日文", "codes": {"library_zh": "ja", "plex": "jpn", "web": "ja", "legacy": "jpn"}},
    "ko": {"label": "韩文", "codes": {"library_zh": "ko", "plex": "kor", "web": "ko", "legacy": "kor"}},
}

PRESETS = {
    "library_zh": {"label": "媒体库中文", "hint": "zh-Hans + default · 默认勾 SRT", "mark_default": True, "formats": ["srt"]},
    "plex": {"label": "Plex", "hint": "chi · 不标 default", "mark_default": False, "formats": ["srt"]},
    "web": {"label": "网页 / Infuse", "hint": "再勾 VTT", "mark_default": True, "formats": ["srt", "vtt"]},
    "legacy": {"label": "兼容旧库", "hint": "chi / chi&eng", "mark_default": False, "formats": ["srt"]},
}


def lang_code(lang_id: str, preset: str) -> str:
    item = LANG_CATALOG.get(lang_id) or LANG_CATALOG["zh-Hans"]
    return item["codes"].get(preset) or item["codes"]["library_zh"]


def font_size_for_rank(rank: int) -> int:
    if rank < 0:
        return LANG_FONT_SIZES[0]
    if rank >= len(LANG_FONT_SIZES):
        return LANG_FONT_SIZES[-1]
    return LANG_FONT_SIZES[rank]


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
) -> str:
    parts = [stem]
    if langs:
        parts.append(".".join(lang for lang in langs if lang))
    if title:
        parts.append(title)
    for flag in flags:
        if flag:
            parts.append(flag)
    suffix = ext.lstrip(".")
    return ".".join(parts) + f".{suffix}"


def export_plan(config: Dict[str, Any], stem: str) -> List[Dict[str, Any]]:
    """语种 × 版式 × 格式 的笛卡尔积。特效 / SDH / ASR 由调用方另加。"""
    preset = str(config.get("export_preset") or "library_zh")
    languages = list(config.get("target_languages") or ["zh-Hans"])[:3]
    layouts = [item for item in (config.get("export_layouts") or ["mono"]) if item in LAYOUTS]
    formats = [item for item in (config.get("export_formats") or ["srt"]) if item in FORMATS]
    mark_default = bool(config.get("mark_default")) and preset != "plex"
    stack = str(config.get("lang_stack") or "main_bottom")
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
                flags = ["default"] if mark_default and primary.startswith("zh") else []
                files.append({
                    "kind": "dialogue",
                    "layout": layout,
                    "format": fmt,
                    "langs": [primary],
                    "codes": [lang_code(primary, preset)],
                    "filename": build_filename(stem, [lang_code(primary, preset)], fmt, flags=flags),
                    "mark_default": bool(flags),
                    "stack": stack,
                    "sizes": [font_size_for_rank(0)],
                })
            elif layout == "stacked":
                if len(languages) < 2:
                    continue
                codes = [lang_code(item, preset) for item in languages]
                # 旧库叠行用 chi&eng 这种播放器认识的写法。
                joined = "&".join(codes) if preset == "legacy" else ".".join(codes)
                files.append({
                    "kind": "dialogue",
                    "layout": layout,
                    "format": fmt,
                    "langs": list(languages),
                    "codes": codes,
                    "filename": build_filename(stem, [joined] if preset == "legacy" else codes, fmt),
                    "mark_default": False,
                    "stack": stack,
                    "sizes": [font_size_for_rank(index) for index in range(len(languages))],
                })
            elif layout == "split":
                for index, lang in enumerate(languages):
                    flags = ["default"] if mark_default and index == 0 and lang.startswith("zh") else []
                    files.append({
                        "kind": "dialogue",
                        "layout": layout,
                        "format": fmt,
                        "langs": [lang],
                        "codes": [lang_code(lang, preset)],
                        "filename": build_filename(stem, [lang_code(lang, preset)], fmt, flags=flags),
                        "mark_default": bool(flags),
                        "stack": stack,
                        "sizes": [font_size_for_rank(0)],
                    })
    return files


def extra_tracks(config: Dict[str, Any], stem: str, *, asr_ran: bool = False, source_lang: str = "en") -> List[Dict[str, Any]]:
    """特效 / SDH / ASR 原轨。不进乘法，各有开关。"""
    preset = str(config.get("export_preset") or "library_zh")
    files: List[Dict[str, Any]] = []
    primary = (config.get("target_languages") or ["zh-Hans"])[0]
    if config.get("effects_enabled"):
        files.append({
            "kind": "notes",
            "layout": "notes",
            "format": "ass",
            "langs": [primary],
            "codes": [lang_code(primary, preset)],
            "filename": build_filename(stem, [lang_code(primary, preset)], "ass", title="notes"),
            "mark_default": False,
            "stack": "",
            "sizes": [16],
        })
    if config.get("enable_sdh"):
        files.append({
            "kind": "sdh",
            "layout": "sdh",
            "format": "srt",
            "langs": [primary],
            "codes": [lang_code(primary, preset)],
            "filename": build_filename(stem, [lang_code(primary, preset)], "srt", title="sdh"),
            "mark_default": False,
            "stack": "",
            "sizes": [22],
        })
    if config.get("save_asr_track") and asr_ran:
        files.append({
            "kind": "asr",
            "layout": "asr",
            "format": "srt",
            "langs": [source_lang],
            "codes": [source_lang],
            "filename": build_filename(stem, [source_lang], "srt", title="asr"),
            "mark_default": False,
            "stack": "",
            "sizes": [22],
        })
    return files


def apply_preset(config: Dict[str, Any], preset: str) -> Dict[str, Any]:
    """点预设只改默认勾选，仍可手改。Plex 强制关掉 default。"""
    next_config = dict(config)
    spec = PRESETS.get(preset) or PRESETS["library_zh"]
    next_config["export_preset"] = preset if preset in PRESETS else "library_zh"
    next_config["mark_default"] = bool(spec["mark_default"])
    formats = list(next_config.get("export_formats") or [])
    for item in spec["formats"]:
        if item not in formats:
            formats.append(item)
    next_config["export_formats"] = formats
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
        lang = str(item).strip()
        if lang in LANG_CATALOG and lang not in langs:
            langs.append(lang)
        if len(langs) >= 3:
            break
    return langs or ["zh-Hans"]
