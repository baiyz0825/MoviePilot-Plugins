"""导出文件名与语种码。

合同：`{stem}[.{flags}][.{lang}][.{title}][.{flags}].{ext}`
搜索偏好 ≠ 导出包。预设只改默认勾选和语言码，不是最终文件清单。
历史版本记在插件库，不再用 `.ai` / `.aiasr` 当文件名。
ASR 原轨只有本次真的跑了 Whisper 才写，不标 default。
特效 notes.ass、SDH、ASR 不进「语种 × 版式 × 格式」乘法。

语言码对齐主流媒体库 / 播放器，而不是内部 id：
- MoviePilot 整理：`DEFAULT_SUB=zh-cn` → `{stem}.default.chi.zh-cn.ext`
- Emby 官方：`zh-CN` / `zh-TW`，`.default` / `.forced` / `.sdh`
- Jellyfin：点分 token，语言与 default/sdh 任意顺序
- Plex：ISO 639-1/2，`chi` / `eng`，不认 default
- Infuse：同目录、同 stem；中文官方建议 `cn`，也认 `zh-CN`
- 飞牛影视：认 `chs`，不认 `zh-Hans`
"""

from __future__ import annotations

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
    "zh-Hans": {
        "label": "简中",
        "codes": {
            "library_zh": "chi.zh-cn",
            "plex": "chi",
            "web": "zh-CN",
            "fnos": "chs",
            "legacy": "chi",
        },
    },
    "zh-Hant": {
        "label": "繁中",
        "codes": {
            "library_zh": "zh-tw",
            "plex": "cht",
            "web": "zh-TW",
            "fnos": "cht",
            "legacy": "cht",
        },
    },
    "en": {
        "label": "英文",
        "codes": {
            "library_zh": "eng",
            "plex": "eng",
            "web": "en",
            "fnos": "eng",
            "legacy": "eng",
        },
    },
    "ja": {
        "label": "日文",
        "codes": {
            "library_zh": "ja",
            "plex": "jpn",
            "web": "ja",
            "fnos": "jpn",
            "legacy": "jpn",
        },
    },
    "ko": {
        "label": "韩文",
        "codes": {
            "library_zh": "ko",
            "plex": "kor",
            "web": "ko",
            "fnos": "kor",
            "legacy": "kor",
        },
    },
}

PRESETS = {
    "library_zh": {
        "label": "媒体库 / MoviePilot",
        "hint": "default.chi.zh-cn · 对齐整理记录",
        "mark_default": True,
        "formats": ["srt"],
        "flag_first": True,
    },
    "plex": {
        "label": "Plex",
        "hint": "chi / eng · 不标 default",
        "mark_default": False,
        "formats": ["srt"],
        "flag_first": False,
    },
    "fnos": {
        "label": "飞牛影视",
        "hint": "chs · 飞牛不认 zh-Hans",
        "mark_default": False,
        "formats": ["srt"],
        "flag_first": False,
    },
    "web": {
        "label": "Infuse / 网页",
        "hint": "zh-CN · 再勾 VTT",
        "mark_default": True,
        "formats": ["srt", "vtt"],
        "flag_first": False,
    },
    "legacy": {
        "label": "兼容旧库",
        "hint": "chi / chi&eng",
        "mark_default": False,
        "formats": ["srt"],
        "flag_first": False,
    },
}

PRESET_IDS = tuple(PRESETS)


def canonical_lang(lang_id: str) -> str:
    text = str(lang_id or "").strip()
    return LANG_ALIASES.get(text.lower(), text) if text else "zh-Hans"


def lang_code(lang_id: str, preset: str) -> str:
    lang = canonical_lang(lang_id)
    item = LANG_CATALOG.get(lang) or LANG_CATALOG["zh-Hans"]
    return item["codes"].get(preset) or item["codes"]["library_zh"]


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
) -> str:
    parts = [stem]
    lang_part = ".".join(lang for lang in langs if lang)
    flag_parts = [flag for flag in flags if flag]
    if flag_first:
        parts.extend(flag_parts)
        if lang_part:
            parts.append(lang_part)
        if title:
            parts.append(title)
    else:
        if lang_part:
            parts.append(lang_part)
        if title:
            parts.append(title)
        parts.extend(flag_parts)
    suffix = ext.lstrip(".")
    return ".".join(parts) + f".{suffix}"


def _flag_first(preset: str) -> bool:
    return bool((PRESETS.get(preset) or {}).get("flag_first"))


def _default_flags(mark_default: bool, lang_id: str) -> List[str]:
    if mark_default and canonical_lang(lang_id).startswith("zh"):
        return ["default"]
    return []


def _stacked_filename(stem: str, codes: Sequence[str], fmt: str, *, preset: str, flag_first: bool) -> str:
    if preset == "legacy":
        return build_filename(stem, ["&".join(codes)], fmt, flag_first=flag_first)
    if preset == "fnos":
        return build_filename(stem, ["&".join(codes)], fmt, flag_first=flag_first)
    # 叠行文件只标主语种。Jellyfin 会把最后一个 token 当语言，写成 zh-CN.en 会被认成英文。
    primary = codes[0] if codes else "chi.zh-cn"
    return build_filename(stem, [primary], fmt, title="bilingual", flag_first=flag_first)


def export_plan(config: Dict[str, Any], stem: str) -> List[Dict[str, Any]]:
    """语种 × 版式 × 格式 的笛卡尔积。特效 / SDH / ASR 由调用方另加。"""
    preset = str(config.get("export_preset") or "library_zh")
    languages = list(config.get("target_languages") or ["zh-Hans"])[:3]
    layouts = [item for item in (config.get("export_layouts") or ["mono"]) if item in LAYOUTS]
    formats = [item for item in (config.get("export_formats") or ["srt"]) if item in FORMATS]
    mark_default = bool(config.get("mark_default")) and preset not in {"plex", "fnos"}
    stack = str(config.get("lang_stack") or "main_bottom")
    flag_first = _flag_first(preset)
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
                flags = _default_flags(mark_default, primary)
                files.append({
                    "kind": "dialogue",
                    "layout": layout,
                    "format": fmt,
                    "langs": [primary],
                    "codes": [lang_code(primary, preset)],
                    "filename": build_filename(stem, [lang_code(primary, preset)], fmt, flags=flags, flag_first=flag_first),
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
                    flags = _default_flags(mark_default, lang) if index == 0 else []
                    files.append({
                        "kind": "dialogue",
                        "layout": layout,
                        "format": fmt,
                        "langs": [lang],
                        "codes": [lang_code(lang, preset)],
                        "filename": build_filename(stem, [lang_code(lang, preset)], fmt, flags=flags, flag_first=flag_first),
                        "mark_default": bool(flags),
                        "stack": stack,
                        "sizes": [font_size_for_rank(0, config)],
                    })
    return files


def extra_tracks(config: Dict[str, Any], stem: str, *, asr_ran: bool = False, source_lang: str = "en") -> List[Dict[str, Any]]:
    """特效 / SDH / ASR 原轨。不进乘法，各有开关。"""
    preset = str(config.get("export_preset") or "library_zh")
    files: List[Dict[str, Any]] = []
    primary = (config.get("target_languages") or ["zh-Hans"])[0]
    flag_first = _flag_first(preset)
    primary_code = lang_code(primary, preset)
    spec = style_from_config(config)
    if config.get("effects_enabled"):
        files.append({
            "kind": "notes",
            "layout": "notes",
            "format": "ass",
            "langs": [primary],
            "codes": [primary_code],
            "filename": build_filename(stem, [primary_code], "ass", title="notes", flag_first=flag_first),
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
            "filename": build_filename(stem, [primary_code], "srt", title="sdh", flag_first=flag_first),
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
            "filename": build_filename(stem, [asr_code], "srt", title="asr", flag_first=flag_first),
            "mark_default": False,
            "stack": "",
            "sizes": [font_size_for_rank(0, config)],
        })
    return files


def apply_preset(config: Dict[str, Any], preset: str) -> Dict[str, Any]:
    """点预设只改默认勾选，仍可手改。Plex / 飞牛强制关掉 default。"""
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
        lang = canonical_lang(item)
        if lang in LANG_CATALOG and lang not in langs:
            langs.append(lang)
        if len(langs) >= 3:
            break
    return langs or ["zh-Hans"]
