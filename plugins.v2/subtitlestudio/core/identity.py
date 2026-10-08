"""媒体身份与入队去重键。

Job 两代都存同一套字段：media_source / media_id / tmdbid / doubanid / path。
页面展示和 5 分钟去重优先用「能合成的对 + 路径」。
V3 宿主通用链路必须成对传入；V2 事件经常只有 tmdbid / doubanid，
由 host 适配器填，不要在领域层假装事件已经带了 media_source。
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

# 内置来源用传输值，不要写枚举名 TMDB。
BUILTIN_SOURCES = {
    "themoviedb",
    "tmdb",
    "douban",
    "bangumi",
    "anilist",
    "imdb",
    "tvdb",
}

SOURCE_ALIASES = {
    "tmdb": "themoviedb",
    "themoviedb": "themoviedb",
    "douban": "douban",
    "bangumi": "bangumi",
    "anilist": "anilist",
    "imdb": "imdb",
    "tvdb": "tvdb",
}


def normalize_media_source(value: Any) -> str:
    text = str(value or "").strip().lower()
    return SOURCE_ALIASES.get(text, text)


def is_valid_media_id(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text != "0"


def build_media_key(media_source: Any, media_id: Any) -> str:
    """比较 / 缓存用的成对键。来源和 ID 必须同时有效，否则返回空串。"""
    source = normalize_media_source(media_source)
    media_id_text = str(media_id or "").strip()
    if not source or not is_valid_media_id(media_id_text):
        return ""
    return f"{source}:{media_id_text}"


def identity_from_v2_ids(tmdbid: Any = None, doubanid: Any = None) -> dict:
    """V2 事件只有散 ID 时，能合成对就合成，合不成留给路径键。"""
    if is_valid_media_id(tmdbid):
        return {
            "media_source": "themoviedb",
            "media_id": str(tmdbid).strip(),
            "tmdbid": str(tmdbid).strip(),
            "doubanid": str(doubanid).strip() if is_valid_media_id(doubanid) else "",
        }
    if is_valid_media_id(doubanid):
        return {
            "media_source": "douban",
            "media_id": str(doubanid).strip(),
            "tmdbid": "",
            "doubanid": str(doubanid).strip(),
        }
    return {"media_source": "", "media_id": "", "tmdbid": "", "doubanid": ""}


def identity_from_v3_pair(media_source: Any, media_id: Any, tmdbid: Any = None, doubanid: Any = None) -> dict:
    """V3 通用链路：同时为空或同时有效；空串 / \"0\" / 非法来源都不是身份。"""
    source = normalize_media_source(media_source)
    media_id_text = str(media_id or "").strip()
    valid_pair = bool(source) and is_valid_media_id(media_id_text)
    return {
        "media_source": source if valid_pair else "",
        "media_id": media_id_text if valid_pair else "",
        "tmdbid": str(tmdbid).strip() if is_valid_media_id(tmdbid) else (
            media_id_text if valid_pair and source == "themoviedb" else ""
        ),
        "doubanid": str(doubanid).strip() if is_valid_media_id(doubanid) else (
            media_id_text if valid_pair and source == "douban" else ""
        ),
    }


def debounce_key(identity: dict, path: Any) -> str:
    """同一媒体 5 分钟内只建一次任务用的键：成对身份优先，否则退回路径。"""
    pair = build_media_key(identity.get("media_source"), identity.get("media_id"))
    path_text = str(path or "").strip()
    if pair and path_text:
        return f"{pair}|{path_text}"
    if pair:
        return pair
    return path_text


def media_item_id(identity: dict, path: Any = "") -> str:
    """页面列表稳定 id，避免用数组下标当 key。"""
    key = debounce_key(identity, path) or str(path or "unknown")
    return quote(key, safe="")
