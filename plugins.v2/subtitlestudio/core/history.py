"""把 MoviePilot 整理记录展开成可入队的视频文件。

插件没有「直接扫 Emby/Jellyfin 库」的官方接口。本地媒体库的标准入口是
TransferHistory.list_by_page（GET /api/v1/history/transfer 同源）。
一条整理记录可能对应多个 dest / files / dest_fileitem.path，这里拆开。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .identity import identity_from_v2_ids, identity_from_v3_pair

VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".ts", ".m2ts", ".mov", ".wmv", ".m4v", ".flv", ".webm", ".strm"}

NUMBER = re.compile(r"(\d+)")
EPISODE_MARKER = re.compile(r"(?i)s(\d{1,2})[ ._-]*e(\d{1,3})")


def media_type_of(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"tv", "电视剧", "series", "show", "shows"}:
        return "tv"
    if text in {"movie", "电影", "movies", "film"}:
        return "movie"
    return "tv" if "剧" in text else "movie"


def first_number(value: Any) -> Optional[int]:
    match = NUMBER.search(str(value or ""))
    if not match:
        return None
    number = int(match.group(1))
    return number or None


def history_object_to_dict(item: Any) -> Dict[str, Any]:
    """宿主模型或已经是 dict，都收成领域层能吃的字典。"""
    if isinstance(item, dict):
        return dict(item)
    return {
        "title": getattr(item, "title", None) or "",
        "year": getattr(item, "year", None) or "",
        "path": getattr(item, "dest", None) or getattr(item, "dest_file", None) or getattr(item, "src", None) or "",
        "src": getattr(item, "src", None) or getattr(item, "source", None) or "",
        "dest": getattr(item, "dest", None) or "",
        "dest_fileitem": getattr(item, "dest_fileitem", None),
        "dest_storage": getattr(item, "dest_storage", None) or "",
        "files": getattr(item, "files", None),
        "tmdbid": getattr(item, "tmdbid", None),
        "doubanid": getattr(item, "doubanid", None),
        "media_source": getattr(item, "media_source", None) or "",
        "media_id": getattr(item, "media_id", None) or "",
        "type": getattr(item, "type", None) or getattr(item, "media_type", None) or "",
        "season": getattr(item, "seasons", None) or getattr(item, "season", None),
        "episode": getattr(item, "episodes", None) or getattr(item, "episode", None),
        "poster": getattr(item, "image", None) or getattr(item, "poster", None) or "",
        "date": getattr(item, "date", None) or "",
        "status": getattr(item, "status", True),
    }


def _norm_path(value: str) -> str:
    return os.path.normcase(os.path.normpath(str(value or "").replace("\\", "/")))


def _is_under(path: str, roots: Iterable[str]) -> bool:
    candidate = _norm_path(path)
    for root in roots:
        base = _norm_path(root).rstrip("/")
        if base and (candidate == base or candidate.startswith(base + "/")):
            return True
    return False


def _episode_marker(path: str) -> tuple[int, int] | None:
    match = EPISODE_MARKER.search(str(path or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def resolve_library_path(job: Any, history_rows: Iterable[Any], library_roots: Iterable[str]) -> str:
    """Resolve an ingest/download path to its organized library destination.

    A path already present in transfer history or under a configured library root is
    final. Otherwise require an exact source-path or media-identity match in history.
    """
    source = str(getattr(job, "path", "") or "")
    if not source:
        return ""
    rows = [history_object_to_dict(row) for row in history_rows or []]
    known_destinations = set()
    for row in rows:
        source_paths = {
            _norm_path(row.get(key))
            for key in ("src", "source", "src_path", "source_path")
            if row.get(key)
        }
        explicit_dest = any(row.get(key) for key in ("dest", "dest_file", "dest_fileitem", "files"))
        for path in paths_from_history(row):
            if is_media_file(path) and (explicit_dest or _norm_path(path) not in source_paths):
                known_destinations.add(_norm_path(path))
    if _norm_path(source) in known_destinations:
        return source

    job_pair = identity_from_v3_pair(
        getattr(job, "media_source", ""), getattr(job, "media_id", ""),
        getattr(job, "tmdbid", ""), getattr(job, "doubanid", ""),
    )
    source_marker = _episode_marker(source)
    source_norm = _norm_path(source)
    for row in rows:
        sources = [row.get(key) for key in ("src", "source", "src_path", "source_path")]
        exact_source = source_norm in {_norm_path(value) for value in sources if value}
        row_pair = identity_from_history(row)
        same_pair = bool(
            job_pair.get("media_source") and job_pair.get("media_id")
            and row_pair.get("media_source") == job_pair.get("media_source")
            and row_pair.get("media_id") == job_pair.get("media_id")
        )
        same_legacy_id = bool(
            (job_pair.get("tmdbid") and row_pair.get("tmdbid") == job_pair.get("tmdbid"))
            or (job_pair.get("doubanid") and row_pair.get("doubanid") == job_pair.get("doubanid"))
        )
        if not (exact_source or same_pair or same_legacy_id):
            continue
        season = first_number(row.get("season") or row.get("seasons"))
        episode = first_number(row.get("episode") or row.get("episodes"))
        if source_marker and season and episode and source_marker != (season, episode):
            continue
        source_paths = {
            _norm_path(row.get(key))
            for key in ("src", "source", "src_path", "source_path")
            if row.get(key)
        }
        explicit_dest = any(row.get(key) for key in ("dest", "dest_file", "dest_fileitem", "files"))
        for candidate in paths_from_history(row):
            if not is_media_file(candidate) or _norm_path(candidate) == source_norm:
                continue
            if not explicit_dest and _norm_path(candidate) in source_paths:
                continue
            if not Path(candidate).is_file():
                continue
            candidate_marker = _episode_marker(candidate)
            if source_marker and candidate_marker and source_marker != candidate_marker:
                continue
            if source_marker and not candidate_marker and season and episode and source_marker != (season, episode):
                continue
            return candidate
    return source if _is_under(source, library_roots) else ""


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []
    if value:
        return [value]
    return []


def paths_from_history(entry: Dict[str, Any]) -> List[str]:
    """一条整理记录里可能有目标文件、fileitem、files JSON。"""
    found: List[str] = []
    fileitem = _as_dict(entry.get("dest_fileitem"))
    storage = str(fileitem.get("storage") or entry.get("dest_storage") or "local").lower()
    if storage and storage not in {"local", "localstorage", ""}:
        # 网盘存储没有本地音轨，手动识别也走不了 ASR；仍允许搜字幕，所以继续收路径。
        pass
    for raw in (
        fileitem.get("path"),
        entry.get("path"),
        entry.get("dest"),
        entry.get("dest_file"),
    ):
        text = str(raw or "").strip()
        if text:
            found.append(text)
    for item in _as_list(entry.get("files")):
        if isinstance(item, str) and item.strip():
            found.append(item.strip())
        elif isinstance(item, dict):
            text = str(item.get("path") or item.get("dest") or item.get("file") or "").strip()
            if text:
                found.append(text)
    unique: List[str] = []
    seen = set()
    for path in found:
        key = path.replace("\\", "/")
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def is_media_file(path: str) -> bool:
    lower = str(path or "").lower()
    return any(lower.endswith(suffix) for suffix in VIDEO_SUFFIXES)


def identity_from_history(entry: Dict[str, Any]) -> Dict[str, str]:
    pair = identity_from_v3_pair(entry.get("media_source"), entry.get("media_id"), entry.get("tmdbid"), entry.get("doubanid"))
    if pair.get("media_source"):
        return pair
    return identity_from_v2_ids(entry.get("tmdbid"), entry.get("doubanid"))


def expand_history_rows(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    """整理记录 → 每个视频文件一条，供媒体页勾选入队。"""
    expanded: List[Dict[str, Any]] = []
    entries = [history_object_to_dict(raw) for raw in rows or []]
    # MoviePilot 会保留每次整理的历史。若旧记录的目标路径后来作为
    # 新记录的来源路径再次整理，旧目标只是中间副本，媒体页只展示链路末端。
    superseded_paths = {
        _norm_path(str(entry.get(key)))
        for entry in entries
        for key in ("src", "source", "src_path", "source_path")
        if entry.get(key)
        and entry.get("status") not in {False, 0, "0", "false", "False"}
        and is_media_file(str(entry.get(key)))
    }
    for entry in entries:
        if entry.get("status") in {False, 0, "0", "false", "False"}:
            continue
        identity = identity_from_history(entry)
        media_type = media_type_of(entry.get("type") or entry.get("media_type"))
        title = str(entry.get("title") or "").strip()
        year = str(entry.get("year") or "").strip()
        season = first_number(entry.get("season") or entry.get("seasons"))
        episode = first_number(entry.get("episode") or entry.get("episodes"))
        if media_type == "tv" and episode and not season:
            season = 1
        media_key = "|".join([
            media_type,
            identity.get("media_source") or "",
            identity.get("media_id") or str(identity.get("tmdbid") or ""),
            title,
            year,
        ])
        for path in paths_from_history(entry):
            if not is_media_file(path):
                continue
            if _norm_path(path) in superseded_paths:
                continue
            filename = path.replace("\\", "/").rsplit("/", 1)[-1]
            stem = filename.rsplit(".", 1)[0]
            expanded.append({
                "title": title or stem,
                "year": year,
                "path": path,
                "filename": filename,
                "type": media_type,
                "season": season,
                "episode": episode,
                "poster": str(entry.get("poster") or entry.get("image") or ""),
                "date": str(entry.get("date") or ""),
                "origin": "transfer_history",
                "media_key": media_key,
                "library_name": "MoviePilot 整理记录",
                **identity,
            })
    return expanded


def group_media_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for item in items or []:
        key = str(item.get("media_key") or item.get("path") or item.get("id") or "")
        if key not in groups:
            groups[key] = {
                "id": key,
                "media_key": key,
                "title": item.get("title") or item.get("filename") or "",
                "year": item.get("year") or "",
                "type": item.get("type") or "movie",
                "poster": item.get("poster") or "",
                "origin": item.get("origin") or "",
                "library_name": item.get("library_name") or "",
                "media_source": item.get("media_source") or "",
                "media_id": item.get("media_id") or "",
                "tmdbid": item.get("tmdbid") or "",
                "doubanid": item.get("doubanid") or "",
                "files": [],
            }
            order.append(key)
        groups[key]["files"].append(item)
        if not groups[key]["poster"] and item.get("poster"):
            groups[key]["poster"] = item["poster"]
    rows = []
    for key in order:
        group = groups[key]
        group["file_count"] = len(group["files"])
        rows.append(group)
    return rows
