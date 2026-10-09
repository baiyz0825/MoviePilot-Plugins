"""媒体目录。优先拉 MoviePilot 整理记录，再补监控目录。

没有「直接枚举 Emby/Jellyfin 媒体库」的官方插件接口。
本地已入库片子走 TransferHistory，和海拉鲁字幕大师同一条源。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..core.history import expand_history_rows, group_media_items
from ..core.identity import media_item_id
from ..core.logging import studio_log
from ..core.paths import parse_multiline_paths
from ..ingest.gates import is_strm_path, is_video_path
from .local import list_sidecars


class MediaCatalog:
    def __init__(
        self,
        config_getter: Callable[[], Dict[str, Any]],
        *,
        history_loader: Optional[Callable[..., Iterable[Dict[str, Any]]]] = None,
        library_roots: Optional[Callable[[], List[str]]] = None,
        logger: Any = None,
    ):
        self.config_getter = config_getter
        self.history_loader = history_loader
        self.library_roots = library_roots
        self.logger = logger
        self._cache: Optional[List[Dict[str, Any]]] = None

    def invalidate(self) -> None:
        self._cache = None

    def list_media(self, *, q: str = "", media_type: str = "", force: bool = False) -> List[Dict[str, Any]]:
        if force:
            self.invalidate()
        if self._cache is None:
            self._cache = self._collect()
        rows = list(self._cache)
        query = (q or "").strip().lower()
        if query:
            rows = [
                item for item in rows
                if query in (item.get("title") or "").lower()
                or query in (item.get("path") or "").lower()
                or query in (item.get("filename") or "").lower()
            ]
        if media_type in {"movie", "tv"}:
            rows = [item for item in rows if item.get("type") == media_type]
        rows.sort(key=lambda item: (item.get("title") or "", item.get("season") or 0, item.get("episode") or 0, item.get("path") or ""))
        return rows

    def list_groups(self, *, q: str = "", media_type: str = "", force: bool = False) -> List[Dict[str, Any]]:
        return group_media_items(self.list_media(q=q, media_type=media_type, force=force))

    def _collect(self) -> List[Dict[str, Any]]:
        config = self.config_getter() or {}
        items: List[Dict[str, Any]] = []
        history_items = self._from_history(config)
        items.extend(history_items)
        # 目录扫描只补监控路径，不要默认把整个 LIBRARY_PATHS rglob 一遍。
        if config.get("ingest_on_watch") or parse_multiline_paths(config.get("watch_paths")):
            items.extend(self._from_disk(config))
        if config.get("strm_enabled"):
            items.extend(self._from_strm(config))
        deduped = {}
        for item in items:
            deduped[item["id"]] = item
        rows = list(deduped.values())
        studio_log(
            self.logger,
            "info",
            "拉取媒体库 整理记录=%s 合计文件=%s",
            len(history_items),
            len(rows),
        )
        return rows

    def _from_history(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.history_loader:
            return []
        try:
            try:
                raw = list(self.history_loader(limit=800) or [])
            except TypeError:
                raw = list(self.history_loader() or [])
        except Exception as exc:  # noqa: BLE001
            studio_log(self.logger, "warning", "读取整理记录失败：%s", exc)
            return []
        rows = []
        for entry in expand_history_rows(raw):
            path = entry.get("path") or ""
            if not path:
                continue
            if not config.get("trust_transfer_history") and not Path(path).exists():
                continue
            rows.append(self._item(path, entry.get("title") or Path(path).stem, entry, extra=entry))
        return rows

    def _from_disk(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        roots = parse_multiline_paths(config.get("watch_paths"))
        rows = []
        for root in roots:
            base = Path(root)
            if not base.is_dir():
                continue
            for item in base.rglob("*"):
                if not item.is_file() or not is_video_path(str(item)):
                    continue
                rows.append(self._item(str(item), item.stem, {}, {"origin": "watch", "library_name": "监控目录", "filename": item.name}))
        return rows

    def _from_strm(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        for root in parse_multiline_paths(config.get("strm_paths")):
            base = Path(root)
            if not base.is_dir():
                continue
            for item in base.rglob("*.strm"):
                rows.append(self._item(str(item), item.stem, {}, {"origin": "strm", "library_name": "STRM 目录", "filename": item.name, "type": "movie"}))
        return rows

    def _item(self, path: str, title: str, identity: Dict[str, str], extra: Dict[str, Any]) -> Dict[str, Any]:
        identity = identity or {}
        media_type = extra.get("type") or extra.get("media_type") or ("tv" if extra.get("season") else "movie")
        return {
            "id": media_item_id(identity, path),
            "title": title,
            "year": extra.get("year") or "",
            "path": path,
            "filename": extra.get("filename") or Path(path).name,
            "type": media_type,
            "season": extra.get("season"),
            "episode": extra.get("episode"),
            "poster": extra.get("poster") or extra.get("image") or "",
            "is_strm": is_strm_path(path),
            "sidecars": list_sidecars(path),
            "origin": extra.get("origin") or "transfer_history",
            "library_name": extra.get("library_name") or "MoviePilot 整理记录",
            "media_key": extra.get("media_key") or media_item_id(identity, title),
            "date": extra.get("date") or "",
            "media_source": identity.get("media_source") or "",
            "media_id": identity.get("media_id") or "",
            "tmdbid": identity.get("tmdbid") or "",
            "doubanid": identity.get("doubanid") or "",
        }
