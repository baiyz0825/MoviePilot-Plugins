"""媒体目录。优先用宿主注入的整理历史，其次扫监控目录。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..core.identity import identity_from_v2_ids, media_item_id
from ..core.paths import parse_multiline_paths
from ..ingest.gates import VIDEO_EXTS, is_strm_path, is_video_path
from .local import list_sidecars


class MediaCatalog:
    def __init__(
        self,
        config_getter: Callable[[], Dict[str, Any]],
        *,
        history_loader: Optional[Callable[[], Iterable[Dict[str, Any]]]] = None,
        library_roots: Optional[Callable[[], List[str]]] = None,
    ):
        self.config_getter = config_getter
        self.history_loader = history_loader
        self.library_roots = library_roots

    def list_media(self, *, q: str = "", media_type: str = "") -> List[Dict[str, Any]]:
        config = self.config_getter() or {}
        items = []
        if config.get("trust_transfer_history") and self.history_loader:
            items.extend(self._from_history(config))
        else:
            items.extend(self._from_history(config))
            items.extend(self._from_disk(config))
        deduped = {}
        for item in items:
            deduped[item["id"]] = item
        rows = list(deduped.values())
        query = (q or "").strip().lower()
        if query:
            rows = [item for item in rows if query in (item.get("title") or "").lower() or query in (item.get("path") or "").lower()]
        if media_type in {"movie", "tv"}:
            rows = [item for item in rows if item.get("type") == media_type]
        rows.sort(key=lambda item: item.get("title") or item.get("path") or "")
        return rows

    def _from_history(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.history_loader:
            return []
        rows = []
        try:
            entries = list(self.history_loader() or [])
        except Exception:
            return []
        for entry in entries:
            path = str(entry.get("path") or entry.get("dest") or entry.get("dest_file") or "")
            if not path or not is_video_path(path):
                continue
            if not config.get("trust_transfer_history") and not Path(path).exists():
                continue
            identity = {
                "media_source": str(entry.get("media_source") or ""),
                "media_id": str(entry.get("media_id") or ""),
                "tmdbid": str(entry.get("tmdbid") or ""),
                "doubanid": str(entry.get("doubanid") or ""),
            }
            if not identity["media_source"]:
                identity.update(identity_from_v2_ids(identity.get("tmdbid"), identity.get("doubanid")))
            rows.append(self._item(path, entry.get("title") or Path(path).stem, identity, entry))
        return rows

    def _from_disk(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        roots = parse_multiline_paths(config.get("watch_paths"))
        if not roots and self.library_roots:
            try:
                roots = list(self.library_roots() or [])
            except Exception:
                roots = []
        rows = []
        for root in roots:
            base = Path(root)
            if not base.is_dir():
                continue
            for item in base.rglob("*"):
                if not item.is_file() or not is_video_path(str(item)):
                    continue
                rows.append(self._item(str(item), item.stem, {}, {"origin": "watch"}))
        return rows

    def _item(self, path: str, title: str, identity: Dict[str, str], extra: Dict[str, Any]) -> Dict[str, Any]:
        identity = identity or {}
        return {
            "id": media_item_id(identity, path),
            "title": title,
            "path": path,
            "type": extra.get("type") or extra.get("media_type") or ("tv" if extra.get("season") else "movie"),
            "season": extra.get("season"),
            "episode": extra.get("episode"),
            "poster": extra.get("poster") or "",
            "is_strm": is_strm_path(path),
            "sidecars": list_sidecars(path),
            **identity,
        }
