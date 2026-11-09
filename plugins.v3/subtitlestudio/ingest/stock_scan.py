"""One-shot scan that queues media missing one or more requested subtitle languages."""

from __future__ import annotations

import copy
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from ..core.cuegraph import parse_subtitle
from ..core.embedded import extract_stream, probe_subtitle_streams
from ..core.language import canonical_language
from ..core.logging import studio_log
from ..core.naming import LANG_CATALOG
from ..ingest.gates import is_strm_path

_LANGUAGE_KEYS = set(LANG_CATALOG)
_SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt"}
_GRAPHICS_CODECS = {"pgssub", "dvd_subtitle", "vobsub", "hdmv_pgs_subtitle", "pgs"}


def _graph_languages(content: str, filename: str, hint: str = "source") -> Set[str]:
    """Use the same CueGraph parser and language inference as normal jobs."""
    if not content:
        return set()
    graph = parse_subtitle(content, filename, lang=hint or "source")
    if not graph.cues:
        return set()
    found = {
        canonical_language(language)
        for cue in graph.cues
        for language, text in (cue.texts or {}).items()
        if text
    }
    if graph.source_lang:
        found.add(canonical_language(graph.source_lang))
    return found & _LANGUAGE_KEYS


def _sidecar_languages(path: str) -> Set[str]:
    try:
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return set()
    try:
        return _graph_languages(content, Path(path).name)
    except Exception:  # noqa: BLE001 — a malformed sidecar should not stop the inventory
        return set()


def available_subtitle_languages(
    item: Dict[str, Any],
    required: Set[str] | None = None,
    logger: Any = None,
) -> Set[str]:
    """Detect sidecar and embedded languages using normal SubtitleStudio parsing."""
    media_path = str(item.get("path") or "")
    available: Set[str] = set()
    for sidecar in item.get("sidecars") or []:
        subtitle_path = str((sidecar or {}).get("path") or "")
        if not subtitle_path or Path(subtitle_path).suffix.lower() not in _SUBTITLE_EXTENSIONS:
            continue
        available.update(_sidecar_languages(subtitle_path))
        if required and required.issubset(available):
            return available

    if is_strm_path(media_path) or not Path(media_path).is_file():
        return available
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is not installed; embedded subtitle languages cannot be checked")
    for stream in probe_subtitle_streams(media_path, logger, include_graphics=True):
        language = canonical_language(stream.get("language") or "")
        if language in _LANGUAGE_KEYS:
            available.add(language)
            if required and required.issubset(available):
                return available
        codec = str(stream.get("codec") or "").lower()
        if codec in _GRAPHICS_CODECS or language != "source":
            continue
        content = extract_stream(media_path, int(stream.get("index") or 0), logger)
        if content:
            try:
                available.update(_graph_languages(
                    content, f"embedded_{stream.get('index') or 0}.srt",
                    str(stream.get("language") or "source"),
                ))
            except Exception:  # noqa: BLE001 — keep checking other tracks
                pass
            if required and required.issubset(available):
                return available
    return available


class StockSubtitleScan:
    """Runs one background inventory pass and enqueues only incomplete target sets."""

    def __init__(self, catalog, scheduler, store, config_getter, logger: Any = None):
        self.catalog = catalog
        self.scheduler = scheduler
        self.store = store
        self.config_getter = config_getter
        self.logger = logger
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: Dict[str, Any] = self._empty_state()

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            "scan_id": "", "status": "idle", "started": False,
            "scanned": 0, "total": 0, "queued": 0, "skipped": 0,
            "failed": 0, "target_languages": [], "current_title": "",
            "started_at": "", "finished_at": "", "error": "",
        }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def start(self) -> Dict[str, Any]:
        with self._lock:
            if self._state.get("status") == "running":
                result = copy.deepcopy(self._state)
                result["started"] = False
                return result
            config = dict(self.config_getter() or {})
            targets = [canonical_language(value) for value in config.get("target_languages") or ["zh-Hans"]]
            targets = list(dict.fromkeys(value for value in targets if value in _LANGUAGE_KEYS)) or ["zh-Hans"]
            self._cancel.clear()
            self._state = {
                **self._empty_state(),
                "scan_id": uuid.uuid4().hex,
                "status": "running",
                "started": True,
                "target_languages": targets,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._thread = threading.Thread(
                target=self._run,
                args=(config, targets),
                name="subtitlestudio-stock-scan",
                daemon=True,
            )
            self._thread.start()
            return copy.deepcopy(self._state)

    def stop(self) -> None:
        self._cancel.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2)

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def _increment(self, key: str) -> None:
        with self._lock:
            self._state[key] = int(self._state.get(key) or 0) + 1

    def _finish(self, status: str, error: str = "") -> None:
        self._update(
            status=status,
            current_title="",
            error=error,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def _run(self, config: Dict[str, Any], targets: List[str]) -> None:
        required = set(targets)
        try:
            items = self.catalog.list_stock_media()
            hidden = self.store.hidden_paths()
            items = [item for item in items if item.get("path") and item.get("path") not in hidden]
            self._update(total=len(items))
            if not items:
                self._finish("completed")
                return

            queue_config = {**config, "skip_existing_chinese": False}
            workers = min(4, len(items))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ss-subtitle-probe") as pool:
                futures = {
                    pool.submit(available_subtitle_languages, item, required, self.logger): item
                    for item in items
                }
                for future in as_completed(futures):
                    if self._cancel.is_set():
                        for pending in futures:
                            pending.cancel()
                        self._finish("cancelled")
                        return
                    item = futures[future]
                    self._update(current_title=str(item.get("title") or item.get("filename") or ""))
                    try:
                        available = future.result()
                        missing_languages = [language for language in targets if language not in available]
                        if not missing_languages:
                            self._increment("skipped")
                        else:
                            identity = {
                                key: str(item.get(key) or "")
                                for key in ("media_source", "media_id", "tmdbid", "doubanid")
                            }
                            job = self.scheduler.enqueue(
                                title=str(item.get("title") or Path(str(item.get("path"))).stem),
                                path=str(item.get("path") or ""),
                                identity=identity,
                                trigger="stock_scan",
                                priority="P1",
                                strategy="standard",
                                config=queue_config,
                                payload={**item, "stock_scan_missing_languages": missing_languages},
                                force=False,
                                recheck_success=True,
                            )
                            if job.status in {"pending", "running"}:
                                self._increment("queued")
                            else:
                                self._increment("skipped")
                    except Exception as exc:  # noqa: BLE001 — continue scanning the rest of the library
                        studio_log(self.logger, "warning", "存量字幕扫描单项失败 title=%s：%s", item.get("title") or "", exc)
                        self._increment("failed")
                    finally:
                        self._increment("scanned")
            if self._cancel.is_set():
                self._finish("cancelled")
            else:
                self._finish("completed")
        except Exception as exc:  # noqa: BLE001
            studio_log(self.logger, "error", "存量字幕扫描失败：%s", exc)
            self._finish("failed", "扫描失败，请查看 MoviePilot 服务日志")
