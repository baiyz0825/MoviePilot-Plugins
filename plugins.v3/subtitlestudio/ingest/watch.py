"""目录监控与 STRM 监控。默认关，和 TransferComplete 互不影响。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

from ..core.paths import parse_multiline_paths
from ..ingest.gates import is_strm_path, is_video_path


class WatchService:
    def __init__(self, on_file: Callable[[str, str], None], logger: Any = None):
        self.on_file = on_file
        self.logger = logger
        self._observer = None

    def start(self, paths: List[str], *, strm: bool = False) -> None:
        self.stop()
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except Exception as exc:  # noqa: BLE001
            if self.logger:
                self.logger.warning("[SubtitleStudio] 未安装 watchdog，目录监控不可用：%s", exc)
            return

        owner = self

        class Handler(FileSystemEventHandler):
            def on_created(self, event):
                owner._emit(event.src_path, strm)

            def on_modified(self, event):
                owner._emit(event.src_path, strm)

        observer = Observer()
        handler = Handler()
        started = False
        for item in paths:
            path = Path(item)
            if not path.is_dir():
                continue
            observer.schedule(handler, str(path), recursive=True)
            started = True
        if not started:
            return
        observer.daemon = True
        observer.start()
        self._observer = observer

    def stop(self) -> None:
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=2)
            except Exception:
                pass
            self._observer = None

    def _emit(self, path: str, strm: bool) -> None:
        if strm and not is_strm_path(path):
            return
        if not strm and not is_video_path(path):
            return
        try:
            self.on_file(path, "strm" if strm else "watch")
        except Exception as exc:  # noqa: BLE001
            if self.logger:
                self.logger.error("[SubtitleStudio] 监控回调失败：%s", exc)


def watch_paths_from_config(config: dict) -> List[str]:
    return parse_multiline_paths(config.get("watch_paths"))


def strm_paths_from_config(config: dict) -> List[str]:
    return parse_multiline_paths(config.get("strm_paths"))
