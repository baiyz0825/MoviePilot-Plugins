"""同一身份 + 路径 5 分钟内只建一次任务。

两路自动入队都开时靠这个避免双入队。事件回调里只记键，真正搜站放到 3 秒防抖任务里。
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Tuple

from ..core.identity import debounce_key


class IngestDebouncer:
    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self._seen: Dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(self, identity: dict, path: str, *, now: float | None = None) -> bool:
        key = debounce_key(identity, path)
        if not key:
            return False
        stamp = time.time() if now is None else now
        with self._lock:
            last = self._seen.get(key, 0)
            if stamp - last < self.window_seconds:
                return False
            self._seen[key] = stamp
            self._prune(stamp)
            return True

    def _prune(self, now: float) -> None:
        expired = [key for key, value in self._seen.items() if now - value > self.window_seconds * 4]
        for key in expired:
            self._seen.pop(key, None)
