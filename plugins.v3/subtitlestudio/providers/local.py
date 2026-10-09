"""本地外挂扫描。搜到可用外挂就不会 ASR。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

SUB_EXTS = {".srt", ".ass", ".ssa", ".vtt", ".webvtt"}


def list_sidecars(media_path: str) -> List[Dict[str, str]]:
    return SidecarIndex().for_path(media_path)


class SidecarIndex:
    """同一目录只 listdir 一次。拉 800 条整理记录时不要每个文件扫一遍盘。"""

    def __init__(self):
        self._dirs: Dict[str, List[Path]] = {}

    def for_path(self, media_path: str) -> List[Dict[str, str]]:
        path = Path(media_path)
        parent = path.parent
        key = str(parent)
        if key not in self._dirs:
            self._dirs[key] = sorted(parent.iterdir()) if parent.is_dir() else []
        stem = path.stem.lower()
        rows = []
        for item in self._dirs[key]:
            if item.suffix.lower() not in SUB_EXTS:
                continue
            if stem not in item.name.lower():
                continue
            rows.append({"path": str(item), "filename": item.name, "ext": item.suffix.lower().lstrip(".")})
        return rows
