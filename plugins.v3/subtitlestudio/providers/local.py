"""本地外挂扫描。搜到可用外挂就不会 ASR。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def list_sidecars(media_path: str) -> List[Dict[str, str]]:
    path = Path(media_path)
    if not path.parent.is_dir():
        return []
    rows = []
    for item in sorted(path.parent.iterdir()):
        if item.suffix.lower() not in {".srt", ".ass", ".ssa", ".vtt", ".webvtt"}:
            continue
        if path.stem.lower() not in item.name.lower():
            continue
        rows.append({"path": str(item), "filename": item.name, "ext": item.suffix.lower().lstrip(".")})
    return rows
