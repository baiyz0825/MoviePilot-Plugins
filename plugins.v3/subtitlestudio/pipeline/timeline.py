"""用音频语音活动检查并保守修正已有字幕的整轨偏移。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

from ..core.cuegraph import parse_srt, render_srt
from ..core.logging import studio_log
from ..core.models import CueGraph


def align_to_audio(graph: CueGraph, media_path: str, data_path: Path, config: Dict[str, Any], logger: Any = None) -> CueGraph:
    if not graph.cues or not Path(media_path).is_file() or not config.get("timeline_auto_align", True):
        return graph
    try:
        import ffsubsync
        from ffsubsync.ffsubsync import make_parser
    except ImportError:
        studio_log(logger, "info", "自动调轴依赖不可用，保留原时间轴")
        return graph
    source = graph.source_lang or "source"
    original = graph.sorted_cues()
    with tempfile.TemporaryDirectory(prefix="ss-align-", dir=str(data_path)) as temp:
        input_path = Path(temp) / "source.srt"
        output_path = Path(temp) / "aligned.srt"
        input_path.write_text(render_srt(graph, source, fallback_lang="source"), encoding="utf-8")
        requested_offset = float(config.get("timeline_max_offset") or 30)
        max_offset = requested_offset if config.get("timeline_allow_risky") else min(30.0, requested_offset)
        args = make_parser().parse_args([
            media_path, "-i", str(input_path), "-o", str(output_path),
            "--skip-sync-on-low-quality", "--quality-max-offset-seconds", str(max_offset),
            "--no-fix-framerate", "--vad", "webrtc",
        ])
        try:
            result = ffsubsync.run(args)
        except Exception as exc:  # noqa: BLE001
            studio_log(logger, "warning", "音频调轴失败，保留原时间轴：%s", exc)
            return graph
        if result.get("retval") != 0 or not result.get("sync_was_successful") or not output_path.is_file():
            studio_log(logger, "info", "音频调轴质量不足，保留原时间轴")
            return graph
        offset = float(result.get("offset_seconds") or 0)
        if abs(offset) < max(0.2, float(config.get("timeline_min_offset") or 0.2)) or abs(offset) > max_offset:
            return graph
        aligned = parse_srt(output_path.read_text(encoding="utf-8"), lang=source, job_id=graph.job_id).sorted_cues()
        if len(aligned) != len(original) or any(a.text() != b.text(source) and a.text() != b.text() for a, b in zip(aligned, original)):
            studio_log(logger, "warning", "调轴输出与源字幕句数或内容不一致，保留原时间轴")
            return graph
        for cue, timed in zip(original, aligned):
            cue.start_ms, cue.end_ms = timed.start_ms, timed.end_ms
        graph.duration_ms = max((cue.end_ms for cue in graph.cues), default=graph.duration_ms)
        studio_log(logger, "info", "音频调轴完成 offset=%.3f 秒 cues=%s", offset, len(original))
        return graph
