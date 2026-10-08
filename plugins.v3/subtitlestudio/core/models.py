"""字幕工坊运行时模型。

CueGraph 是工作台和导出的唯一真源：改句、挂载预览、写盘都读它，
不要再各写一份 SRT 字符串当状态。Job 落自有库，禁止只靠内存队列。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


JOB_STATUSES = (
    "pending",
    "running",
    "success",
    "failed",
    "skipped",
    "cancelled",
)

PRIORITIES = ("P0", "P1", "P2")
PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}

TRIGGERS = ("event", "watch", "strm", "manual", "workflow", "command")

STRATEGIES = {
    "search_then_translate": "先搜后译",
    "search_only": "只搜索",
    "translate_only": "只识别翻译",
}

LAYOUTS = {
    "mono": "单语",
    "stacked": "叠行",
    "split": "分轨",
}

FORMATS = ("srt", "ass", "vtt")

LANG_STACKS = {
    "main_bottom": "主下小上",
    "main_top": "主上小下",
}

# 顺序即字号。第 1 位 100%，第 2 位约 78%，第 3 位约 64%。
LANG_FONT_SIZES = (22, 17, 14)
LANG_SIZE_RATIOS = (1.0, 0.78, 0.64)


@dataclass
class Issue:
    """工作台可见的问题。格式修复可以记低置信，但不偷偷改语义。"""

    code: str
    message: str
    cue_id: str = ""
    severity: str = "warn"
    confirmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Issue":
        return cls(
            code=str(data.get("code") or ""),
            message=str(data.get("message") or ""),
            cue_id=str(data.get("cue_id") or ""),
            severity=str(data.get("severity") or "warn"),
            confirmed=bool(data.get("confirmed")),
        )


@dataclass
class Cue:
    """一句字幕。texts 按语种存，不要把叠行提前压成一个字符串。"""

    cue_id: str
    index: int
    start_ms: int
    end_ms: int
    texts: Dict[str, str] = field(default_factory=dict)
    speaker: str = ""
    kind: str = "dialogue"  # dialogue / note / sdh / asr
    issues: List[Issue] = field(default_factory=list)

    def duration_ms(self) -> int:
        return max(0, int(self.end_ms) - int(self.start_ms))

    def text(self, lang: str = "") -> str:
        if lang:
            return str(self.texts.get(lang) or "")
        if not self.texts:
            return ""
        return next(iter(self.texts.values()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cue_id": self.cue_id,
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "texts": dict(self.texts),
            "speaker": self.speaker,
            "kind": self.kind,
            "issues": [item.to_dict() for item in self.issues],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cue":
        return cls(
            cue_id=str(data.get("cue_id") or ""),
            index=int(data.get("index") or 0),
            start_ms=int(data.get("start_ms") or 0),
            end_ms=int(data.get("end_ms") or 0),
            texts={str(key): str(value) for key, value in dict(data.get("texts") or {}).items()},
            speaker=str(data.get("speaker") or ""),
            kind=str(data.get("kind") or "dialogue"),
            issues=[Issue.from_dict(item) for item in data.get("issues") or [] if isinstance(item, dict)],
        )


@dataclass
class CueGraph:
    """当前任务的字幕图。预览 ASS 必须由它现场打包，不读已落盘旧文件。"""

    job_id: str
    source_lang: str = "en"
    cues: List[Cue] = field(default_factory=list)
    notes: List[Cue] = field(default_factory=list)
    briefing: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: int = 0

    def all_cues(self) -> List[Cue]:
        return list(self.cues) + list(self.notes)

    def sorted_cues(self) -> List[Cue]:
        return sorted(self.cues, key=lambda item: (item.start_ms, item.index, item.cue_id))

    def at_time(self, position_ms: int) -> List[Cue]:
        return [
            cue
            for cue in self.all_cues()
            if cue.start_ms <= position_ms < cue.end_ms
        ]

    def replace_cue(self, cue_id: str, **changes: Any) -> Optional[Cue]:
        for bucket in (self.cues, self.notes):
            for index, cue in enumerate(bucket):
                if cue.cue_id != cue_id:
                    continue
                payload = cue.to_dict()
                payload.update(changes)
                bucket[index] = Cue.from_dict(payload)
                return bucket[index]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_lang": self.source_lang,
            "cues": [item.to_dict() for item in self.cues],
            "notes": [item.to_dict() for item in self.notes],
            "briefing": list(self.briefing),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CueGraph":
        return cls(
            job_id=str(data.get("job_id") or ""),
            source_lang=str(data.get("source_lang") or "en"),
            cues=[Cue.from_dict(item) for item in data.get("cues") or [] if isinstance(item, dict)],
            notes=[Cue.from_dict(item) for item in data.get("notes") or [] if isinstance(item, dict)],
            briefing=[item for item in data.get("briefing") or [] if isinstance(item, dict)],
            duration_ms=int(data.get("duration_ms") or 0),
        )


@dataclass
class Job:
    """持久化任务。插队只改未运行的：升 P0 并排到 pending 队首，不打断 running。"""

    job_id: str
    title: str
    path: str
    media_source: str = ""
    media_id: str = ""
    tmdbid: str = ""
    doubanid: str = ""
    priority: str = "P1"
    status: str = "pending"
    trigger: str = "manual"
    strategy: str = "search_then_translate"
    is_strm: bool = False
    error: str = ""
    created_at: str = ""
    updated_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    queue_rank: int = 0
    plugin_id: str = "SubtitleStudio"
    payload: Dict[str, Any] = field(default_factory=dict)

    def identity(self) -> Dict[str, str]:
        return {
            "media_source": self.media_source,
            "media_id": self.media_id,
            "tmdbid": self.tmdbid,
            "doubanid": self.doubanid,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["is_strm"] = bool(self.is_strm)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        return cls(
            job_id=str(data.get("job_id") or ""),
            title=str(data.get("title") or ""),
            path=str(data.get("path") or ""),
            media_source=str(data.get("media_source") or ""),
            media_id=str(data.get("media_id") or ""),
            tmdbid=str(data.get("tmdbid") or ""),
            doubanid=str(data.get("doubanid") or ""),
            priority=str(data.get("priority") or "P1"),
            status=str(data.get("status") or "pending"),
            trigger=str(data.get("trigger") or "manual"),
            strategy=str(data.get("strategy") or "search_then_translate"),
            is_strm=bool(data.get("is_strm")),
            error=str(data.get("error") or ""),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            started_at=str(data.get("started_at") or ""),
            finished_at=str(data.get("finished_at") or ""),
            queue_rank=int(data.get("queue_rank") or 0),
            plugin_id=str(data.get("plugin_id") or "SubtitleStudio"),
            payload=dict(data.get("payload") or {}),
        )


@dataclass
class Endpoint:
    """OpenAI 兼容线路。Key 只存在插件配置，页面展示要脱敏。"""

    endpoint_id: str
    name: str = ""
    api_url: str = ""
    api_key: str = ""
    model: str = ""
    enabled: bool = True
    primary: bool = False
    use_proxy: bool = False
    compatible: bool = False

    def to_dict(self, mask_key: bool = False) -> Dict[str, Any]:
        key = self.api_key
        if mask_key and key:
            key = key[:4] + "****" + key[-2:] if len(key) > 8 else "****"
        return {
            "endpoint_id": self.endpoint_id,
            "name": self.name,
            "api_url": self.api_url,
            "api_key": key,
            "model": self.model,
            "enabled": self.enabled,
            "primary": self.primary,
            "use_proxy": self.use_proxy,
            "compatible": self.compatible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Endpoint":
        return cls(
            endpoint_id=str(data.get("endpoint_id") or data.get("id") or ""),
            name=str(data.get("name") or ""),
            api_url=str(data.get("api_url") or data.get("url") or ""),
            api_key=str(data.get("api_key") or data.get("key") or ""),
            model=str(data.get("model") or ""),
            enabled=bool(data.get("enabled", True)),
            primary=bool(data.get("primary") or data.get("is_primary")),
            use_proxy=bool(data.get("use_proxy")),
            compatible=bool(data.get("compatible")),
        )
