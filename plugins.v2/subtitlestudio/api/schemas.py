"""插件 API 的业务模型。不 import app，由宿主决定套 Response 还是本地信封。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

try:
    from pydantic import BaseModel, Field
    try:
        from pydantic import ConfigDict

        class _Extra(BaseModel):
            model_config = ConfigDict(extra="allow")
    except Exception:  # noqa: BLE001 — Pydantic v1
        class _Extra(BaseModel):
            class Config:
                extra = "allow"

    class StatusData(_Extra):
        enabled: bool
        generation: str = ""
        version: str = ""
        counts: Dict[str, int] = Field(default_factory=dict)

    class ConfigData(_Extra):
        enabled: bool = False

    class FieldsData(_Extra):
        fields: List[Dict[str, Any]] = Field(default_factory=list)
        panes: List[Any] = Field(default_factory=list)
        defaults: Dict[str, Any] = Field(default_factory=dict)

    class MediaListData(_Extra):
        groups: List[Dict[str, Any]] = Field(default_factory=list)
        counts: Dict[str, int] = Field(default_factory=dict)
        source: str = "transfer_history"

    class JobData(_Extra):
        job_id: str = ""
        title: str = ""
        path: str = ""
        status: str = ""

    class JobListData(_Extra):
        items: List[Dict[str, Any]] = Field(default_factory=list)
        counts: Dict[str, int] = Field(default_factory=dict)

    class JobBatchData(_Extra):
        items: List[Dict[str, Any]] = Field(default_factory=list)
        skipped: List[Dict[str, Any]] = Field(default_factory=list)
        count: int = 0

    class CueGraphData(_Extra):
        job_id: str = ""
        cues: List[Dict[str, Any]] = Field(default_factory=list)
        notes: List[Dict[str, Any]] = Field(default_factory=list)
        briefing: List[Any] = Field(default_factory=list)

    class CueData(_Extra):
        cue_id: str = ""

    class ExportData(_Extra):
        items: List[Dict[str, Any]] = Field(default_factory=list)

    class SearchData(_Extra):
        items: List[Dict[str, Any]] = Field(default_factory=list)

    class EndpointProbeData(_Extra):
        ok: bool = False
        payload: Any = None
        items: List[str] = Field(default_factory=list)

    HAS_PYDANTIC = True
except Exception:  # noqa: BLE001 — 单测环境可能没有 pydantic
    HAS_PYDANTIC = False
    BaseModel = object  # type: ignore[misc,assignment]

    class StatusData:  # type: ignore[no-redef]
        pass

    class ConfigData:  # type: ignore[no-redef]
        pass

    class FieldsData:  # type: ignore[no-redef]
        pass

    class MediaListData:  # type: ignore[no-redef]
        pass

    class JobData:  # type: ignore[no-redef]
        pass

    class JobListData:  # type: ignore[no-redef]
        pass

    class JobBatchData:  # type: ignore[no-redef]
        pass

    class CueGraphData:  # type: ignore[no-redef]
        pass

    class CueData:  # type: ignore[no-redef]
        pass

    class ExportData:  # type: ignore[no-redef]
        pass

    class SearchData:  # type: ignore[no-redef]
        pass

    class EndpointProbeData:  # type: ignore[no-redef]
        pass


_PAYLOADS = {
    ("/status", "GET"): StatusData,
    ("/config", "GET"): ConfigData,
    ("/config", "POST"): ConfigData,
    ("/fields", "GET"): FieldsData,
    ("/media", "GET"): MediaListData,
    ("/media/refresh", "POST"): MediaListData,
    ("/jobs", "GET"): JobListData,
    ("/jobs", "POST"): JobData,
    ("/jobs/batch", "POST"): JobBatchData,
    ("/jobs/{job_id}", "GET"): JobData,
    ("/jobs/{job_id}/cut-in", "POST"): JobData,
    ("/jobs/{job_id}/priority", "POST"): JobData,
    ("/jobs/{job_id}/cancel", "POST"): JobData,
    ("/jobs/{job_id}/retry", "POST"): JobData,
    ("/jobs/{job_id}/cues", "GET"): CueGraphData,
    ("/jobs/{job_id}/cues/{cue_id}", "PUT"): CueData,
    ("/jobs/{job_id}/export", "POST"): ExportData,
    ("/jobs/{job_id}/search", "POST"): SearchData,
    ("/endpoints/test", "POST"): EndpointProbeData,
    ("/endpoints/models", "POST"): EndpointProbeData,
}


def payload_model_for(path: str, method: str) -> Optional[Type]:
    return _PAYLOADS.get((path, method.upper()))


def make_envelope(data_cls: Type) -> Optional[Type]:
    if not HAS_PYDANTIC or BaseModel is object:
        return data_cls
    name = f"{getattr(data_cls, '__name__', 'Payload')}Envelope"

    class Envelope(BaseModel):
        success: bool
        message: str = ""
        data: Optional[data_cls] = None

    Envelope.__name__ = name
    Envelope.__qualname__ = name
    return Envelope
