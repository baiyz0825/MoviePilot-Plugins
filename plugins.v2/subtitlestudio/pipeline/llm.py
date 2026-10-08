"""OpenAI 兼容调用。HTTP 由宿主注入，V2 用 httpx，V3 用 HTTPX2。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..core.models import Endpoint


class LlmRouter:
    def __init__(self, http: Callable[..., Any], config: Dict[str, Any], logger: Any = None):
        self.http = http
        self.config = config
        self.logger = logger

    def endpoints(self) -> List[Endpoint]:
        return [Endpoint.from_dict(item) for item in self.config.get("openai_endpoints") or [] if item]

    def primary(self) -> Optional[Endpoint]:
        rows = [item for item in self.endpoints() if item.enabled]
        return next((item for item in rows if item.primary), rows[0] if rows else None)

    def for_role(self, role: str) -> Optional[Endpoint]:
        bound = str(self.config.get(f"role_{role}") or "")
        rows = {item.endpoint_id: item for item in self.endpoints() if item.enabled}
        if bound and bound in rows:
            return rows[bound]
        return self.primary()

    def complete(self, prompt: str, *, system: str = "", role: str = "translate") -> str:
        ordered = []
        current = self.for_role(role)
        if current:
            ordered.append(current)
        if self.config.get("openai_fallback_enabled", True):
            for item in self.endpoints():
                if item.enabled and item.endpoint_id not in {x.endpoint_id for x in ordered}:
                    ordered.append(item)
        last_error = "没有可用线路"
        retries = max(1, int(self.config.get("max_retries") or 3))
        for endpoint in ordered:
            for _attempt in range(retries):
                try:
                    return self._call(endpoint, prompt, system)
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    if self.logger:
                        self.logger.warning("[SubtitleStudio] 线路 %s 失败：%s", endpoint.name or endpoint.endpoint_id, exc)
        raise RuntimeError(last_error)

    def _call(self, endpoint: Endpoint, prompt: str, system: str) -> str:
        base = endpoint.api_url.rstrip("/")
        url = base if endpoint.compatible or base.endswith("/chat/completions") else f"{base}/v1/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = self.http(
            "POST",
            url,
            json_body={"model": endpoint.model, "messages": messages, "temperature": 0.2},
            headers={"Authorization": f"Bearer {endpoint.api_key}", "Content-Type": "application/json"},
            use_proxy=endpoint.use_proxy,
        )
        if isinstance(payload, dict):
            choices = payload.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                return str(message.get("content") or "")
        raise RuntimeError("线路返回无法解析")
