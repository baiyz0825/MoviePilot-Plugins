"""OpenAI 兼容调用。HTTP 由宿主注入，V2 用 httpx，V3 用 HTTPX2。"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

from ..core.logging import studio_log
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
                    studio_log(self.logger, "warning", "线路 %s 失败（第 %s/%s 次）role=%s：%s", endpoint.name or endpoint.endpoint_id, _attempt + 1, retries, role, exc)
        raise RuntimeError(last_error)

    def _call(self, endpoint: Endpoint, prompt: str, system: str) -> str:
        base = endpoint.api_url.rstrip("/")
        url = base if endpoint.compatible or base.endswith("/chat/completions") else f"{base}/v1/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        request_body = {"model": endpoint.model, "messages": messages, "temperature": 0.2}
        timeout = 600 if endpoint.local_model else 20
        started = time.monotonic()
        studio_log(
            self.logger, "info", "LLM 请求开始 endpoint=%s model=%s timeout=%ss prompt_chars=%s",
            self._safe_url(url), endpoint.model, timeout, len(prompt),
        )
        if self.config.get("llm_debug_logging"):
            debug_request = {
                "method": "POST",
                "url": self._safe_url(url),
                "headers": {"Authorization": "Bearer [REDACTED]", "Content-Type": "application/json"},
                "json": request_body,
            }
            studio_log(self.logger, "info", "LLM 请求调试 request=%s", self._json_for_log(debug_request))
        try:
            payload = self.http(
                "POST",
                url,
                json_body=request_body,
                headers={"Authorization": f"Bearer {endpoint.api_key}", "Content-Type": "application/json"},
                use_proxy=endpoint.use_proxy,
                timeout=timeout,
            )
        except Exception as exc:
            studio_log(
                self.logger, "info", "LLM 请求失败 model=%s elapsed=%.1fs error=%s",
                endpoint.model, time.monotonic() - started, exc,
            )
            raise
        elapsed = time.monotonic() - started
        studio_log(self.logger, "info", "LLM 请求完成 model=%s elapsed=%.1fs", endpoint.model, elapsed)
        if self.config.get("llm_debug_logging"):
            studio_log(self.logger, "info", "LLM 响应调试 model=%s elapsed=%.1fs response=%s", endpoint.model, elapsed, self._json_for_log(payload))
        if isinstance(payload, dict):
            choices = payload.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                return str(message.get("content") or "")
        raise RuntimeError("线路返回无法解析")

    @staticmethod
    def _safe_url(url: str) -> str:
        """日志里移除 URL 用户信息和查询参数，避免泄露 URL 内嵌凭据。"""
        try:
            parts = urlsplit(url)
            host = parts.hostname or ""
            if parts.port:
                host = f"{host}:{parts.port}"
            return urlunsplit((parts.scheme, host, parts.path, "", ""))
        except Exception:  # noqa: BLE001
            return "[URL REDACTED]"

    @staticmethod
    def _json_for_log(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)
