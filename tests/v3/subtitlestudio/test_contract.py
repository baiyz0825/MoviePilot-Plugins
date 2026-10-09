from __future__ import annotations

import json
from pathlib import Path

from tests.subtitlestudio_support.loader import ROOT, load_plugin_package, plugin_root

GEN = "v3"


def _source(rel: str) -> str:
    return (plugin_root(GEN) / rel).read_text(encoding="utf-8-sig")


def test_v3_index_and_class_version_match():
    package = json.loads((ROOT / "package.v3.json").read_text(encoding="utf-8"))
    plugin_pkg = json.loads((plugin_root(GEN) / "package.json").read_text(encoding="utf-8"))
    meta = package["SubtitleStudio"]
    assert meta["version"] == "2.0.0"
    assert meta["system_version"] == ">=3.0.0"
    assert meta["release"] is True
    assert "v2.0.0" in meta["history"]
    assert plugin_pkg["version"] == "2.0.0"
    init = _source("__init__.py")
    assert 'plugin_version = "2.0.0"' in init
    assert "app.sdk.plugin" in init
    assert "from app.core.event" not in init
    assert "from app.log import logger" not in init


def test_v3_host_adapter_uses_sdk():
    host = _source("host.py")
    imports = [line.strip() for line in host.splitlines() if line.strip().startswith(("import ", "from "))]
    assert "from app.sdk.config import settings" in host
    assert "from app.sdk.logging import logger" in host
    assert "app.sdk.events" in host
    assert "add_plugin_once_job" in host
    assert "identity_from_v3_pair" in host
    assert any("httpx2" in line for line in imports)
    assert not any("alias_httpx" in line for line in imports)
    assert "from app.core.event" not in imports
    assert "from app.log import logger" not in imports
    assert "from app.core.config import settings" not in imports


def test_v3_does_not_register_events_at_import():
    init = _source("__init__.py")
    assert "@eventmanager.register" not in init
    assert "_bind_events" in init
    assert "register_listener" in init
    assert "listen_transfer_complete" in init
    assert "scheduler.enqueue" in init


def test_v3_requirements_use_httpx2():
    req = _source("requirements.txt")
    assert "httpx2" in req
    assert "httpx>=" not in req
    pyproject = _source("pyproject.toml")
    assert 'version = "2.0.0"' in pyproject


def test_domain_has_no_app_imports():
    import re
    root = plugin_root(GEN)
    offenders = []
    for path in root.rglob("*.py"):
        if path.name in {"host.py"} or "node_modules" in path.parts or path.parent.name == "src":
            continue
        if path.relative_to(root) == Path("__init__.py"):
            continue
        if path.name == "agent_tools.py":
            continue
        text = path.read_text(encoding="utf-8-sig")
        if re.search(r"^(from app\.|import app\.)", text, re.M):
            offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_official_hooks_exist():
    module = load_plugin_package(GEN)
    plugin = module.SubtitleStudio()
    plugin._data_path = Path("/tmp/subtitlestudio-v3-test")
    assert plugin.get_render_mode() == ("vue", "dist/assets")
    _form, defaults = plugin.get_form()
    assert defaults["ingest_on_event"] is True
    assert defaults["target_languages"] == ["zh-Hans"]
    assert plugin.get_command()[0]["cmd"] == "/subtitle_studio_run"
    assert defaults["send_notify"] is False
    assert defaults["notify_on"] == ["success", "failed"]
    paths = [item["path"] for item in plugin.get_api()]
    assert "/jobs" in paths
    assert "/jobs/{job_id}/preview/video" in paths


def test_notify_job_uses_plugin_channel():
    module = load_plugin_package(GEN)
    plugin = module.SubtitleStudio()
    sent = []
    plugin.post_message = lambda **kwargs: sent.append(kwargs)
    from tests.subtitlestudio_support.loader import load_domain
    job = load_domain(GEN, "core.models").Job(
        job_id="n1",
        title="沙丘",
        path="/media/Dune.mkv",
        status="failed",
        trigger="event",
        error="没有可用字幕源",
    )
    plugin.notify_job(job)
    assert sent
    assert sent[0]["title"] == "字幕工坊 · 失败"
    assert "没有可用字幕源" in sent[0]["text"]
    assert sent[0]["mtype"] == "Plugin"
