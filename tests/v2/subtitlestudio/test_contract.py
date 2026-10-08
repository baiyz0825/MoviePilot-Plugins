from __future__ import annotations

import json
from pathlib import Path

from tests.subtitlestudio_support.loader import ROOT, load_plugin_package, plugin_root

GEN = "v2"


def _source(rel: str) -> str:
    return (plugin_root(GEN) / rel).read_text(encoding="utf-8-sig")


def test_v2_index_and_class_version_match():
    package = json.loads((ROOT / "package.v2.json").read_text(encoding="utf-8"))
    plugin_pkg = json.loads((plugin_root(GEN) / "package.json").read_text(encoding="utf-8"))
    meta = package["SubtitleStudio"]
    assert meta["version"] == "1.0.0"
    assert meta["v3"] is False
    assert meta["system_version"] == ">=2.13.5"
    assert meta["release"] is True
    assert "v1.0.0" in meta["history"]
    assert plugin_pkg["version"] == "1.0.0"
    init = _source("__init__.py")
    assert 'plugin_version = "1.0.0"' in init
    assert 'from app.plugins import _PluginBase' in init
    assert "app.sdk" not in init


def test_v2_host_adapter_does_not_import_sdk():
    host = _source("host.py")
    imports = [line.strip() for line in host.splitlines() if line.strip().startswith(("import ", "from "))]
    assert "from app.core.config import settings" in host
    assert "from app.log import logger" in host
    assert "from app.core.event import eventmanager" in host
    assert "DateTrigger" in host
    assert "identity_from_v2_ids" in host
    assert not any("httpx2" in line or "app.sdk" in line or "alias_httpx" in line for line in imports)


def test_v2_transfer_complete_really_enqueues():
    init = _source("__init__.py")
    assert "listen_transfer_complete" in init
    assert "trigger=\"event\"" in init or "trigger='event'" in init
    assert "scheduler.enqueue" in init
    assert "ingest_on_event" in init


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
    plugin._data_path = Path("/tmp/subtitlestudio-v2-test")
    plugin._config = {"enabled": False, "show_sidebar_nav": True}
    assert plugin.get_render_mode() == ("vue", "dist/assets")
    form, defaults = plugin.get_form()
    assert isinstance(form, list)
    assert defaults["ingest_on_event"] is True
    assert defaults["ingest_on_watch"] is False
    assert plugin.get_command()[0]["cmd"] == "/subtitle_studio_run"
    assert any(item["path"] == "/jobs" for item in plugin.get_api())
    assert any(item["path"] == "/jobs/{job_id}/preview/ass" for item in plugin.get_api())
    assert plugin.get_page() == []


def test_sidebar_nav_requires_enabled():
    module = load_plugin_package(GEN)
    plugin = module.SubtitleStudio()
    plugin._enabled = False
    plugin._show_sidebar_nav = True
    assert plugin.get_sidebar_nav() == []
    plugin._enabled = True
    nav = plugin.get_sidebar_nav()
    assert nav[0]["nav_key"] == "main"
    assert nav[0]["title"] == "字幕工坊"
    assert nav[0]["section"] == "organize"
    assert nav[0]["icon"] == "mdi-subtitles-outline"
