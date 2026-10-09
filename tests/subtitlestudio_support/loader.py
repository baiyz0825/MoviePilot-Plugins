from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def plugin_root(generation: str) -> Path:
    return ROOT / f"plugins.{generation}" / "subtitlestudio"


def _ensure_pkg(name: str, path: Path | None = None) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    if path is not None:
        module.__path__ = [str(path)]
        module.__file__ = str(path / "__init__.py")
    sys.modules[name] = module
    parent_name, _, child = name.rpartition(".")
    if parent_name:
        setattr(_ensure_pkg(parent_name), child, module)
    return module


def domain_package(generation: str) -> str:
    """把对应代的插件树挂到私有包名下，避免 `import subtitlestudio`。"""
    name = f"_subtitlestudio_{generation}"
    if name in sys.modules and getattr(sys.modules[name], "__path__", None):
        return name
    root = plugin_root(generation)
    _ensure_pkg(name, root)
    for child in ("core", "packager", "storage", "ingest", "pipeline", "providers", "api", "automation"):
        _ensure_pkg(f"{name}.{child}", root / child)
    return name


def load_domain(generation: str, dotted: str):
    pkg = domain_package(generation)
    return importlib.import_module(f"{pkg}.{dotted}")


def install_app_stubs(generation: str = "v2") -> None:
    if "app" not in sys.modules:
        sys.modules["app"] = types.ModuleType("app")

    class PluginBase:
        def get_data_path(self):
            return getattr(self, "_data_path", Path("."))

        def get_config(self):
            return getattr(self, "_config", {})

        def update_config(self, config):
            self._config = config

        def get_data(self, key):
            return getattr(self, "_data", {}).get(key)

        def save_data(self, key, value):
            self._data = getattr(self, "_data", {})
            self._data[key] = value

        def post_message(self, **kwargs):
            return None

        def init_plugin(self, config=None):
            return None

    def ensure(name: str, **attrs):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        for key, value in attrs.items():
            setattr(module, key, value)
        parent_name, _, child = name.rpartition(".")
        if parent_name:
            parent = ensure(parent_name)
            setattr(parent, child, module)
        return module

    class EventManager:
        @staticmethod
        def register(_event_type):
            def decorator(func):
                return func
            return decorator

        @staticmethod
        def add_event_listener(*_a, **_k):
            return None

        @staticmethod
        def remove_event_listener(*_a, **_k):
            return None

    class EventType:
        TransferComplete = "transfer.complete"
        PluginAction = "plugin.action"

    class NotificationType:
        Plugin = "Plugin"

    ensure("app.plugins", _PluginBase=PluginBase)
    ensure("app.sdk.plugin", _PluginBase=PluginBase)
    ensure("app.core.config", settings=types.SimpleNamespace(RMT_MEDIAEXT=[".mp4", ".mkv", ".strm"], PROXY=None))
    ensure("app.sdk.config", settings=types.SimpleNamespace(RMT_MEDIAEXT=[".mp4", ".mkv", ".strm"], PROXY=None))
    ensure("app.log", logger=types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None))
    ensure("app.sdk.logging", logger=types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None))
    ensure("app.core.event", eventmanager=EventManager(), Event=object)
    ensure("app.sdk.events", eventmanager=EventManager(), Event=object, EventType=EventType)
    ensure("app.schemas.types", EventType=EventType, NotificationType=NotificationType)
    ensure("app.sdk.schema", NotificationType=NotificationType)
    ensure("app.sdk.schemas", NotificationType=NotificationType)
    ensure("app.schemas", Response=dict)
    ensure("app.core.plugin", PluginManager=None)
    ensure("app.sdk.plugins", PluginManager=None)
    ensure("app.core.cache")
    ensure("app.sdk.cache")
    ensure("app.scheduler")
    ensure("app.sdk.scheduler")
    ensure("app.db.models.transferhistory", TransferHistory=object)
    ensure("app.db.oper.transferhistory", TransferHistoryOper=object)
    ensure("app.db.oper.transfer", TransferHistoryOper=object)
    ensure("app.agent.tools.base")


def load_plugin_package(generation: str):
    install_app_stubs(generation)
    root = plugin_root(generation)
    name = "app.plugins.subtitlestudio"
    for key in list(sys.modules):
        if key == name or key.startswith(f"{name}."):
            del sys.modules[key]
    _ensure_pkg("app")
    _ensure_pkg("app.plugins")
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module
