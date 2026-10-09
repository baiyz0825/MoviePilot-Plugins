from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _class_name_and_version(path: Path) -> tuple[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    plugin = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    version = next(
        node.value.value
        for node in plugin.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "plugin_version" for target in node.targets)
        and isinstance(node.value, ast.Constant)
    )
    return plugin.name, version


def test_v3_subtitlestudio_keeps_stable_identity_and_version():
    assert _class_name_and_version(ROOT / "plugins.v3/subtitlestudio/__init__.py") == ("SubtitleStudio", "2.0.2")


def test_v3_index_and_v2_opt_out_are_consistent():
    package_v3 = json.loads((ROOT / "package.v3.json").read_text(encoding="utf-8"))
    package_v2 = json.loads((ROOT / "package.v2.json").read_text(encoding="utf-8"))
    assert set(package_v3) == {"SubtitleStudio"}
    assert set(package_v2) == {"SubtitleStudio"}
    package_v1 = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    assert package_v1 == {}
    assert "AutoSubv3" not in package_v3
    assert "SubtitleManualUpload" not in package_v3
    assert "MediaCoverGenerator" not in package_v2
    assert "MediaCoverGenerator" not in package_v1
    assert package_v3["SubtitleStudio"]["version"] == "2.0.2"
    assert package_v3["SubtitleStudio"]["system_version"] == ">=3.0.0"
    assert "v2.0.2" in package_v3["SubtitleStudio"]["history"]
    assert "v2.0.1" in package_v3["SubtitleStudio"]["history"]
    assert "v2.0.0" in package_v3["SubtitleStudio"]["history"]
    assert package_v2["SubtitleStudio"]["v3"] is False


def test_v3_subtitlestudio_uses_sdk_imports():
    source = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in (ROOT / "plugins.v3/subtitlestudio").rglob("*.py")
        if "node_modules" not in path.parts and "dist" not in path.parts
    )
    assert "from app.sdk.plugin import _PluginBase" in source or "app.sdk.plugin" in source
    assert "from app.core.event import eventmanager" not in source
    assert "from app.log import logger" not in source
    assert "from app.core.plugin" not in source
    assert "SessionFactory" not in source
    assert "from app.db.models" not in source
