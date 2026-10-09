from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _zip_plugin(plugin_dir: Path, dest: Path) -> None:
    dest.unlink(missing_ok=True)
    subprocess.run(
        [
            "zip",
            "-r",
            str(dest),
            plugin_dir.name,
            "-x",
            "*/node_modules/*",
            "-x",
            "*/__pycache__/*",
            "-x",
            "*.pyc",
            "-x",
            "*/.pytest_cache/*",
        ],
        cwd=plugin_dir.parent,
        check=True,
        capture_output=True,
        text=True,
    )


def _assert_release_zip(plugin_dir: Path, dest: Path) -> None:
    _zip_plugin(plugin_dir, dest)
    names = zipfile.ZipFile(dest).namelist()
    prefix = plugin_dir.name
    assert all(name.startswith(f"{prefix}/") for name in names)
    for rel in (
        f"{prefix}/__init__.py",
        f"{prefix}/api/__init__.py",
        f"{prefix}/api/routes.py",
        f"{prefix}/dist/assets/remoteEntry.js",
    ):
        assert rel in names, rel


def test_v2_release_zip_contains_api_package(tmp_path: Path) -> None:
    _assert_release_zip(ROOT / "plugins.v2" / "subtitlestudio", tmp_path / "subtitlestudio_v2.zip")


def test_v3_release_zip_contains_api_package(tmp_path: Path) -> None:
    _assert_release_zip(ROOT / "plugins.v3" / "subtitlestudio", tmp_path / "subtitlestudio_v3.zip")


def test_release_workflow_runs_packaging_script() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    script = ROOT / ".github/scripts/release_plugins.sh"
    assert script.is_file()
    assert "bash .github/scripts/release_plugins.sh" in workflow
    assert "Diagnose environment" in workflow
    assert "workflow_dispatch" in workflow
    assert "paths:" not in workflow
    assert not (ROOT / ".github/workflows/build-webrtcvad-wheels.yml").exists()
    text = script.read_text(encoding="utf-8")
    assert "last_published_version" in text
    assert "gh release create" in text
    assert "unzip -l" in text
    assert "api/__init__.py" in text


def test_requirements_do_not_ship_unused_webrtcvad() -> None:
    v2 = (ROOT / "plugins.v2/subtitlestudio/requirements.txt").read_text(encoding="utf-8")
    v3 = (ROOT / "plugins.v3/subtitlestudio/requirements.txt").read_text(encoding="utf-8")
    pyproject = (ROOT / "plugins.v3/subtitlestudio/pyproject.toml").read_text(encoding="utf-8")
    assert "webrtcvad" not in v2
    assert "webrtcvad" not in v3
    assert "webrtcvad" not in pyproject
