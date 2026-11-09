from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _zip_plugin(plugin_dir: Path, dest: Path) -> None:
    dest.unlink(missing_ok=True)
    excluded_dirs = {"node_modules", "__pycache__", ".pytest_cache"}
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in plugin_dir.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(plugin_dir.parent)
            if path.suffix == ".pyc" or excluded_dirs.intersection(relative.parts):
                continue
            archive.write(path, relative.as_posix())


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
    assert "check_federation_css.py" in workflow
    assert "Diagnose environment" in workflow
    assert "workflow_dispatch" in workflow
    assert "paths:" not in workflow
    assert not (ROOT / ".github/workflows/build-webrtcvad-wheels.yml").exists()
    text = script.read_text(encoding="utf-8")
    assert "last_published_version" in text
    assert "gh release create" in text
    assert "unzip -l" in text
    assert "api/__init__.py" in text


def test_federation_css_gate_passes() -> None:
    script = ROOT / ".github/scripts/check_federation_css.py"
    assert script.is_file()
    result = subprocess.run(
        ["python3", str(script), "--root", str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "门禁通过" in result.stdout


def test_runtime_manifests_do_not_ship_unused_webrtcvad() -> None:
    v2 = (ROOT / "plugins.v2/subtitlestudio/requirements.txt").read_text(encoding="utf-8")
    v3 = (ROOT / "plugins.v3/subtitlestudio/pyproject.toml").read_text(encoding="utf-8")
    assert "webrtcvad" not in v2
    assert "webrtcvad" not in v3
