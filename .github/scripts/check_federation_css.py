#!/usr/bin/env python3
"""阻止联邦插件把 Vuetify 基础样式注入主程序文档。

对照官方仓 jxxghp/MoviePilot-Plugins/.github/scripts/check_federation_css.py。
构建后、Release 前都应跑一遍。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FEDERATION_CONFIG_PATTERN = re.compile(r"\bfederation\s*\(")
DYNAMIC_CSS_PATTERN = re.compile(r"dynamicLoadingCss\s*\(\s*\[([^]]*)]", re.DOTALL)
CSS_PATH_PATTERN = re.compile(r"['\"]([^'\"]+\.css)['\"]")
SELECTOR_PATTERN = re.compile(r"(?:^|})\s*([^@{}][^{}]*)\{", re.MULTILINE)
GLOBAL_SELECTOR_PATTERN = re.compile(
    r"^(?:html\b|body\b|:root\b|\*\s*(?:$|[,>+~.#[:])|"
    r"\.v-|\.mdi-|\.rounded(?:\b|-)|\.elevation-\d+\b)"
)
PACKAGE_BY_GENERATION = {
    "plugins": "package.json",
    "plugins.v2": "package.v2.json",
    "plugins.v3": "package.v3.json",
}


def _federation_plugin_dirs(root: Path) -> list[Path]:
    plugin_dirs: set[Path] = set()
    for config in root.glob("plugins*/**/vite.config.*"):
        if "node_modules" in config.parts or not config.is_file():
            continue
        if FEDERATION_CONFIG_PATTERN.search(config.read_text(encoding="utf-8")):
            plugin_dirs.add(config.parent)
    for remote_entry in root.glob("plugins*/**/remoteEntry.js"):
        if "node_modules" in remote_entry.parts:
            continue
        relative = remote_entry.relative_to(root)
        if len(relative.parts) >= 2:
            plugin_dirs.add(root / relative.parts[0] / relative.parts[1])
    return sorted(plugin_dirs)


def _release_error(root: Path, plugin_dir: Path) -> str | None:
    relative = plugin_dir.relative_to(root)
    package_name = PACKAGE_BY_GENERATION.get(relative.parts[0])
    if not package_name:
        return f"{relative}: 无法确定插件市场索引"
    package_path = root / package_name
    if not package_path.is_file():
        return f"{relative}: 缺少市场索引 {package_name}"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    matches = [
        (plugin_id, metadata)
        for plugin_id, metadata in package.items()
        if plugin_id.casefold() == plugin_dir.name.casefold()
    ]
    if not matches:
        return f"{relative}: 未在 {package_name} 中登记"
    plugin_id, metadata = matches[0]
    if not isinstance(metadata, dict) or metadata.get("release") is not True:
        return f"{package_name}: 联邦插件 {plugin_id} 必须设置 release=true"
    return None


def _referenced_css(remote_entry: Path) -> list[Path]:
    source = remote_entry.read_text(encoding="utf-8")
    paths: set[Path] = set()
    for array_source in DYNAMIC_CSS_PATTERN.findall(source):
        for css_path in CSS_PATH_PATTERN.findall(array_source):
            paths.add((remote_entry.parent / css_path).resolve())
    return sorted(paths)


def _global_selectors(css_file: Path) -> list[str]:
    css = re.sub(r"/\*.*?\*/", "", css_file.read_text(encoding="utf-8"), flags=re.DOTALL)
    violations: set[str] = set()
    for selector_group in SELECTOR_PATTERN.findall(css):
        for selector in selector_group.split(","):
            normalized = re.sub(r"\s+", " ", selector).strip()
            if GLOBAL_SELECTOR_PATTERN.match(normalized):
                violations.add(normalized)
    return sorted(violations)


def check_repository(root: Path) -> list[str]:
    errors: list[str] = []
    for plugin_dir in _federation_plugin_dirs(root):
        relative_plugin = plugin_dir.relative_to(root)
        release_error = _release_error(root, plugin_dir)
        if release_error:
            errors.append(release_error)
        shared_styles = sorted(plugin_dir.glob("**/__federation_shared_vuetify/styles-*.css"))
        for css_file in shared_styles:
            errors.append(f"{css_file.relative_to(root)}: 不得发布 Vuetify 共享基础样式")

        remote_entries = sorted(plugin_dir.glob("**/remoteEntry.js"))
        for remote_entry in remote_entries:
            if "node_modules" in remote_entry.parts:
                continue
            for css_file in _referenced_css(remote_entry):
                try:
                    relative_css = css_file.relative_to(root)
                except ValueError:
                    errors.append(
                        f"{remote_entry.relative_to(root)}: CSS 引用越出仓库：{css_file}"
                    )
                    continue
                if not css_file.is_file():
                    errors.append(
                        f"{remote_entry.relative_to(root)}: CSS 文件不存在：{relative_css}"
                    )
                    continue
                selectors = _global_selectors(css_file)
                if selectors:
                    preview = ", ".join(selectors[:3])
                    errors.append(
                        f"{relative_css}: 包含未限定作用域的宿主样式选择器：{preview}"
                    )

        if not remote_entries:
            print(f"跳过未构建联邦插件：{relative_plugin}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查联邦插件 CSS 是否污染宿主页面")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="插件仓库根目录",
    )
    return parser.parse_args()


def main() -> int:
    root = parse_args().root.resolve()
    errors = check_repository(root)
    if errors:
        print("联邦插件 CSS 门禁失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    print("联邦插件 CSS 门禁通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
