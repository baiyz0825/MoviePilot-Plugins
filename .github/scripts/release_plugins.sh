#!/usr/bin/env bash
# 按 package.v2.json / package.v3.json / package.json 里 release=true 的条目打包并发布。
# 没有对应 Tag、目录相对 Tag 有变更、或 Release 缺少 zip 时，自动创建或替换。
set -euo pipefail

FILTER_PLUGIN_ID="${FILTER_PLUGIN_ID:-}"
FORCE_RELEASE="${FORCE_RELEASE:-false}"
DRY_RUN="${DRY_RUN:-false}"
WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
TARGET_SHA="${TARGET_SHA:-}"

if [ -z "$TARGET_SHA" ]; then
  TARGET_SHA="$(git rev-parse HEAD)"
fi

STATE_DIR="${STATE_DIR:-${TMPDIR:-/tmp}/subtitlestudio-release-$$}"
mkdir -p "$STATE_DIR"
PROCESSED_TAGS="$STATE_DIR/processed_tags.txt"
SUMMARY_ROWS="$STATE_DIR/release_summary.tsv"
: > "$PROCESSED_TAGS"
: > "$SUMMARY_ROWS"

log() {
  printf '%s\n' "$*"
}

have_gh() {
  command -v gh >/dev/null 2>&1
}

section() {
  log ""
  log "======== $* ========"
}

plugin_dir_for() {
  local pkg_file="$1"
  local plugin_id_lc="$2"
  local dirs=("plugins.v2/${plugin_id_lc}" "plugins/${plugin_id_lc}")
  if [ "$pkg_file" = "package.json" ]; then
    dirs=("plugins/${plugin_id_lc}" "plugins.v2/${plugin_id_lc}")
  elif [ "$pkg_file" = "package.v3.json" ]; then
    dirs=("plugins.v3/${plugin_id_lc}" "plugins.v2/${plugin_id_lc}" "plugins/${plugin_id_lc}")
  fi
  local candidate
  for candidate in "${dirs[@]}"; do
    if [ -d "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

record() {
  local action="$1"
  local plugin_id="$2"
  local version="$3"
  local tag="$4"
  local detail="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "$action" "$plugin_id" "$version" "$tag" "$detail" >> "$SUMMARY_ROWS"
}

dir_changed_since_tag() {
  local tag="$1"
  local plugin_dir="$2"
  if ! git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    return 0
  fi
  if git diff --quiet "$tag" -- "$plugin_dir"; then
    return 1
  fi
  return 0
}

release_asset_present() {
  local tag="$1"
  local asset="$2"
  local names
  if ! names="$(gh release view "$tag" --json assets --jq '.assets[].name' 2>/dev/null)"; then
    return 1
  fi
  printf '%s\n' "$names" | grep -qxF "$asset"
}

zip_has_file() {
  local zip_path="$1"
  local rel="$2"
  unzip -l "$zip_path" | awk '{print $4}' | grep -qxF "$rel"
}

verify_one() {
  local zip_path="$1"
  local rel="$2"
  local src="$3"
  if [ ! -f "$src" ]; then
    log "  源文件不存在，跳过校验 $rel"
    return 0
  fi
  if zip_has_file "$zip_path" "$rel"; then
    log "  zip 包含 $rel"
    return 0
  fi
  log "  ERROR: zip 缺少 $rel"
  return 1
}

verify_zip() {
  local zip_path="$1"
  local plugin_dir="$2"
  local prefix
  prefix="$(basename "$plugin_dir")"
  local missing=0
  verify_one "$zip_path" "${prefix}/__init__.py" "$plugin_dir/__init__.py" || missing=1
  verify_one "$zip_path" "${prefix}/api/__init__.py" "$plugin_dir/api/__init__.py" || missing=1
  verify_one "$zip_path" "${prefix}/api/routes.py" "$plugin_dir/api/routes.py" || missing=1
  verify_one "$zip_path" "${prefix}/dist/assets/remoteEntry.js" "$plugin_dir/dist/assets/remoteEntry.js" || missing=1
  if [ "$missing" -ne 0 ]; then
    return 1
  fi
  return 0
}

append_step_summary() {
  if [ -z "${GITHUB_STEP_SUMMARY:-}" ] || [ ! -s "$SUMMARY_ROWS" ]; then
    return 0
  fi
  {
    echo "## 字幕工坊 Release"
    echo ""
    echo "| 动作 | 插件 | 版本 | Tag | 说明 |"
    echo "| --- | --- | --- | --- | --- |"
    while IFS=$'\t' read -r action plugin_id version tag detail; do
      echo "| $action | $plugin_id | $version | $tag | $detail |"
    done < "$SUMMARY_ROWS"
  } >> "$GITHUB_STEP_SUMMARY"
}

process_package() {
  local pkg_file="$1"
  section "处理索引 $pkg_file"

  if [ ! -f "$pkg_file" ]; then
    log "索引不存在，跳过：$pkg_file"
    return 0
  fi

  log "索引全部条目："
  jq -r 'to_entries[] | "  - \(.key): version=\(.value.version // "") release=\(.value.release // false)"' "$pkg_file"

  found=0
  while IFS= read -r entry; do
    [ -z "$entry" ] && continue
    found=1
    plugin_id="${entry%%|*}"
    plugin_version="${entry##*|}"
    plugin_id_lc="$(printf '%s' "$plugin_id" | tr '[:upper:]' '[:lower:]')"
    tag="${plugin_id}_v${plugin_version}"
    asset="${plugin_id_lc}_v${plugin_version}.zip"
    section "$pkg_file / $plugin_id $plugin_version"
    log "plugin_id=$plugin_id"
    log "version=$plugin_version"
    log "tag=$tag"
    log "asset=$asset"
    log "filter=${FILTER_PLUGIN_ID:-<all>}"
    log "force=$FORCE_RELEASE"
    log "dry_run=$DRY_RUN"
    log "target_sha=$TARGET_SHA"

    if ! plugin_dir="$(plugin_dir_for "$pkg_file" "$plugin_id_lc")"; then
      log "ERROR: 找不到插件目录，已检查 plugins.v3 / plugins.v2 / plugins"
      record "fail" "$plugin_id" "$plugin_version" "$tag" "missing plugin directory"
      return 1
    fi
    log "plugin_dir=$plugin_dir"
    log "目录文件数：$(find "$plugin_dir" -type f ! -path '*/node_modules/*' ! -path '*/__pycache__/*' | wc -l | tr -d ' ')"

    release_notes="$(jq -r --arg plugin_id "$plugin_id" --arg version "v$plugin_version" '.[$plugin_id].history[$version] // empty' "$pkg_file")"
    if [ -z "$release_notes" ]; then
      release_notes="Automated release of $plugin_id $plugin_version"
      log "history 为空，使用默认说明"
    else
      log "release_notes=$release_notes"
    fi

    if grep -qxF "$tag" "$PROCESSED_TAGS"; then
      log "本轮已处理过 $tag，跳过重复索引"
      record "skip" "$plugin_id" "$plugin_version" "$tag" "already processed in this run"
      continue
    fi

    local_tag="no"
    if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
      local_tag="yes"
    fi
    log "本地 Tag：$local_tag"

    remote_release="no"
    if have_gh && gh release view "$tag" >/dev/null 2>&1; then
      remote_release="yes"
      log "远程 Release 存在，详情："
      gh release view "$tag"
    elif have_gh; then
      log "远程 Release 不存在"
    else
      log "未找到 gh，按远程 Release 不存在处理（Actions 里必须有 gh）"
    fi

    asset_ok="no"
    if [ "$remote_release" = "yes" ] && release_asset_present "$tag" "$asset"; then
      asset_ok="yes"
    fi
    log "远程资产 $asset：$asset_ok"

    changed="yes"
    if [ "$local_tag" = "yes" ] && ! dir_changed_since_tag "$tag" "$plugin_dir"; then
      changed="no"
      log "相对 Tag $tag，目录 $plugin_dir 无变更"
      log "变更文件（应为空）："
      git diff --name-only "$tag" -- "$plugin_dir" || true
    else
      if [ "$local_tag" = "no" ]; then
        log "本地没有 Tag，视为需要打包"
      else
        log "相对 Tag $tag，目录 $plugin_dir 有变更："
        git diff --name-only "$tag" -- "$plugin_dir" || true
      fi
    fi

    if [ "$FORCE_RELEASE" != "true" ] && [ "$changed" = "no" ] && [ "$remote_release" = "yes" ] && [ "$asset_ok" = "yes" ]; then
      log "无需重新打包：目录未变且 Release 资产齐全"
      if [ "$DRY_RUN" != "true" ] && have_gh; then
        gh release edit "$tag" --notes "$release_notes"
        log "已同步 Release 说明"
      fi
      echo "$tag" >> "$PROCESSED_TAGS"
      record "skip" "$plugin_id" "$plugin_version" "$tag" "unchanged and asset present"
      continue
    fi

    if [ "$FORCE_RELEASE" = "true" ]; then
      action_reason="force=true"
    elif [ "$remote_release" = "no" ]; then
      action_reason="release missing"
    elif [ "$asset_ok" = "no" ]; then
      action_reason="zip asset missing"
    else
      action_reason="plugin directory changed"
    fi
    log "将打包并发布，原因：$action_reason"

    log "开始 zip（完整文件清单）："
    rm -f "$WORKSPACE/$asset"
    (
      cd "$(dirname "$plugin_dir")"
      zip -r "$WORKSPACE/$asset" "$(basename "$plugin_dir")" \
        -x "*/node_modules/*" \
        -x "*/__pycache__/*" \
        -x "*.pyc" \
        -x "*/.pytest_cache/*"
    )
    log "zip 完成：$(ls -lh "$WORKSPACE/$asset" | awk '{print $5, $9}')"
    log "zip 清单："
    unzip -l "$WORKSPACE/$asset"
    verify_zip "$WORKSPACE/$asset" "$plugin_dir"

    if [ "$DRY_RUN" = "true" ]; then
      log "dry-run：不创建 GitHub Release"
      echo "$tag" >> "$PROCESSED_TAGS"
      record "dry-run" "$plugin_id" "$plugin_version" "$tag" "$action_reason"
      continue
    fi

    if ! have_gh; then
      log "ERROR: 未找到 gh，无法创建 Release"
      record "fail" "$plugin_id" "$plugin_version" "$tag" "gh not found"
      return 1
    fi

    if [ "$remote_release" = "yes" ]; then
      log "删除已有 Release $tag"
      gh release delete "$tag" -y
      log "删除远程 Tag $tag（不存在则忽略）"
      git push origin ":refs/tags/$tag" || true
    fi
    git tag -d "$tag" >/dev/null 2>&1 || true

    log "创建 Release $tag，target=$TARGET_SHA"
    gh release create "$tag" "$WORKSPACE/$asset" \
      --title "$tag" \
      --notes "$release_notes" \
      --latest \
      --target "$TARGET_SHA"
    gh release view "$tag"
    echo "$tag" >> "$PROCESSED_TAGS"
    record "published" "$plugin_id" "$plugin_version" "$tag" "$action_reason"
  done < <(
    jq -r --arg filter "$FILTER_PLUGIN_ID" '
      to_entries
      | map(select(.value.release == true))
      | map(select($filter == "" or .key == $filter))
      | .[]
      | "\(.key)|\(.value.version)"
    ' "$pkg_file"
  )

  if [ "$found" -eq 0 ]; then
    log "没有需要打包的 release=true 插件（filter=${FILTER_PLUGIN_ID:-<all>}）"
  fi
}

section "发布脚本启动"
log "workspace=$WORKSPACE"
log "state_dir=$STATE_DIR"
log "pwd=$(pwd)"
log "git HEAD=$(git rev-parse HEAD)"
log "git 描述=$(git log -1 --format='%h %s')"
log "已有本地 Tag："
git tag -l 'SubtitleStudio_v*' || true
log "gh 身份："
if have_gh; then
  gh auth status || true
else
  log "gh 未安装（本地 dry-run 可忽略；GitHub Actions 里必须有）"
fi

process_package package.json
process_package package.v2.json
process_package package.v3.json

section "本轮结果"
if [ -s "$SUMMARY_ROWS" ]; then
  printf '%s\t%s\t%s\t%s\t%s\n' "action" "plugin" "version" "tag" "detail"
  cat "$SUMMARY_ROWS"
else
  log "没有任何插件被处理"
fi
append_step_summary
log "发布脚本结束"
