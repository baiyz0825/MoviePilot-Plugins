# Agent 操作手册

给后续在本仓库改代码的 agent 用。先读本文，再动字幕工坊。用户文档在 [README.md](README.md)。

## 这是什么仓库

个人 MoviePilot 插件仓，同时发 V2 和 V3。**只维护、只上架字幕工坊**（`SubtitleStudio`）。海拉鲁字幕大师、AutoSubv3、Emby 媒体库封面生成已下架，禁止再写进 `package*.json`，也禁止运行时委托旧字幕插件。

官方规范（以官方仓当前文档为准，不要凭记忆）：

- https://github.com/jxxghp/MoviePilot-Plugins/blob/main/docs/Plugin_Development.md
- https://github.com/jxxghp/MoviePilot-Plugins/blob/main/docs/V2_Plugin_Development.md
- https://github.com/jxxghp/MoviePilot-Plugins/blob/main/docs/V3_Plugin_Adaptation.md
- https://github.com/jxxghp/MoviePilot-Plugins/blob/main/docs/V3_API_Response_Adaptation.md
- https://github.com/jxxghp/MoviePilot-Plugins/blob/main/docs/Repository_Guide.md

本仓设计稿（`docs/` 被 gitignore，已跟踪的文件用 `git add -f`）：

- `docs/SubtitleStudio功能使用教学.md`
- `docs/subtitle-studio-v1.html`
- `docs/call-graphs/subtitle-studio-*.html`

## 双树必须同时改

| 代 | 目录 | 版本线 | 索引 | 宿主下界 |
| --- | --- | --- | --- | --- |
| V2 | `plugins.v2/subtitlestudio/` | `1.x` | `package.v2.json`，`"v3": false` | `>=2.13.5` |
| V3 | `plugins.v3/subtitlestudio/` | `2.x` | `package.v3.json` | `>=3.0.0` |

- 产品行为、FIELDS、页面、导出命名两边一致。只允许 `host.py`、`__init__.py` 宿主钩子、依赖清单不同。
- 领域层（`core/` `pipeline/` `packager/` `ingest/` `storage/` `api/`）改一边就同步另一边。
- **禁止** V2 / V3 共用同一个 `plugin_version`。Tag 是 `SubtitleStudio_v版本`，撞号会互相覆盖。
- 根 `package.json` **不要**再给字幕工坊加条目。
- 类名 `SubtitleStudio`，目录必须小写 `subtitlestudio`。

## 宿主合同（硬约束）

领域 `.py` 行首禁止 `from app.` / `import app.`（`host.py`、`__init__.py`、`agent_tools.py` 除外）。

**V2**

- 基类 `app.plugins._PluginBase`
- `app.log` / `app.core.event` + 方法上 `@eventmanager.register`
- HTTP `httpx`，不要 `httpx2`
- 身份从 `tmdbid` / `doubanid` 合成
- 防抖：`DateTrigger` + `replace_existing`，id 用 `{plugin_id}.ingest_once`
- `TransferComplete` 必须 `scheduler.enqueue`，禁止空转
- 依赖 `requirements.txt`

**V3**

- 基类 `app.sdk.plugin._PluginBase`
- `app.sdk.logging` / `app.sdk.events`；`EventType` 走 `app.schemas.types`
- HTTP `httpx2`，禁止 `alias_httpx()`
- 身份必须成对 `media_source` + `media_id`
- 防抖：`add_plugin_once_job(self.__class__.__name__, ...)`
- 事件在 `init_plugin` 里 `register_listener`，导入期禁止 `eventmanager.register`
- 宿主数据只走 `app.db.oper.*`，禁止 `SessionFactory` / `app.db.models`
- 依赖 `pyproject.toml`：`name = "moviepilot-plugin-subtitlestudio"`，`dynamic = ["version"]`，`requires-python = ">=3.14"`
- API：`finalize_api_routes(..., generation="v3")` 套 `schemas.Response[T]`；预览视频 / ASS 保持 `response_model=None` + `response_class` + OpenAPI `responses`

虚拟分身：插件 ID、once-job、Agent 查找、`get_service` id 一律 `self.__class__.__name__`，不要写死 `"SubtitleStudio"`。

通知走基类 `post_message(mtype=NotificationType.Plugin)`。

## 前端

- `get_render_mode()` → `("vue", "dist/assets")`。宿主只加载 `dist/assets/remoteEntry.js`。
- 联邦名 `SubtitleStudio`，暴露 `./Page` `./Config` `./AppPage` `./Dashboard`。
- shared 必须是 `vue` / `vuetify` / `vuetify/styles`，且 `generate: false`、`singleton: true`。
- PostCSS 清掉 `node_modules/vuetify`、`@mdi`；产物里禁止 `__federation_shared_vuetify*`。
- 构建会丢掉 `dist/index.html` 和 `assets/index-*.js`，不要再加回去。
- 组件必须声明并优先使用宿主注入的 `api`、`pluginId`、`sourcePluginId`。请求走 `plugin/${pluginId}/...`，不要写 `/api/v1`，不要自己 new Axios。
- 手机断点 `899px`。不要插件自己的 `VBottomNavigation`。
- 改 `src/` 后两边都要 `npm run build`，并提交新的 `dist/assets/`。

## 业务红线

- STRM：只搜外挂 / 翻译 / 导出；**永不 ASR、不调轴**。
- 搜索偏好 ≠ 导出包。导出预设：`library_zh` / `plex` / `fnos` / `web` / `legacy`。
- 目标语言 1–3 种，顺序即字号层级。不生成 SUB/IDX/PGS。
- 特效是顶注 / 保留特效 ASS，不是卡拉 OK、不是飞字。
- 空查询 `success=true`，用 `data` 表示空；`message` 不写堆栈、Cookie、令牌、内部路径。
- 运行数据写 `get_data_path()`，不要写回插件源码目录。
- 导入期禁止启动任务、访问网络、连数据库。

## 版本与发布

只改代码不改版本，市场不会提示升级。升版本时同一代这些必须一致：

1. `__init__.py` 的 `plugin_version`
2. `package.v2.json` 或 `package.v3.json` 的 `version`
3. 同文件 `history` 置顶 `v{version}`
4. 插件目录 `package.json` 的 `version`

V3 `pyproject.toml` 不要写死 `version`。两边都改就各升一号（`1.0.0→1.0.1` 且 `2.0.0→2.0.1`）。

```bash
python .github/scripts/check_plugin_versions.py package.json package.v2.json package.v3.json
```

`"release": true` 后由 `.github/workflows/release.yml` 自动打包。目录相对旧 Tag 有变更、Release/zip 缺失、或手动 `force=true` 时，会删除同名 Release 再打 `subtitlestudio_v版本.zip` 和 Tag `SubtitleStudio_v版本`。日志会打出索引、决策、zip 全量清单和 `api/` 校验。仓库 Settings 里必须打开 GitHub Actions；没有对应 Release 时，宿主会按 release 路径安装失败或只落到部分文件，加载报 `No module named 'app.plugins.subtitlestudio.api'`。

## 测试

测试必须 `import app.plugins.subtitlestudio`（`tests/subtitlestudio_support/loader.py`），不要把插件目录当顶层包。

```bash
PYTHONPATH=. pytest tests/v2/subtitlestudio tests/v3/subtitlestudio tests/test_v3_plugin_adaptation.py
```

本地常用解释器：`/tmp/ss-pytest/bin/pytest`（若存在）。

## Git

- **用户没说 commit / push，就不要做。**
- 提交说明沿用：`feat(subtitlestudio): --other=一句话说明为什么`
- `docs/` 被 gitignore，更新已跟踪文档用 `git add -f`
- 不要改 git config，不要 `--no-verify`，不要 force push main

## 动手前自检

1. 这次改动是不是产品合同？是 → V2 / V3 一起改。
2. 动了 Vue？两边 `npm run build`，确认没有 `dist/index.html` 和 shared vuetify CSS。
3. 要给市场用户用？按「版本与发布」对齐四元组后再提。
4. 领域层有没有新出现行首 `from app.`？
5. 有没有把插件 ID 写死成 `"SubtitleStudio"`？
