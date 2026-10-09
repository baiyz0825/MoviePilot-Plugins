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
- https://github.com/jxxghp/MoviePilot-Plugins/blob/main/docs/FAQ.md
- https://github.com/jxxghp/MoviePilot-Plugins/blob/main/docs/faq/16-register-agent-tools.md
- https://github.com/jxxghp/MoviePilot-Plugins/blob/main/docs/faq/17-register-plugin-sidebar-nav.md
- https://github.com/jxxghp/MoviePilot-Frontend/blob/v3/docs/module-federation-guide.md

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

`"release": true` 后只留 `.github/workflows/release.yml`。`main` 推送或手动触发时，对比当前索引版本和上一份同代 Release：没有包、版本变了、插件目录有变化，或 `force=true`，就打 `subtitlestudio_v版本.zip` 和 Tag `SubtitleStudio_v版本`。V2 必须是 `1.x`，V3 必须是 `2.x`，宿主靠 `package.v2.json` / `package.v3.json` 选代，再按这个 Tag 下 zip。日志会打出对比过程、zip 全量清单和 `api/` 校验。仓库 Settings 里必须打开 GitHub Actions。

## 测试

测试必须 `import app.plugins.subtitlestudio`（`tests/subtitlestudio_support/loader.py`），不要把插件目录当顶层包。

```bash
PYTHONPATH=. pytest tests/v2/subtitlestudio tests/v3/subtitlestudio tests/test_v3_plugin_adaptation.py
```

本地常用解释器：`/tmp/ss-pytest/bin/pytest`（若存在）。

## 官方规范与已踩坑

手册文件名是 **`AGENTS.md`**，不要再新建 `agent.md`。改规范前先读上面「官方规范」链接的当前文档，不要凭记忆。

**2026-10-09 对照官方仓结论**：双树目录、版本线、`v3: false`、生命周期、Vue 联邦 shared、`auth=bear`、信封、空查询、`get_data_path()`、V3 SDK / Oper / `httpx2`、`media_source+media_id`、`add_plugin_once_job`、侧栏 `get_sidebar_nav`、`release: true`、labels 逗号字符串、Agent `ClassVar` + FAQ 16 字段标注，都已按当前官方文档落地。官方要求构建后跑的 `check_federation_css.py` 已放进本仓，Release 会执行。

下面这些已经在字幕工坊里踩过或对过官方合同，**不要再引入**：

| 坑 | 现象 | 正确做法 |
| --- | --- | --- |
| `release: true` 但没有 GitHub Release / zip | 宿主按 Tag 下包失败或只落到 `__init__.py`，加载报 `No module named 'app.plugins.subtitlestudio.api'` | 仓库必须能跑 Actions；zip 根是 `subtitlestudio/`，必须含 `api/`、`core/`、`dist/assets/remoteEntry.js`。发布脚本会校验这些文件。 |
| V2 / V3 共用 `plugin_version` | Tag `SubtitleStudio_v版本` 互相覆盖 | V2 只走 `1.x`，V3 只走 `2.x`。宿主先读 `package.v2.json` / `package.v3.json`，再按下标版本下对应 zip。 |
| 只改代码不改四元组 | 市场不提示升级 | 对齐 `__init__.py` / 索引 `version` / `history` 置顶 / 插件目录 `package.json`。 |
| 同版本 Tag 已存在就跳过打包 | 后补的 `api/` 永远发不出去 | 现在目录有变更、包缺失或 `force=true` 会删旧 Release 再打。 |
| Agent 工具未标注类型 | Pydantic 2.13：`plugin_lookup_id` 被当成模型字段，导入失败 | `MoviePilotTool` 是 Pydantic 模型。`name: str` / `description: str` / `args_schema: Type[BaseModel]` 按官方 FAQ 写成字段；**多出来的类属性必须 `ClassVar`**。禁止 `type(..., {"plugin_lookup_id": ...})` 这种无标注动态类。`args_schema` 用 BaseModel，不要 `{}`。 |
| 测试或源码写进插件目录 | 市场同步把测试拷进运行目录 | 测试只放 `tests/v2/subtitlestudio`、`tests/v3/subtitlestudio`。 |
| 测试 `import subtitlestudio` | 和宿主 `app.plugins.subtitlestudio` 变成两套模块 | 必须走 `tests/subtitlestudio_support/loader.py`。 |
| 领域层 `from app.` | V3 适配失败、双树无法共用 | 只有 `host.py`、`__init__.py`、`agent_tools.py` 能碰 `app.`。 |
| V3 `SessionFactory` / `app.db.models` / `alias_httpx()` | 官方明确禁止 | Oper + `httpx2`。 |
| V3 导入期 `@eventmanager.register` | 热重载 / 分身重复注册 | 事件在 `init_plugin` 里 `register_listener`。 |
| Vue 自带 Vuetify CSS | 宿主样式炸掉；官方门禁拒收 `__federation_shared_vuetify*` | shared 仅 `vue` / `vuetify` / `vuetify/styles` 且 `generate: false`；PostCSS 滤掉 `node_modules/vuetify`、`@mdi`。构建后必须跑 `python .github/scripts/check_federation_css.py`（官方同款门禁，Release 也会跑）。局部样式只写 `.plugin-root .v-btn`，不要裸写 `.v-btn`。`build.target` 必须是 `esnext`。 |
| 自己 new Axios 或写 `/api/v1` | 丢认证、双层 `data` | 用注入的 `api`，路径 `plugin/${pluginId}/...`。轮询 `feedback: 'silent'`，保存 `feedback: 'all'`，不要对同一错误再弹 Toast。 |
| 空查询 `success=false` | 前端当失败弹 Toast | 查完但没数据：`success=true`，空放 `data`。 |
| `message` 写堆栈 / 路径 / Cookie | 官方禁止 | 只给用户看的一句话。 |
| 普通 JSON 用 `response_model=None` | OpenAPI 空洞 | 只有预览视频 / ASS 等原生响应可以；JSON 走具体模型或 `Response[T]`。 |
| 索引 `labels` 写成数组 | 旧宿主市场解析 / 序列化异常 | 用逗号分隔字符串，例如 `"字幕,翻译,ASR,工作台"`。 |
| 把测试、文档、旧插件写进 `package*.json` | 市场出现已下架插件 | 本仓索引只留 `SubtitleStudio`；根 `package.json` 保持 `{}`。 |
| 为 ASR 自建 webrtcvad wheel 工作流 | 本仓不调用 `webrtcvad`，脚本还指向已删的 `subtitlemanualupload` | 不要恢复 `build-webrtcvad-wheels.yml`。 |
| 运行数据写进插件源码目录 | 升级 zip 覆盖用户数据 | 只写 `get_data_path()`。 |
| 导入期启动任务 / 访问网络 / 连库 | 加载慢或直接失败 | 只在 `init_plugin` 之后做事。 |
| 插件 ID 写死 `"SubtitleStudio"` | 虚拟分身 once-job / Agent / 服务串台 | 用 `self.__class__.__name__`。 |
| V3 新代码走 `app.core` / `app.helper` / `app.utils` / `app.sdk._legacy` | `DEBUG=true` 报兼容警告；以后兼容层会撤 | V3 只从 `app.sdk.*`、`app.schemas`、`app.db.oper.*`、`app.agent.*` 进。 |
| V3 自建 `BackgroundScheduler` 做一次性任务 | 线程泄漏、重载后旧任务还在跑 | 用 `add_plugin_once_job(self.__class__.__name__, ...)`。V2 用 `DateTrigger` + `replace_existing`。 |
| 把 `get_page()` 当侧栏全页 | 只出现在插件管理弹窗 | 侧栏必须 `get_sidebar_nav()` + 联邦暴露 `./AppPage`，并声明 `navKey`。 |
| 插件代码里 `pip` / `uv`，或提交 `uv.lock` | 宿主共享环境被改坏 | V3 只写 `pyproject.toml` 的 `dependencies`，`dynamic = ["version"]`。 |
| 市场日志 `插件包数据解析失败` | 宿主把索引 HTML（GitHub raw 超时/反爬）当 JSON 解 | 先确认 `package.v2.json` / `package.v3.json` 是合法 JSON、`labels` 是字符串。这是宿主拉索引失败，不是插件 zip 坏了。 |
| zip 带上 `node_modules` / `__pycache__` | 包巨大或污染运行目录 | 发布脚本已排除。宿主只加载 `dist/assets/remoteEntry.js`，不要再提交 `dist/index.html`。 |

宿主怎么选 V2 / V3：V2 读 `package.v2.json`（`v3: false`，`system_version >=2.13.5`），V3 读 `package.v3.json`（`>=3.0.0`）。`release: true` 时下载 `SubtitleStudio_v{索引version}` 的 `subtitlestudio_v{version}.zip`，解压到 `app/plugins/subtitlestudio/`。

官方示例里 V3 仍可用方法上的 `@eventmanager.register`。本仓 V3 **故意**改成 `init_plugin` 里 `register_listener`、`stop_service` 里摘掉，避免虚拟分身 / 热重载重复注册。不要改回导入期装饰器。

本地联调（官方推荐，本仓没有独立宿主）：`PLUGIN_LOCAL_REPO_PATHS` 指到本仓库，`PLUGIN_AUTO_RELOAD=true`，`DEBUG=true`。插件最终仍应在真实 V2 / V3 宿主里至少加载一次。

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
6. 动了 `MoviePilotTool` 子类？额外属性是 `ClassVar`，`name` / `description` / `args_schema` 有类型标注。
7. 动了 Vue？跑 `python .github/scripts/check_federation_css.py`。
8. `release: true` 的改动推上去后，确认 Actions 打出的 zip 仍含 `api/` 和 `remoteEntry.js`。
