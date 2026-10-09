# 字幕工坊 / SubtitleStudio（V3）

独立完成字幕搜索、识别、免费引擎 / 大模型翻译、特效顶注和工作台预览。不委托其它字幕插件。本仓库已下架海拉鲁字幕大师和 AutoSubv3。

- 版本：`2.0.2`
- 宿主：MoviePilot `>=3.0.0`
- 索引：`package.v3.json`
- 发布包：`subtitlestudio_v2.0.2.zip`，Tag `SubtitleStudio_v2.0.2`

产品合同、四个一级页、FIELDS、CueGraph、导出包与 [V2 副本](../../plugins.v2/subtitlestudio/README.md) 相同，只是宿主接口不同。

## 文档

相对本目录的仓库文档：

| 文档 | 看什么 |
| --- | --- |
| [功能使用教学](../../docs/SubtitleStudio功能使用教学.md) | 安装、配置、导出命名、常见问题 |
| [V1 设计说明](../../docs/subtitle-studio-v1.html) | 功能对比、FIELDS、页面稿、开发约定 |
| [宿主边界](../../docs/call-graphs/subtitle-studio-host.html) | V3 适配：`app.sdk` / `httpx2` / `add_plugin_once_job` |
| [入库链路](../../docs/call-graphs/subtitle-studio-ingest.html) | 事件、目录监控、STRM、手动勾选 |
| [任务处理](../../docs/call-graphs/subtitle-studio-job.html) | 单任务从入队到导出 |
| [任务生命周期](../../docs/call-graphs/subtitle-studio-job-lifecycle.html) | 队列状态、插队、取消、重试 |
| [生成流水线](../../docs/call-graphs/subtitle-studio-pipeline.html) | 搜索、ASR、翻译、特效、写盘 |

仓库总览见 [根 README](../../README.md)。

## 和 V2 的差别

| 项 | V3 |
| --- | --- |
| 基类 | `app.sdk.plugin._PluginBase` |
| 日志 / 事件 | `app.sdk.logging` / `app.sdk.events`；`EventType` 走 `app.schemas.types` |
| HTTP | `httpx2`，禁止 `alias_httpx()` |
| 身份 | 必须成对 `media_source` + `media_id` |
| 防抖 | `add_plugin_once_job`，插件 ID 用运行实例名 |
| 事件注册 | 在 `init_plugin` 里挂，导入期不 `register` |
| 宿主数据 | 只走 `app.db.oper.*`，不碰 Model / SessionFactory |
| 依赖 | `pyproject.toml`（`dynamic = ["version"]`，`requires-python >= 3.14`） |
| API | 普通 JSON 声明 `schemas.Response[T]`；预览视频 / ASS 保持原生响应 |

虚拟分身用 `self.__class__.__name__` 找自己。联邦组件接收宿主注入的 `pluginId`、`sourcePluginId` 和 `api`。

## 当前能力

- 整理完成事件、目录监控、STRM（只搜外挂）和媒体库手动勾选入队。
- 导出预设：媒体库 / Plex / 飞牛 / Infuse / 兼容旧库。
- ASS 字体、颜色、多行字号可配置，设置页带预览。
- 可选通知（`NotificationType.Plugin`），默认同步成功和失败。
- 工作台时间轴、挂载预览、队列插队。

## 页面

侧栏「字幕工坊」一条入口，页内切：媒体 / 队列 / 工作台 / 设置。手机 `≤899px` 用顶部分段，不要插件底栏。

`get_render_mode()` 返回 `vue` + `dist/assets`。宿主加载 `dist/assets/remoteEntry.js`，不读 `dist/index.html`（构建时已去掉）。

## 开发

```bash
cd plugins.v3/subtitlestudio
npm install
npm run build
```

改 `src/` 后必须重新构建并提交 `dist/assets/`。单测：`pytest tests/v3/subtitlestudio`。
