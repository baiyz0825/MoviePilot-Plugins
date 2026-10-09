# 字幕工坊 / SubtitleStudio（V2）

独立完成字幕搜索、识别、免费引擎 / 大模型翻译、特效顶注和工作台预览。不委托其它字幕插件。本仓库已下架海拉鲁字幕大师和 AutoSubv3。

- 版本：`1.0.2`
- 宿主：MoviePilot `>=2.13.5`
- 索引：`package.v2.json`，必须 `"v3": false`
- 发布包：`subtitlestudio_v1.0.2.zip`，Tag `SubtitleStudio_v1.0.2`

产品合同、四个一级页、FIELDS、CueGraph、导出包与 [V3 副本](../../plugins.v3/subtitlestudio/README.md) 相同，只是宿主接口不同。

## 文档

相对本目录的仓库文档：

| 文档 | 看什么 |
| --- | --- |
| [功能使用教学](../../docs/SubtitleStudio功能使用教学.md) | 安装、配置、导出命名、常见问题 |
| [V1 设计说明](../../docs/subtitle-studio-v1.html) | 功能对比、FIELDS、页面稿、开发约定 |
| [宿主边界](../../docs/call-graphs/subtitle-studio-host.html) | V2 适配：`app.plugins` / `httpx` / `DateTrigger` |
| [入库链路](../../docs/call-graphs/subtitle-studio-ingest.html) | 事件、目录监控、STRM、手动勾选 |
| [任务处理](../../docs/call-graphs/subtitle-studio-job.html) | 单任务从入队到导出 |
| [任务生命周期](../../docs/call-graphs/subtitle-studio-job-lifecycle.html) | 队列状态、插队、取消、重试 |
| [生成流水线](../../docs/call-graphs/subtitle-studio-pipeline.html) | 搜索、ASR、翻译、特效、写盘 |

仓库总览见 [根 README](../../README.md)。

## 和 V3 的差别

| 项 | V2 |
| --- | --- |
| 基类 | `app.plugins._PluginBase` |
| 日志 / 事件 | `app.log` / `app.core.event`，方法上 `@eventmanager.register` |
| HTTP | `httpx` |
| 身份 | 从 `tmdbid` / `doubanid` 合成 |
| 防抖 | `DateTrigger` + `replace_existing`，任务 id 带运行实例名 |
| 依赖 | `requirements.txt` |
| TransferComplete | 必须真正入队 |

虚拟分身用 `self.__class__.__name__` 找自己，不要写死源类名。联邦组件接收宿主注入的 `pluginId`、`sourcePluginId` 和 `api`。

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
cd plugins.v2/subtitlestudio
npm install
npm run build
```

改 `src/` 后必须重新构建并提交 `dist/assets/`。单测：`pytest tests/v2/subtitlestudio`。
