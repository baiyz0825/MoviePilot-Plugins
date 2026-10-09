# MoviePilot-Plugins

同时维护 MoviePilot V2 与 V3 的个人插件仓库。本仓库只发布 **字幕工坊**，已下架海拉鲁字幕大师、AutoSubv3 和 Emby 媒体库封面生成。

## 文档

使用、设计和调用图都在 `docs/`，打开即可预览：

| 文档 | 看什么 |
| --- | --- |
| [字幕工坊功能使用教学](docs/SubtitleStudio功能使用教学.md) | 安装、四个一级页、配置项、导出命名、常见问题 |
| [字幕工坊 V1 设计说明](docs/subtitle-studio-v1.html) | 功能对比、FIELDS、页面稿、开发约定 |
| [宿主边界](docs/call-graphs/subtitle-studio-host.html) | V2 / V3 宿主适配与官方钩子 |
| [入库链路](docs/call-graphs/subtitle-studio-ingest.html) | 事件、目录监控、STRM、手动勾选 |
| [任务处理](docs/call-graphs/subtitle-studio-job.html) | 单任务从入队到导出 |
| [任务生命周期](docs/call-graphs/subtitle-studio-job-lifecycle.html) | 队列状态、插队、取消、重试 |
| [生成流水线](docs/call-graphs/subtitle-studio-pipeline.html) | 搜索、ASR、翻译、特效、写盘 |

插件目录里的说明：

- [V2 字幕工坊](plugins.v2/subtitlestudio/README.md)
- [V3 字幕工坊](plugins.v3/subtitlestudio/README.md)
- [Agent 操作手册](AGENTS.md)（给后续改代码的 agent 用）

## 插件列表

| 插件 | V2 版本 | V3 版本 | 主要功能 |
| --- | --- | --- | --- |
| 字幕工坊 | 1.0.2 | 2.0.2 | 搜索 / 识别 / 翻译 / 特效顶注 / 导出包 / 工作台预览 |

## 字幕工坊

从 MoviePilot 整理记录拉取媒体，勾选后入队；也可听整理完成事件或监控目录。后台搜索或 ASR、翻译，再按播放器语言码写出字幕。V2 与 V3 页面和配置相同，只是对接的宿主接口不同。

主要功能：

- 整理完成事件、目录监控、STRM（只搜外挂，不做 ASR / 调轴）和媒体库手动勾选入队。
- SubHD、Zimuku、ASSRT、OpenSubtitles 搜索；搜到外挂就不 ASR。
- 免费翻译引擎优先，大模型补译和格式修复。
- 导出预设：媒体库 / Plex / 飞牛 / Infuse / 兼容旧库。
- ASS 字体、颜色、多行字号可配置，设置页带预览。
- 可选 MoviePilot 通知（类型：插件），默认同步成功和失败。
- 工作台时间轴、挂载预览、队列插队。

版本：V2 `1.0.2`（`>=2.13.5`，`package.v2.json` 且 `"v3": false`），V3 `2.0.2`（`>=3.0.0`）。索引 `"release": true`，安装走 GitHub Release 里的完整 zip（`subtitlestudio_v1.0.2.zip` / `subtitlestudio_v2.0.2.zip`）。没有这个包时，宿主可能只落到入口文件，加载报 `No module named 'app.plugins.subtitlestudio.api'`。

## 安装方式

在 MoviePilot 第三方插件仓库中添加本仓库地址：

```text
https://github.com/baiyz0825/MoviePilot-Plugins
```

安装「字幕工坊」即可，不必再装旧的字幕匹配或 AI 字幕生成插件。

## 注意事项

- 字幕站点和 API 可能变化，在线下载不能保证所有来源始终可用。
- ASR 依赖 `faster-whisper` 和本机 ffmpeg。
- 字体颜色只写入 ASS；SRT / VTT 由播放器自己排版。
- AI 翻译质量取决于模型、提示词和原字幕质量。
- Vue 联邦产物在插件目录的 `dist/assets/`，宿主读 `remoteEntry.js`；改 `src/` 后要重新 `npm run build`。

## 致谢

本仓库基于 MoviePilot 插件机制开发。

- [MoviePilot](https://github.com/jxxghp/MoviePilot)
- [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
