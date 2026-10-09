# MoviePilot-Plugins

同时维护 MoviePilot V2 与 V3 的个人插件仓库。字幕链路现在只维护 **字幕工坊**，不再提供海拉鲁字幕大师和 AutoSubv3。

## 📘 [字幕工坊功能使用教学](docs/SubtitleStudio功能使用教学.md)

独立完成搜索、识别、免费引擎 / 大模型翻译、特效顶注、导出包和工作台预览。

## 插件列表

| 插件 | V2 版本 | V3 版本 | 主要功能 |
| --- | --- | --- | --- |
| 字幕工坊 | 1.0.0 | 2.0.0 | 搜索 / 识别 / 翻译 / 特效顶注 / 导出包 / 工作台预览 |
| Emby媒体库封面生成 | 0.9.11 | — | 生成媒体库动态 / 静态封面 |

## 字幕工坊

从 MoviePilot 整理记录拉取媒体，勾选后入队，后台搜索或 ASR、翻译、按播放器语言码写出字幕。

V2 版本 `1.0.0`，V3 版本 `2.0.0`。完整配置、四个一级页和开发约定见：

- [字幕工坊功能使用教学](docs/SubtitleStudio功能使用教学.md)
- [字幕工坊 V1 设计说明](docs/subtitle-studio-v1.html)

主要功能：

- 整理完成事件、目录监控、STRM 和手动勾选入队。
- SubHD、Zimuku、ASSRT、OpenSubtitles 搜索，搜到外挂就不 ASR。
- 免费翻译引擎优先，大模型补译和格式修复。
- 按 MoviePilot / Emby / Plex / 飞牛 / Infuse 语言码导出。
- ASS 字体、颜色、多行字号可配置，设置页带预览。
- 工作台时间轴、挂载预览、队列插队。

## 安装方式

在 MoviePilot 第三方插件仓库中添加本仓库地址：

```text
https://github.com/ifsherlock/MoviePilot-Plugins
```

安装「字幕工坊」即可，无需再装旧的字幕匹配或 AI 字幕生成插件。

## 注意事项

- 字幕站点和 API 可能变化，在线下载不能保证所有来源始终可用。
- ASR 依赖 `faster-whisper` 和本机 ffmpeg。
- 字体颜色只写入 ASS；SRT / VTT 由播放器自己排版。
- AI 翻译质量取决于模型、提示词和原字幕质量。

## 致谢

本仓库基于 MoviePilot 插件机制开发。

- [MoviePilot](https://github.com/jxxghp/MoviePilot)
- [MoviePilot-Plugins](https://github.com/jxxghp/MoviePilot-Plugins)
