# 字幕工坊 / SubtitleStudio（V2）

独立完成字幕搜索、识别、免费引擎 / 大模型翻译、特效顶注和工作台预览。不委托海拉鲁字幕大师或 AutoSub。

- 版本：`1.0.0`
- 宿主：MoviePilot `>=2.13.5`
- 索引：`package.v2.json`，必须 `"v3": false`
- 发布包：`subtitlestudio_v1.0.0.zip`，Tag `SubtitleStudio_v1.0.0`

## 和 V3 的差别

| 项 | V2 |
| --- | --- |
| 基类 | `app.plugins._PluginBase` |
| 日志 / 事件 | `app.log` / `app.core.event` |
| HTTP | `httpx` |
| 身份 | 从 `tmdbid` / `doubanid` 合成 |
| 防抖 | `DateTrigger` + `replace_existing` |
| TransferComplete | 必须真正入队 |

产品合同、四个一级页、FIELDS、CueGraph、导出包与 V3 相同。完整说明见仓库 `docs/SubtitleStudio功能使用教学.md` 和 `docs/subtitle-studio-v1.html`。

## 页面

侧栏「字幕工坊」一条入口，页内切：媒体 / 队列 / 工作台 / 设置。手机 899px 用顶部分段，不要插件底栏。

## 开发

```bash
cd plugins.v2/subtitlestudio
npm install
npm run build
```

单测：`pytest tests/v2/subtitlestudio`
