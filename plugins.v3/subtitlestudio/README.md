# 字幕工坊 / SubtitleStudio（V3）

独立完成字幕搜索、识别、免费引擎 / 大模型翻译、特效顶注和工作台预览。不委托海拉鲁字幕大师或 AutoSub。

- 版本：`2.0.0`
- 宿主：MoviePilot `>=3.0.0`
- 索引：`package.v3.json`
- 发布包：`subtitlestudio_v2.0.0.zip`，Tag `SubtitleStudio_v2.0.0`

## 和 V2 的差别

| 项 | V3 |
| --- | --- |
| 基类 | `app.sdk.plugin._PluginBase` |
| 日志 / 事件 | `app.sdk.logging` / `app.sdk.events` |
| HTTP | `httpx2`，禁止 `alias_httpx()` |
| 身份 | 必须成对 `media_source` + `media_id` |
| 防抖 | `add_plugin_once_job` |
| 事件注册 | 在 `init_plugin` 里挂，导入期不 `register` |

依赖清单同时提供 `requirements.txt` 与 `pyproject.toml`。产品合同与 V2 相同，见 `docs/SubtitleStudio功能使用教学.md`。

## 开发

```bash
cd plugins.v3/subtitlestudio
npm install
npm run build
```

单测：`pytest tests/v3/subtitlestudio`
