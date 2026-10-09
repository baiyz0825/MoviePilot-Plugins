"""配置合同。字段、默认值、帮助文案与设计文档 FIELDS 对齐。

get_form() 在 Vue 模式下仍必须返回默认 dict，供 Config.vue 的 initialConfig 合并。
没有「AI 联动」：本插件自己完成搜索到导出。
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Tuple

from .models import Endpoint, FORMATS, LAYOUTS, STRATEGIES
from .naming import PRESET_IDS, apply_preset, validate_target_languages
from .style import DEFAULT_ASS_STYLE, normalize_ass_style
from .paths import join_multiline_paths, parse_multiline_paths

PLUGIN_ID = "SubtitleStudio"
PLUGIN_NAME = "字幕工坊"
CONFIG_PREFIX = "subtitlestudio_"

TRANSFER_STRATEGY_ALIASES = {
    "先搜后译": "search_then_translate",
    "只搜索": "search_only",
    "只识别翻译": "translate_only",
    "online_then_ai_source": "search_then_translate",
    "online_source_only": "search_only",
    "ai_source_only": "translate_only",
}

LAYOUT_ALIASES = {"单语": "mono", "叠行": "stacked", "分轨": "split"}
STACK_ALIASES = {"主下小上": "main_bottom", "主上小下": "main_top"}
BACKENDS = ("free_first", "free_only", "llm_only")
BACKEND_ALIASES = {
    "免费引擎优先": "free_first",
    "仅免费引擎": "free_only",
    "仅大模型": "llm_only",
}
MT_ENGINES = ("edge", "gtx", "deeplx", "microsoft", "google", "libre")
PROVIDERS = ("subhd", "zimuku", "assrt", "opensubtitles")
SEARCH_LANGS = ("bilingual", "zh-Hans", "zh-Hant", "en")
WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
RAR_MODES = ("none", "container_install", "mapped_binary")
RAR_ALIASES = {"无": "none", "容器内安装": "container_install", "映射二进制": "mapped_binary"}
PREVIEW_SOURCES = ("local_first", "mediaserver", "blackboard")
PREVIEW_SOURCE_ALIASES = {
    "本地文件优先": "local_first",
    "媒体服务器": "mediaserver",
    "仅字幕黑板": "blackboard",
}
OVERWRITE = ("skip", "backup", "overwrite")
OVERWRITE_ALIASES = {"跳过": "skip", "备份后替换": "backup", "直接覆盖": "overwrite"}
ENCODINGS = ("utf-8", "utf-8-sig", "gb18030")
ROLES = ("translate", "judge", "correct", "research")

# 与设计稿 FIELDS 一一对应，Vue 设置页按 pane/group 渲染，控件下方必须带 purpose/after。
FIELDS: List[Dict[str, Any]] = [
    {"pane": "basic", "group": "基础", "key": "enabled", "label": "启用插件", "control": "switch", "default": False,
     "purpose": "插件总闸。决定后台是否接新活。",
     "after": "开：按其它配置监听入库、跑队列、响应工作流。关：不接新任务；已在跑的任务做到当前步骤结束。页面仍可看历史。"},
    {"pane": "basic", "group": "基础", "key": "show_sidebar_nav", "label": "显示侧栏入口", "control": "switch", "default": True,
     "purpose": "在 MoviePilot 侧栏放「字幕工坊」。",
     "after": "开：侧栏直接进四个一级页。关：只能从插件管理打开。不影响已经在跑的任务。"},
    {"pane": "basic", "group": "基础", "key": "send_notify", "label": "任务完成通知", "control": "switch", "default": False,
     "purpose": "走 MoviePilot 已配置的通知渠道（类型：插件）。",
     "after": "开：按下面勾选的结果推送。关（默认）：只在队列页看结果，避免剧集刷屏。渠道在 MoviePilot「通知」里启用，并允许「插件」类型。"},
    {"pane": "basic", "group": "基础", "key": "notify_on", "label": "通知哪些结果", "control": "multi",
     "default": ["success", "failed"],
     "options": [{"title": "成功", "value": "success"}, {"title": "失败", "value": "failed"},
                 {"title": "跳过", "value": "skipped"}, {"title": "取消", "value": "cancelled"}],
     "purpose": "总开关打开后，哪些终态要推送。",
     "after": "默认成功和失败。跳过：门禁没过（已有中字、中文片等）。取消：队列里点了取消。剧集建议只勾失败。"},
    {"pane": "ingest", "group": "入库", "key": "ingest_on_event", "label": "整理完成事件入队", "control": "switch", "default": True,
     "purpose": "听 TransferComplete。整理完成就建任务。这是默认的自动入队方式，比扫盘轻。",
     "after": "开（默认）：新入库按下面策略入队，优先级 P1。关：事件不再入队。和「目录监控」互不影响，可以只开这个。"},
    {"pane": "ingest", "group": "入库", "key": "ingest_on_watch", "label": "媒体目录监控入队", "control": "switch", "default": False,
     "purpose": "单独盯媒体库目录的文件变化。给网盘补拷、手工扔文件、事件丢了用。",
     "after": "开：目录新增/变更可入队。关（默认）：不扫盘。和事件可以同时开；同一 media_source+media_id+路径 5 分钟内只建一次，避免双入队。"},
    {"pane": "ingest", "group": "入库", "key": "watch_paths", "label": "监控的媒体目录", "control": "textarea", "default": "",
     "rows": 4, "placeholder": "/media/movies\n/media/tv",
     "purpose": "目录监控开了以后去哪看。必须是多行输入，不要单行框。",
     "after": "空：跟 MoviePilot 媒体库根走。有填：一行一个容器内绝对路径，用真实换行，不要把 \\n 当两个字符写进框里。"},
    {"pane": "ingest", "group": "入库", "key": "skip_chinese_media", "label": "跳过中文资源", "control": "switch", "default": True,
     "purpose": "片源或音轨已经是中文时，不再做中字。",
     "after": "开：国语片、中文元数据直接标跳过。关：中文片也会搜/译，适合国语片还缺字幕的情况。"},
    {"pane": "ingest", "group": "入库", "key": "skip_existing_chinese", "label": "已有中字则跳过", "control": "switch", "default": True,
     "purpose": "入队前检查目录里是否已有中文字幕。",
     "after": "开：已有中字不建任务。关：仍入队，最终写不写盘看「覆盖策略」。这是门禁，覆盖策略是落盘时的行为。"},
    {"pane": "ingest", "group": "入库", "key": "transfer_strategy", "label": "入库处理策略", "control": "select",
     "default": "search_then_translate",
     "options": [{"title": "先搜后译", "value": "search_then_translate"}, {"title": "只搜索", "value": "search_only"},
                 {"title": "只识别翻译", "value": "translate_only"}],
     "purpose": "自动任务先干什么。",
     "after": "先搜后译：能搜到外挂就不再 ASR。只搜索：只下载，不翻译、不识别。只识别翻译：不去字幕站，用外挂/内嵌/ASR 再译。"},
    {"pane": "ingest", "group": "入库", "key": "trust_transfer_history", "label": "信任整理历史路径", "control": "switch", "default": False,
     "purpose": "刷新媒体目录时少碰磁盘，给网盘/CD2/SMB 用。",
     "after": "开：按整理记录列文件，不逐个探测是否存在，刷新更快。关：更准但慢路径会卡。"},
    {"pane": "ingest", "group": "入库", "key": "strm_enabled", "label": "监控 STRM 目录", "control": "switch", "default": False,
     "purpose": "监视额外的 .strm 目录变化。",
     "after": "开：STRM 新增或变更可入队搜索。关：不管 STRM。STRM 没有本地音轨，永远不做 ASR 和调轴。"},
    {"pane": "ingest", "group": "入库", "key": "strm_paths", "label": "STRM 本地目录", "control": "textarea", "default": "",
     "rows": 5, "placeholder": "/media/strm-movies\n/media/strm-tv",
     "purpose": "告诉插件去哪看 .strm。同样是多行，一行一条路径。",
     "after": "每行一个容器内绝对路径，真实换行。不填等于没开监控。只看文件变化，不解析远程地址。"},
    {"pane": "ingest", "group": "入库", "key": "strm_auto_search", "label": "STRM 变化后自动搜索", "control": "switch", "default": False,
     "purpose": "STRM 变了就搜字幕，不用手点。",
     "after": "开：新文件或内容变化才搜，已处理且没变的不重复打。关：只监控，要你在媒体页点搜索。"},
    {"pane": "search", "group": "搜索", "key": "online_providers", "label": "启用字幕源", "control": "multi",
     "default": ["subhd", "zimuku"],
     "options": [{"title": "SubHD", "value": "subhd"}, {"title": "Zimuku", "value": "zimuku"},
                 {"title": "ASSRT（需 Key）", "value": "assrt"}, {"title": "OpenSubtitles（需 Key）", "value": "opensubtitles"}],
     "purpose": "自动/手动在线搜索去哪些站。",
     "after": "勾上才会请求该站。ASSRT、OpenSubtitles 还要填 Key。这里只决定「到哪找」，不决定写出 SRT 还是 ASS。"},
    {"pane": "search", "group": "搜索", "key": "search_language_priority", "label": "搜索语言优先", "control": "order",
     "default": ["bilingual", "zh-Hans", "zh-Hant", "en"],
     "options": [{"title": "双语", "value": "bilingual"}, {"title": "简中", "value": "zh-Hans"},
                 {"title": "繁中", "value": "zh-Hant"}, {"title": "英", "value": "en"}],
     "purpose": "多条结果时先下载哪条。",
     "after": "只影响选源。特效总开关打开时，标题带「特效/解说」的 ASS 会再提前一档。"},
    {"pane": "search", "group": "搜索", "key": "search_format_priority", "label": "搜索格式优先", "control": "order",
     "default": ["ass", "srt", "ssa", "vtt"],
     "options": [{"title": "ASS", "value": "ass"}, {"title": "SRT", "value": "srt"},
                 {"title": "SSA", "value": "ssa"}, {"title": "VTT", "value": "vtt"}],
     "purpose": "语言差不多时，先要带样式的成品还是纯文本。",
     "after": "这是下载偏好，不是导出格式。导出在「导出包」里另勾。"},
    {"pane": "search", "group": "搜索", "key": "multi_subtitle_mode", "label": "自动多字幕", "control": "select",
     "default": "best",
     "options": [{"title": "只要最好的一条", "value": "best"}, {"title": "中文都留下", "value": "chinese_all"},
                 {"title": "全部留下", "value": "all"}],
     "purpose": "一次搜索留下几条字幕。",
     "after": "最好一条：只留分最高的。中文都留下：简中/繁中/双语都下载。全部：外文也留，磁盘会多几份。"},
    {"pane": "search", "group": "搜索", "key": "prefer_season_pack", "label": "整季包优先", "control": "switch", "default": True,
     "purpose": "剧集先下整季压缩包再拆集。",
     "after": "开：少请求、少验证码，同季共享一个包。关：每集单独搜，站点不稳时更慢。"},
    {"pane": "search", "group": "搜索", "key": "online_use_proxy", "label": "搜索走代理", "control": "switch", "default": False,
     "purpose": "访问字幕站是否走 MoviePilot 代理。",
     "after": "国内站一般关。OpenSubtitles 等海外源打不开再开。和「大模型线路代理」分开。"},
    {"pane": "search", "group": "搜索", "key": "assrt_api_key", "label": "ASSRT API Key", "control": "password", "default": "",
     "purpose": "射手网(伪)的钥匙。", "after": "不填则该源是灰的，不会去请求。Key 只存在插件配置里。"},
    {"pane": "search", "group": "搜索", "key": "opensubtitles_api_key", "label": "OpenSubtitles API Key", "control": "password", "default": "",
     "purpose": "OpenSubtitles 的钥匙。", "after": "不填则该源是灰的。"},
    {"pane": "export", "group": "导出", "key": "export_preset", "label": "导出预设", "control": "preset", "default": "library_zh",
     "purpose": "一键改语言码、是否 default、默认勾哪些格式。",
     "after": "点预设会改语言码和默认勾选，仍可手改。媒体库对齐 MoviePilot 整理：Movie.default.chi.zh-cn.srt。Plex 用 chi/eng。飞牛用 chs。Infuse/网页用 zh-CN 并勾 VTT。旧库用 chi / chi&eng。"},
    {"pane": "export", "group": "导出", "key": "target_languages", "label": "目标语种（有序，1–3）", "control": "lang-order",
     "default": ["zh-Hans"],
     "purpose": "这次要生成哪些文种。最少 1 个，最多 3 个。列表顺序就是主、次、再次。",
     "after": "第 1 位用主字号，第 2 / 3 位用叠行小字。具体大小、颜色在下面「字幕样式」改。不能全空。"},
    {"pane": "export", "group": "导出", "key": "lang_stack", "label": "叠行位置", "control": "select", "default": "main_bottom",
     "options": [{"title": "主下小上", "value": "main_bottom"}, {"title": "主上小下", "value": "main_top"}],
     "purpose": "多个语种叠在同一屏时，大和小谁在上。",
     "after": "主下小上：国内常见「上英下中」。主上小下：接近现网 AutoSub「中上外下」。只勾 1 个语种时此项无效果。"},
    {"pane": "export", "group": "导出", "key": "export_layouts", "label": "版式", "control": "multi", "default": ["mono"],
     "options": [{"title": "单语", "value": "mono"}, {"title": "叠行", "value": "stacked"}, {"title": "分轨", "value": "split"}],
     "purpose": "同一套语种以几种排法出现。",
     "after": "单语：只写主语种。叠行：按顺序把 2–3 种语写进同一文件。分轨：每种语各一份。可同时勾。"},
    {"pane": "export", "group": "导出", "key": "export_formats", "label": "输出格式", "control": "multi", "default": ["srt"],
     "options": [{"title": "SRT", "value": "srt"}, {"title": "ASS", "value": "ass"}, {"title": "VTT", "value": "vtt"}],
     "purpose": "对白写成哪些容器。勾几个写几份。",
     "after": "SRT：电视和媒体库最稳。ASS：对白要样式或换色双语。VTT：Infuse/网页。特效 notes.ass、SDH、ASR 原轨不走这个乘法。"},
    {"pane": "export", "group": "导出", "key": "mark_default", "label": "主中文标 default", "control": "switch", "default": True,
     "purpose": "文件名加 .default，让播放器默认选中文轨。",
     "after": "开：按 MoviePilot 习惯写成 .default.chi.zh-cn，Emby/Jellyfin 会当默认字幕。Plex 和飞牛不认这个标记，对应预设会关掉。"},
    {"pane": "export", "group": "导出", "key": "encoding", "label": "文件编码", "control": "select", "default": "utf-8",
     "options": [{"title": "UTF-8", "value": "utf-8"}, {"title": "UTF-8 BOM", "value": "utf-8-sig"},
                 {"title": "GB18030", "value": "gb18030"}],
     "purpose": "字幕文件怎么落盘。",
     "after": "UTF-8 覆盖绝大多数播放器。电视乱码再试 BOM。GB18030 只给极老设备。"},
    {"pane": "export", "group": "导出", "key": "overwrite_policy", "label": "覆盖策略", "control": "select", "default": "skip",
     "options": [{"title": "跳过", "value": "skip"}, {"title": "备份后替换", "value": "backup"},
                 {"title": "直接覆盖", "value": "overwrite"}],
     "purpose": "目标位置已有同名字幕时怎么办。",
     "after": "跳过：不改已有文件。备份后替换：先复制一份再写。直接覆盖：旧文件没了。"},
    {"pane": "export", "group": "导出", "key": "enable_sdh", "label": "写出 SDH 轨", "control": "switch", "default": False,
     "purpose": "给听不清谁在说话、场外音效用的无障碍轨。",
     "after": "开：另写 .sdh.srt，不和特效注释混。关：不生成。"},
    {"pane": "export", "group": "样式", "key": "ass_style", "label": "字幕样式", "control": "style-editor",
     "default": copy.deepcopy(DEFAULT_ASS_STYLE),
     "purpose": "ASS 对白和顶注的字体、颜色、主/次/再次字号。",
     "after": "改完即时预览。字体颜色只写入 ASS；SRT/VTT 播放器会用自己的字体。勾了叠行会自动补一份 ASS。"},
    {"pane": "effects", "group": "特效", "key": "effects_enabled", "label": "支持特效字幕", "control": "switch", "default": False,
     "purpose": "特效总闸：成品解说、人物背景顶注、关键词补注、Briefing。",
     "after": "开：下面子项生效，搜索抬高「特效/解说」ASS。关：只出干净对白，不写 notes.ass。"},
    {"pane": "effects", "group": "特效", "key": "effects_keep_downloaded", "label": "保留下载的解说 / 屏字", "control": "switch", "default": True,
     "purpose": "成品 ASS 里的顶注和屏字不要压成纯对白。",
     "after": "开：译对白，留原有顶注/屏字样式。关：只抽对白。不重做卡拉 OK，不补字体包。"},
    {"pane": "effects", "group": "特效", "key": "effects_character_briefs", "label": "人物 / 专名背景顶注", "control": "switch", "default": True,
     "purpose": "角色或专名第一次出现时，在画面顶部给一句很短的背景。",
     "after": "开：例如对白出现钢铁侠时，顶栏出「托尼·斯塔克／钢铁侠」。用 ASS \\an8，每屏 1 条，首次出场写一次。"},
    {"pane": "effects", "group": "特效", "key": "effects_keyword_notes", "label": "关键词检索补充注释轨", "control": "switch", "default": True,
     "purpose": "抽文化梗、制度、地名，按片名检索后写成顶栏短注。",
     "after": "开：和人物顶注写在同一条 .notes.ass，但类型分开。解释不上对白轨。"},
    {"pane": "effects", "group": "特效", "key": "effects_briefing", "label": "写入 Briefing 词条", "control": "switch", "default": True,
     "purpose": "把写不下的背景放到工作台词条卡。",
     "after": "开：暂停再看，不上屏、不进媒体目录。不是弹幕。"},
    {"pane": "effects", "group": "特效", "key": "effects_density_minutes", "label": "注释密度（分钟/条）", "control": "number", "default": 2.5,
     "purpose": "限制顶注有多密，避免挡画面。",
     "after": "默认约每 2.5 分钟最多 1 条，每屏 1 条。过长内容进 Briefing。"},
    {"pane": "asr", "group": "识别", "key": "enable_asr", "label": "允许 ASR", "control": "switch", "default": True,
     "purpose": "没有外挂/内嵌时，用 Whisper 听写。",
     "after": "开：搜不到字幕就听写再译。关：搜不到任务失败。和「大模型调用」不是同一组。"},
    {"pane": "asr", "group": "识别", "key": "save_asr_track", "label": "保存 ASR 原轨", "control": "switch", "default": True,
     "purpose": "把听写稿另存一份，方便以后重译。",
     "after": "开且本次真的跑了 Whisper：写 Movie.en.asr.srt，不标 default。用了在线包或外挂则不写。"},
    {"pane": "asr", "group": "识别", "key": "whisper_model", "label": "Whisper 模型", "control": "select", "default": "base",
     "options": [{"title": item, "value": item} for item in WHISPER_MODELS],
     "purpose": "听写用哪档模型。",
     "after": "越大越准、越慢、越占内存。只影响识别，不影响翻译模型。"},
    {"pane": "asr", "group": "识别", "key": "whisper_download_proxy", "label": "代理下载模型", "control": "switch", "default": True,
     "purpose": "第一次拉 Whisper 权重是否走 MP 代理。",
     "after": "开：按环境变量 PROXY 下载。模型已经在本地则无影响。"},
    {"pane": "asr", "group": "识别", "key": "source_preference", "label": "字幕源语言偏好", "control": "select",
     "default": "english_first",
     "options": [{"title": "英文优先", "value": "english_first"}, {"title": "仅英文", "value": "english_only"},
                 {"title": "原音优先", "value": "original_first"}],
     "purpose": "有多条外挂/内嵌时先用哪种。",
     "after": "搜到可用外挂就不会 ASR。"},
    {"pane": "asr", "group": "识别", "key": "auto_detect_language", "label": "自动检测语言", "control": "switch", "default": False,
     "purpose": "听写时语言跟谁走。",
     "after": "开：Whisper 自己认。关：信视频元数据，更稳。"},
    {"pane": "asr", "group": "识别", "key": "min_file_mb", "label": "最小文件体积（MB）", "control": "number", "default": 10,
     "purpose": "太小的视频当样片或损坏，不处理。",
     "after": "小于此值跳过。调低才会处理短片。"},
    {"pane": "asr", "group": "识别", "key": "skip_no_audio", "label": "无音轨则跳过", "control": "switch", "default": True,
     "purpose": "没有音频流就不要 ASR。",
     "after": "STRM 本来就没有音轨。"},
    {"pane": "asr", "group": "识别", "key": "max_segment_seconds", "label": "每段最长秒数", "control": "number", "default": 8,
     "purpose": "一句字幕最多挂多久。",
     "after": "超时切开。不再用「50 字」切中英文。"},
    {"pane": "model", "group": "大模型", "key": "translate_enabled", "label": "启用翻译", "control": "switch", "default": True,
     "purpose": "要不要把外语变成目标语种。",
     "after": "开：按下面的引擎策略翻译。关：只搜/识别/落原字幕。"},
    {"pane": "model", "group": "大模型", "key": "translate_backend", "label": "对白翻译后端", "control": "select",
     "default": "free_first",
     "options": [{"title": "免费引擎优先", "value": "free_first"}, {"title": "仅免费引擎", "value": "free_only"},
                 {"title": "仅大模型", "value": "llm_only"}],
     "purpose": "对白先走谁。人物背景检索仍尽量用大模型。",
     "after": "免费引擎优先：Edge/GTX 等免 Key 先译，失败再走大模型。"},
    {"pane": "model", "group": "大模型", "key": "mt_engines", "label": "免费引擎顺序", "control": "order",
     "default": ["edge", "gtx", "deeplx"],
     "options": [{"title": "Edge", "value": "edge"}, {"title": "Google GTX", "value": "gtx"},
                 {"title": "DeepLX", "value": "deeplx"}, {"title": "微软", "value": "microsoft"},
                 {"title": "Google", "value": "google"}, {"title": "LibreTranslate", "value": "libre"}],
     "purpose": "免 Key 或免费额度的机器翻译，按顺序试。",
     "after": "质检、纠错、人物检索不走这些引擎。"},
    {"pane": "model", "group": "大模型", "key": "openai_fallback_enabled", "label": "自动 fallback", "control": "switch", "default": True,
     "purpose": "主线路失败后是否试列表里下一条。",
     "after": "开：按顺序换线路，再切半、再逐句。关：只用主线路。"},
    {"pane": "model", "group": "大模型", "key": "openai_endpoints", "label": "API 线路", "control": "endpoints", "default": [],
     "purpose": "OpenAI 兼容的 URL / Key / 模型池。",
     "after": "至少一条启用线路。可拉模型列表、测连通、设主线路、排序。没有「AI 联动」——本插件自己调。"},
    {"pane": "model", "group": "大模型", "key": "role_translate", "label": "角色：译", "control": "role", "default": "",
     "purpose": "对白翻译走哪条线路，主消耗。", "after": "不选就用主线路。"},
    {"pane": "model", "group": "大模型", "key": "role_judge", "label": "角色：质检", "control": "role", "default": "",
     "purpose": "抽查对不齐、漏译、乱码。", "after": "不绑则跟主线路。"},
    {"pane": "model", "group": "大模型", "key": "role_correct", "label": "角色：纠错", "control": "role", "default": "",
     "purpose": "工作台里重译单句或邻句。", "after": "关了翻译总开关后这里也没用。"},
    {"pane": "model", "group": "大模型", "key": "role_research", "label": "角色：检索", "control": "role", "default": "",
     "purpose": "特效关键词和 Briefing 的检索压缩。", "after": "特效开了必须有可用线路，否则特效步会失败，对白仍可写。"},
    {"pane": "model", "group": "大模型", "key": "enable_batch", "label": "启用批量翻译", "control": "switch", "default": True,
     "purpose": "多句打成 JSON 一批送出，加快速度。",
     "after": "开：按每批句数走，上下文必须带上。"},
    {"pane": "model", "group": "大模型", "key": "batch_size", "label": "每批句数", "control": "number", "default": 20,
     "purpose": "一批塞多少句。", "after": "建议不超过 30。"},
    {"pane": "model", "group": "大模型", "key": "parallel_workers", "label": "并发线程", "control": "number", "default": 3,
     "purpose": "同时打多少个翻译请求。", "after": "现网默认 10 容易把中转打爆。3 更稳。"},
    {"pane": "model", "group": "大模型", "key": "context_window", "label": "上下文窗口", "control": "number", "default": 5,
     "purpose": "翻译时带上前后几句，避免人称和术语来回变。",
     "after": "现网设置有、批量路径没用。V1 批量必须带上。"},
    {"pane": "model", "group": "大模型", "key": "max_retries", "label": "请求重试次数", "control": "number", "default": 3,
     "purpose": "单次 API 失败再试几次。", "after": "再失败才换线路或降级。"},
    {"pane": "quality", "group": "质量", "key": "format_repair_enabled", "label": "自动修复模型输出格式", "control": "switch", "default": True,
     "purpose": "翻译/质检/纠错/检索返回脏 JSON 时，先修格式再当失败。",
     "after": "开：剥壳、抽出 JSON、按 id 重排、缺句只重跑。修的是格式，不改译文用词。"},
    {"pane": "quality", "group": "质量", "key": "asr_repair_enabled", "label": "自动修复 ASR 轴和分段", "control": "switch", "default": True,
     "purpose": "Whisper 出来的空句、重叠轴、倒序时间、过长句先进 CueGraph 前修一遍。",
     "after": "保存 ASR 原轨时，磁盘上的是修好之后的稿。"},
    {"pane": "quality", "group": "质量", "key": "repair_partial_accept", "label": "好句先收下", "control": "switch", "default": True,
     "purpose": "一批里对上的句子先留下，只补缺的 id。",
     "after": "开：20 句里缺 2 句只重打 2 句。"},
    {"pane": "quality", "group": "质量", "key": "abort_on_high_failure", "label": "失败率过高则整片不写", "control": "switch", "default": False,
     "purpose": "现网 AutoSub：失败率超过 30% 整份字幕不落盘。",
     "after": "关（默认）：好句暂存，坏句标 Issue。一般不要开。"},
    {"pane": "quality", "group": "质量", "key": "write_failure_placeholder", "label": "失败句写入「[翻译失败]」", "control": "switch", "default": False,
     "purpose": "对不上的句子在字幕文件里写什么。",
     "after": "关（默认）：保留原文，工作台标 Issue。"},
    {"pane": "queue", "group": "调轴", "key": "timeline_max_offset", "label": "最大偏移（秒）", "control": "number", "default": 120,
     "purpose": "整轨最多能挪多久。", "after": "默认 120，上限 300。"},
    {"pane": "queue", "group": "调轴", "key": "timeline_min_offset", "label": "最小偏移（秒）", "control": "number", "default": 0.2,
     "purpose": "差多少才值得调。", "after": "小于此值当已经对齐。"},
    {"pane": "queue", "group": "调轴", "key": "timeline_allow_risky", "label": "允许风险偏移", "control": "switch", "default": False,
     "purpose": "是否接受算出来很大的偏移。", "after": "关：超过阈值就跳过调轴。"},
    {"pane": "queue", "group": "调轴", "key": "rar_dependency_mode", "label": "RAR 解包方式", "control": "select", "default": "none",
     "options": [{"title": "无", "value": "none"}, {"title": "容器内安装", "value": "container_install"},
                 {"title": "映射二进制", "value": "mapped_binary"}],
     "purpose": "上传或下载的 RAR 怎么解开。", "after": "无：RAR 包解不了，ZIP/7Z 仍可。"},
    {"pane": "queue", "group": "工作台", "key": "preview_enabled", "label": "工作台在线预览", "control": "switch", "default": True,
     "purpose": "在工作台把当前字幕挂到画面上，看真实字号、叠行和顶注。",
     "after": "开（默认）：上面出预览窗和时间轴。关了不影响导出。"},
    {"pane": "queue", "group": "工作台", "key": "preview_source", "label": "预览画面来源", "control": "select",
     "default": "local_first",
     "options": [{"title": "本地文件优先", "value": "local_first"}, {"title": "媒体服务器", "value": "mediaserver"},
                 {"title": "仅字幕黑板", "value": "blackboard"}],
     "purpose": "预览窗底下那张画面从哪来。",
     "after": "STRM、冷门编码播不了时自动改黑板。"},
    {"pane": "queue", "group": "工作台", "key": "preview_tracks_default", "label": "默认挂载的轨", "control": "multi",
     "default": ["dialogue", "notes"],
     "options": [{"title": "对白", "value": "dialogue"}, {"title": "特效顶注", "value": "notes"},
                 {"title": "叠行", "value": "stacked"}, {"title": "SDH", "value": "sdh"}],
     "purpose": "打开工作台时先挂哪些轨。",
     "after": "工作台里还能临时摘掉，方便对比挡不挡画面。"},
    {"pane": "queue", "group": "队列", "key": "queue_offpeak_enabled", "label": "错峰只跑 P2", "control": "switch", "default": False,
     "purpose": "把低优先级任务留到夜里跑。",
     "after": "开：P2 只在时间窗里跑；P0 人工、P1 入库随时。"},
    {"pane": "queue", "group": "队列", "key": "queue_offpeak_window", "label": "错峰时间窗", "control": "text", "default": "",
     "placeholder": "02:00-07:00",
     "purpose": "P2 允许运行的时段。", "after": "例如 02:00–07:00。不填则错峰开关等于没开。跨天可以。"},
    {"pane": "queue", "group": "扩展", "key": "workflow_enabled", "label": "开放工作流动作", "control": "switch", "default": True,
     "purpose": "给 MoviePilot 工作流/Agent 用。",
     "after": "开：工作流能调这些动作。关：只走插件页面。不委托旧插件。"},
]

PANES = [
    ("basic", "基础", "没有「AI 联动」。本插件自己完成搜索到导出。"),
    ("ingest", "入库与监控", "事件默认开，目录监控默认关。两个稳定开关。STRM 另开，只搜不识别。"),
    ("search", "搜索偏好", "只决定下载哪一条源，不决定最后写成什么文件。"),
    ("export", "导出包", "语种 1–3 个有序。格式和版式多选。ASS 样式可改字体颜色和多行字号。"),
    ("effects", "特效字幕", "默认关。人物背景顶注可单独关。不做卡拉 OK 和飞字。"),
    ("asr", "识别 ASR", "Whisper 听写。和大模型调用分开配。"),
    ("model", "翻译与大模型", "对白可走免费引擎；质检/检索仍走大模型。"),
    ("quality", "质量与格式修复", "先修模型/识别输出，再降级。默认不开现网「失败率过高整片不写」。"),
    ("queue", "调轴与队列", "队列始终落库。工作台默认可预览、可看时间轴。插队只动未运行的。"),
]

DEFAULT_CONFIG: Dict[str, Any] = {field["key"]: copy.deepcopy(field["default"]) for field in FIELDS}
DEFAULT_CONFIG.update({
    "subhd_url": "https://subhd.tv",
    "zimuku_url": "https://zmk.pw",
    "assrt_api_url": "https://api.assrt.net",
    "opensubtitles_api_url": "https://api.opensubtitles.com/api/v1",
    "deeplx_url": "",
    "libre_url": "",
    "microsoft_key": "",
    "google_key": "",
    "rar_tool_path": "/usr/bin/unar",
})


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "开"}
    return bool(value)


def _as_number(value: Any, default: float, integer: bool = False) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return int(default) if integer else default
    if integer:
        return int(number)
    return number


def _pick(value: Any, allowed: Tuple[str, ...], default: str, aliases: Dict[str, str] | None = None) -> str:
    text = str(value or "").strip()
    if aliases and text in aliases:
        text = aliases[text]
    return text if text in allowed else default


def _pick_list(value: Any, allowed: Tuple[str, ...], default: List[str]) -> List[str]:
    items = value if isinstance(value, (list, tuple)) else (str(value).split(",") if value else default)
    result = []
    for item in items:
        text = str(item).strip()
        if text in LAYOUT_ALIASES:
            text = LAYOUT_ALIASES[text]
        if text.startswith("."):
            text = text[1:]
        if text in allowed and text not in result:
            result.append(text)
    return result or list(default)


def normalize_endpoints(value: Any) -> List[Dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    endpoints: List[Endpoint] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        endpoint = Endpoint.from_dict(item)
        if not endpoint.endpoint_id:
            endpoint.endpoint_id = uuid.uuid4().hex[:12]
        endpoints.append(endpoint)
    if endpoints and not any(item.primary for item in endpoints):
        for item in endpoints:
            if item.enabled:
                item.primary = True
                break
    return [item.to_dict() for item in endpoints]


def normalize_plugin_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = dict(config or {})
    normalized = copy.deepcopy(DEFAULT_CONFIG)
    normalized.update({key: raw[key] for key in raw if key in normalized or key in DEFAULT_CONFIG or True})
    for field in FIELDS:
        key = field["key"]
        default = field["default"]
        control = field["control"]
        incoming = raw.get(key, default)
        if control == "switch":
            normalized[key] = _as_bool(incoming, bool(default))
        elif control == "number":
            normalized[key] = _as_number(incoming, float(default), integer=isinstance(default, int))
        elif control == "textarea":
            normalized[key] = join_multiline_paths(parse_multiline_paths(incoming))
        elif key == "target_languages":
            normalized[key] = validate_target_languages(incoming)
        elif key == "openai_endpoints":
            normalized[key] = normalize_endpoints(incoming)
        elif key == "ass_style":
            normalized[key] = normalize_ass_style(incoming)
        elif control in {"multi", "order"}:
            options = tuple(item["value"] for item in field.get("options") or [])
            normalized[key] = _pick_list(incoming, options, list(default))
        elif control == "select":
            options = tuple(item["value"] for item in field.get("options") or [])
            aliases = {}
            if key == "transfer_strategy":
                aliases = TRANSFER_STRATEGY_ALIASES
            elif key == "lang_stack":
                aliases = STACK_ALIASES
            elif key == "translate_backend":
                aliases = BACKEND_ALIASES
            elif key == "preview_source":
                aliases = PREVIEW_SOURCE_ALIASES
            elif key == "overwrite_policy":
                aliases = OVERWRITE_ALIASES
            elif key == "rar_dependency_mode":
                aliases = RAR_ALIASES
            normalized[key] = _pick(incoming, options, str(default), aliases)
        elif control == "preset":
            normalized[key] = incoming if incoming in PRESET_IDS else "library_zh"
        else:
            normalized[key] = incoming if incoming is not None else default

    if normalized["export_preset"] in {"plex", "fnos"}:
        normalized["mark_default"] = False
    normalized["target_languages"] = validate_target_languages(normalized.get("target_languages"))
    normalized["export_layouts"] = _pick_list(normalized.get("export_layouts"), tuple(LAYOUTS), ["mono"])
    normalized["export_formats"] = _pick_list(normalized.get("export_formats"), FORMATS, ["srt"])
    normalized["timeline_max_offset"] = min(300, max(1, float(normalized.get("timeline_max_offset") or 120)))
    normalized["timeline_min_offset"] = max(0, float(normalized.get("timeline_min_offset") or 0.2))
    normalized["parallel_workers"] = max(1, min(8, int(normalized.get("parallel_workers") or 3)))
    normalized["batch_size"] = max(1, min(40, int(normalized.get("batch_size") or 20)))
    normalized["watch_paths"] = join_multiline_paths(parse_multiline_paths(normalized.get("watch_paths")))
    normalized["strm_paths"] = join_multiline_paths(parse_multiline_paths(normalized.get("strm_paths")))
    for extra in ("subhd_url", "zimuku_url", "assrt_api_url", "opensubtitles_api_url", "deeplx_url", "libre_url",
                  "microsoft_key", "google_key", "rar_tool_path", "assrt_api_key", "opensubtitles_api_key"):
        if extra in raw:
            normalized[extra] = str(raw.get(extra) or "")
        else:
            normalized.setdefault(extra, DEFAULT_CONFIG.get(extra, ""))
    return normalized


def default_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def apply_export_preset(config: Dict[str, Any], preset: str) -> Dict[str, Any]:
    return normalize_plugin_config(apply_preset(config, preset))


def field_map() -> Dict[str, Dict[str, Any]]:
    return {item["key"]: item for item in FIELDS}


def build_config_form() -> Tuple[List[dict], Dict[str, Any]]:
    """Vue 模式仍要返回默认模型；表单只给插件管理一个提示。完整设置在 AppPage。"""
    form = [
        {
            "component": "VForm",
            "content": [
                {
                    "component": "VRow",
                    "content": [
                        {
                            "component": "VCol",
                            "props": {"cols": 12, "md": 4},
                            "content": [{"component": "VSwitch", "props": {"model": "enabled", "label": "启用插件"}}],
                        },
                        {
                            "component": "VCol",
                            "props": {"cols": 12, "md": 4},
                            "content": [{"component": "VSwitch", "props": {"model": "show_sidebar_nav", "label": "显示侧栏入口"}}],
                        },
                        {
                            "component": "VCol",
                            "props": {"cols": 12, "md": 4},
                            "content": [{"component": "VSwitch", "props": {"model": "ingest_on_event", "label": "整理完成事件入队"}}],
                        },
                    ],
                },
                {
                    "component": "VRow",
                    "content": [
                        {
                            "component": "VCol",
                            "props": {"cols": 12},
                            "content": [
                                {
                                    "component": "VAlert",
                                    "props": {
                                        "type": "info",
                                        "variant": "tonal",
                                        "text": "完整四个一级页（媒体 / 队列 / 工作台 / 设置）在侧栏「字幕工坊」。设置页字段与帮助文案以设计文档 FIELDS 为准。",
                                    },
                                }
                            ],
                        }
                    ],
                },
            ],
        }
    ]
    return form, default_config()
