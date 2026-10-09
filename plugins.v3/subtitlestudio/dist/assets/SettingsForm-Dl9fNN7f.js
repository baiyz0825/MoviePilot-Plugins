import { importShared } from './__federation_fn_import-JrT3xvdd.js';

const {inject} = await importShared('vue');


function fallbackToast() {
  return {
    success: message => console.info(message),
    error: message => console.error(message),
    info: message => console.info(message),
  }
}

function useHostInjects() {
  const toast = inject('moviepilot:toast', fallbackToast());
  const dialog = inject('moviepilot:dialog', null);
  const confirm = inject('moviepilot:confirm', async () => window.confirm('确定？'));
  const nativeSubscribe = inject('moviepilot:nativeSubscribe', null);
  return { toast, dialog, confirm, nativeSubscribe }
}

const {onBeforeUnmount,onMounted,ref: ref$1} = await importShared('vue');


const MOBILE_QUERY = '(max-width: 899px)';

function useMobileViewport(query = MOBILE_QUERY) {
  const isMobileViewport = ref$1(false);
  let mediaQueryList = null;

  function syncViewport(event) {
    isMobileViewport.value = Boolean(event?.matches);
  }

  onMounted(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    mediaQueryList = window.matchMedia(query);
    syncViewport(mediaQueryList);
    if (typeof mediaQueryList.addEventListener === 'function') {
      mediaQueryList.addEventListener('change', syncViewport);
      return
    }
    mediaQueryList.addListener(syncViewport);
  });

  onBeforeUnmount(() => {
    if (!mediaQueryList) return
    if (typeof mediaQueryList.removeEventListener === 'function') {
      mediaQueryList.removeEventListener('change', syncViewport);
      return
    }
    mediaQueryList.removeListener(syncViewport);
  });

  return isMobileViewport
}

const LANGS = [
  { title: '简中', value: 'zh-Hans' },
  { title: '繁中', value: 'zh-Hant' },
  { title: '英文', value: 'en' },
  { title: '日文', value: 'ja' },
  { title: '韩文', value: 'ko' },
];

const PRESETS = [
  { value: 'library_zh', title: '媒体库 / MoviePilot', hint: 'default.chi.zh-cn · 对齐整理记录' },
  { value: 'plex', title: 'Plex', hint: 'chi / eng · 不标 default' },
  { value: 'fnos', title: '飞牛影视', hint: 'chs · 飞牛不认 zh-Hans' },
  { value: 'web', title: 'Infuse / 网页', hint: 'zh-CN · 再勾 VTT' },
  { value: 'legacy', title: '兼容旧库', hint: 'chi / chi&eng' },
];

const LANG_CODES = {
  'zh-Hans': { library_zh: 'chi.zh-cn', plex: 'chi', web: 'zh-CN', fnos: 'chs', legacy: 'chi' },
  'zh-Hant': { library_zh: 'zh-tw', plex: 'cht', web: 'zh-TW', fnos: 'cht', legacy: 'cht' },
  en: { library_zh: 'eng', plex: 'eng', web: 'en', fnos: 'eng', legacy: 'eng' },
  ja: { library_zh: 'ja', plex: 'jpn', web: 'ja', fnos: 'jpn', legacy: 'jpn' },
  ko: { library_zh: 'ko', plex: 'kor', web: 'ko', fnos: 'kor', legacy: 'kor' },
};

const PANES = [
  ['basic', '基础'],
  ['ingest', '入库与监控'],
  ['search', '搜索偏好'],
  ['export', '导出包'],
  ['effects', '特效字幕'],
  ['asr', '识别 ASR'],
  ['model', '翻译与大模型'],
  ['quality', '质量与格式修复'],
  ['queue', '调轴与队列'],
];

function langLabel(value) {
  return LANGS.find(item => item.value === value)?.title || value
}

const STYLE_FONTS = [
  'Arial',
  'Microsoft YaHei',
  'PingFang SC',
  'Source Han Sans SC',
  'Noto Sans CJK SC',
  'SimHei',
  'Helvetica',
];

const STYLE_LOOKS = [
  { title: '白字黑边', patch: { primary_color: '#FFFFFF', outline_color: '#000000', outline: 2, shadow: 2 } },
  { title: '黄字黑边', patch: { primary_color: '#FFE566', outline_color: '#000000', outline: 3, shadow: 1 } },
  { title: '白字软影', patch: { primary_color: '#FFFFFF', outline_color: '#000000', outline: 1, shadow: 4 } },
];

const DEFAULT_ASS_STYLE = {
  font_name: 'Arial',
  primary_color: '#FFFFFF',
  outline_color: '#000000',
  back_color: '#000000',
  outline: 2,
  shadow: 2,
  bold: false,
  italic: false,
  sizes: [22, 17, 14],
  note_size: 16,
  note_color: '#B4E0FF',
  margin_v: 20,
  line_gap: 28,
};

function normalizeStyle(value) {
  const raw = value && typeof value === 'object' ? value : {};
  const sizes = [0, 1, 2].map(index => {
    const next = Number((raw.sizes || DEFAULT_ASS_STYLE.sizes)[index] ?? DEFAULT_ASS_STYLE.sizes[index]);
    return Math.max(8, Math.min(72, Number.isFinite(next) ? next : DEFAULT_ASS_STYLE.sizes[index]))
  });
  return {
    ...DEFAULT_ASS_STYLE,
    ...raw,
    font_name: String(raw.font_name || DEFAULT_ASS_STYLE.font_name).replace(/[,\\]/g, '') || 'Arial',
    sizes,
    outline: Math.max(0, Math.min(8, Number(raw.outline ?? 2))),
    shadow: Math.max(0, Math.min(8, Number(raw.shadow ?? 2))),
    note_size: Math.max(8, Math.min(48, Number(raw.note_size ?? 16))),
    margin_v: Math.max(0, Math.min(160, Number(raw.margin_v ?? 20))),
    line_gap: Math.max(8, Math.min(80, Number(raw.line_gap ?? 28))),
    bold: Boolean(raw.bold),
    italic: Boolean(raw.italic),
  }
}

function sizeForRank(index, style) {
  const sizes = normalizeStyle(style).sizes;
  return sizes[index] ?? sizes[sizes.length - 1] ?? 14
}

function overlayCss(style, rank = 0, kind = 'dialogue') {
  const spec = normalizeStyle(style);
  const size = kind === 'note' ? spec.note_size : sizeForRank(rank, spec);
  const color = kind === 'note' ? spec.note_color : spec.primary_color;
  const outline = `${spec.outline}px ${spec.outline_color}`;
  return {
    fontFamily: `'${spec.font_name}', sans-serif`,
    fontSize: `${size}px`,
    color,
    fontWeight: spec.bold ? '700' : '400',
    fontStyle: spec.italic ? 'italic' : 'normal',
    textShadow: `0 0 ${spec.outline}px ${spec.outline_color}, 0 ${spec.shadow}px ${spec.shadow}px rgba(0,0,0,.65), -1px 0 ${outline}, 1px 0 ${outline}, 0 -1px ${outline}, 0 1px ${outline}`,
  }
}

function langCode(lang, preset) {
  return LANG_CODES[lang]?.[preset] || LANG_CODES[lang]?.library_zh || lang
}

function buildFilename(stem, langs, ext, { title = '', flags = [], flagFirst = false } = {}) {
  const parts = [stem];
  const langPart = langs.filter(Boolean).join('.');
  const flagParts = flags.filter(Boolean);
  if (flagFirst) {
    parts.push(...flagParts);
    if (langPart) parts.push(langPart);
    if (title) parts.push(title);
  } else {
    if (langPart) parts.push(langPart);
    if (title) parts.push(title);
    parts.push(...flagParts);
  }
  return `${parts.join('.')}.${ext}`
}

function applyPreset(config, preset) {
  const next = { ...config, export_preset: preset };
  if (preset === 'plex' || preset === 'fnos') next.mark_default = false;
  if (preset === 'library_zh' || preset === 'web') next.mark_default = true;
  const formats = new Set(next.export_formats || ['srt']);
  formats.add('srt');
  if (preset === 'web') formats.add('vtt');
  next.export_formats = Array.from(formats);
  return next
}

function exportPreview(config, stem = 'Movie') {
  const preset = config.export_preset || 'library_zh';
  const langs = (config.target_languages || ['zh-Hans']).slice(0, 3);
  const layouts = config.export_layouts || ['mono'];
  const formats = config.export_formats || ['srt'];
  const markDefault = Boolean(config.mark_default) && preset !== 'plex' && preset !== 'fnos';
  const flagFirst = preset === 'library_zh';
  const files = [];
  for (const layout of layouts) {
    for (const fmt of formats) {
      if (layout === 'mono') {
        const flags = markDefault && langs[0]?.startsWith('zh') ? ['default'] : [];
        files.push(buildFilename(stem, [langCode(langs[0], preset)], fmt, { flags, flagFirst }));
      }
      if (layout === 'stacked' && langs.length > 1) {
        const codes = langs.map(item => langCode(item, preset));
        if (preset === 'legacy' || preset === 'fnos') {
          files.push(buildFilename(stem, [codes.join('&')], fmt, { flagFirst }));
        } else {
          files.push(buildFilename(stem, [codes[0]], fmt, { title: 'bilingual', flagFirst }));
        }
      }
      if (layout === 'split') {
        langs.forEach((lang, index) => {
          const flags = markDefault && index === 0 && lang.startsWith('zh') ? ['default'] : [];
          files.push(buildFilename(stem, [langCode(lang, preset)], fmt, { flags, flagFirst }));
        });
      }
    }
  }
  if (config.effects_enabled) files.push(buildFilename(stem, [langCode(langs[0], preset)], 'ass', { title: 'notes', flagFirst }));
  if (config.enable_sdh) files.push(buildFilename(stem, [langCode(langs[0], preset)], 'srt', { title: 'sdh', flagFirst }));
  if (config.save_asr_track) files.push(buildFilename(stem, [langCode('en', preset)], 'srt', { title: 'asr', flagFirst }));
  return files
}

const {unref:_unref,renderList:_renderList,Fragment:_Fragment,openBlock:_openBlock,createElementBlock:_createElementBlock,toDisplayString:_toDisplayString,createTextVNode:_createTextVNode,resolveComponent:_resolveComponent,withCtx:_withCtx,createBlock:_createBlock,createCommentVNode:_createCommentVNode,createElementVNode:_createElementVNode,createVNode:_createVNode,normalizeStyle:_normalizeStyle,normalizeClass:_normalizeClass} = await importShared('vue');


const _hoisted_1 = { class: "ss-settings" };
const _hoisted_2 = { class: "mt-4" };
const _hoisted_3 = { key: 6 };
const _hoisted_4 = { class: "text-body-2 mb-2" };
const _hoisted_5 = { key: 7 };
const _hoisted_6 = { class: "text-body-2 mb-2" };
const _hoisted_7 = { key: 8 };
const _hoisted_8 = { class: "text-body-2 mb-2" };
const _hoisted_9 = { key: 9 };
const _hoisted_10 = { class: "d-flex align-center mb-2" };
const _hoisted_11 = { class: "text-body-2" };
const _hoisted_12 = { class: "d-flex flex-wrap ga-2" };
const _hoisted_13 = { class: "d-flex ga-2 mt-2" };
const _hoisted_14 = {
  key: 10,
  class: "ss-style-editor"
};
const _hoisted_15 = { class: "text-body-2 mb-2" };
const _hoisted_16 = { class: "d-flex flex-wrap ga-2 mb-3" };
const _hoisted_17 = { class: "ss-color-field" };
const _hoisted_18 = ["value"];
const _hoisted_19 = { class: "ss-color-field" };
const _hoisted_20 = ["value"];
const _hoisted_21 = { class: "ss-color-field" };
const _hoisted_22 = ["value"];
const _hoisted_23 = { class: "d-flex flex-wrap ga-2" };
const _hoisted_24 = { class: "ss-help" };

const {computed,ref,watch} = await importShared('vue');


const _sfc_main = {
  __name: 'SettingsForm',
  props: {
  modelValue: { type: Object, required: true },
  fields: { type: Array, default: () => [] },
  mobile: { type: Boolean, default: false },
},
  emits: ['update:modelValue', 'test-endpoint', 'list-models', 'dirty'],
  setup(__props, { emit: __emit }) {

const props = __props;
const emit = __emit;

const pane = ref('basic');
const config = computed({
  get: () => props.modelValue,
  set: value => emit('update:modelValue', value),
});

watch(config, () => emit('dirty', true), { deep: true });

const grouped = computed(() => {
  const rows = props.fields.length ? props.fields : fallbackFields();
  return rows.filter(item => {
    if (item.pane !== pane.value || item.mock === 'skip') return false
    if (item.key === 'notify_on' && !config.value.send_notify) return false
    return true
  })
});

function setField(key, value) {
  config.value = { ...config.value, [key]: value };
}

function toggleList(key, value) {
  const current = [...(config.value[key] || [])];
  const index = current.indexOf(value);
  if (index >= 0) {
    if (key === 'target_languages' && current.length <= 1) return
    current.splice(index, 1);
  } else if (key === 'target_languages' && current.length >= 3) {
    return
  } else {
    current.push(value);
  }
  setField(key, current);
}

function moveLang(index, delta) {
  const langs = [...(config.value.target_languages || [])];
  const next = index + delta;
  if (next < 0 || next >= langs.length) return
  ;[langs[index], langs[next]] = [langs[next], langs[index]];
  setField('target_languages', langs);
}

function usePreset(value) {
  config.value = applyPreset(config.value, value);
}

const assStyle = computed(() => normalizeStyle(config.value.ass_style));

function setStyle(patch) {
  const next = { ...assStyle.value, ...patch };
  if (Array.isArray(patch.sizes)) next.sizes = [...patch.sizes];
  setField('ass_style', next);
}

function setStyleSize(index, value) {
  const sizes = [...assStyle.value.sizes];
  sizes[index] = Number(value);
  setStyle({ sizes });
}

function useStyleLook(look) {
  setStyle(look.patch);
}

function ensureAssFormat() {
  const formats = [...(config.value.export_formats || [])];
  if (!formats.includes('ass')) formats.push('ass');
  setField('export_formats', formats);
}

const stylePreview = computed(() => {
  const stack = config.value.lang_stack || 'main_bottom';
  const langs = config.value.target_languages || ['zh-Hans'];
  return {
    stack,
    langs,
    main: overlayCss(assStyle.value, 0),
    second: overlayCss(assStyle.value, 1),
    third: overlayCss(assStyle.value, 2),
    note: overlayCss(assStyle.value, 0, 'note'),
    mainBottom: `${assStyle.value.margin_v}px`,
    secondBottom: `${assStyle.value.margin_v + assStyle.value.line_gap}px`,
    thirdBottom: `${assStyle.value.margin_v + assStyle.value.line_gap * 2}px`,
    noteTop: `${Math.max(12, assStyle.value.margin_v - 4)}px`,
  }
});

function addEndpoint() {
  const rows = [...(config.value.openai_endpoints || [])];
  rows.push({
    endpoint_id: `ep-${Date.now()}`,
    name: `线路 ${rows.length + 1}`,
    api_url: '',
    api_key: '',
    model: '',
    enabled: true,
    primary: rows.length === 0,
    use_proxy: false,
    compatible: false,
  });
  setField('openai_endpoints', rows);
}

function updateEndpoint(index, patch) {
  const rows = (config.value.openai_endpoints || []).map((item, idx) => (idx === index ? { ...item, ...patch } : item));
  if (patch.primary) {
    rows.forEach((item, idx) => {
      item.primary = idx === index;
    });
  }
  setField('openai_endpoints', rows);
}

function fallbackFields() {
  return [
    { pane: 'basic', key: 'enabled', label: '启用插件', control: 'switch', purpose: '插件总闸。', after: '开：接新活。关：不接新任务。' },
    { pane: 'basic', key: 'show_sidebar_nav', label: '显示侧栏入口', control: 'switch', purpose: '侧栏放字幕工坊。', after: '关了只能从插件管理进。' },
    { pane: 'basic', key: 'send_notify', label: '任务完成通知', control: 'switch', purpose: '走 MoviePilot 已配置的通知渠道（类型：插件）。', after: '默认关，避免剧集刷屏。渠道在 MoviePilot「通知」里启用。' },
    { pane: 'basic', key: 'notify_on', label: '通知哪些结果', control: 'multi', options: [
      { title: '成功', value: 'success' }, { title: '失败', value: 'failed' }, { title: '跳过', value: 'skipped' }, { title: '取消', value: 'cancelled' },
    ], purpose: '总开关打开后，哪些终态要推送。', after: '默认成功和失败。剧集建议只勾失败。' },
    { pane: 'ingest', key: 'ingest_on_event', label: '整理完成事件入队', control: 'switch', purpose: '听 TransferComplete。', after: '默认开，和目录监控互不影响。' },
    { pane: 'ingest', key: 'ingest_on_watch', label: '媒体目录监控入队', control: 'switch', purpose: '盯媒体库目录。', after: '默认关。' },
    { pane: 'ingest', key: 'watch_paths', label: '监控的媒体目录', control: 'textarea', rows: 4, purpose: '一行一条路径。', after: '真实换行，不要写字面 \\n。', placeholder: '/media/movies\n/media/tv' },
    { pane: 'ingest', key: 'skip_chinese_media', label: '跳过中文资源', control: 'switch' },
    { pane: 'ingest', key: 'skip_existing_chinese', label: '已有中字则跳过', control: 'switch' },
    { pane: 'ingest', key: 'transfer_strategy', label: '入库处理策略', control: 'select', options: [
      { title: '先搜后译', value: 'search_then_translate' }, { title: '只搜索', value: 'search_only' }, { title: '只识别翻译', value: 'translate_only' },
    ] },
    { pane: 'ingest', key: 'trust_transfer_history', label: '信任整理历史路径', control: 'switch' },
    { pane: 'ingest', key: 'strm_enabled', label: '监控 STRM 目录', control: 'switch' },
    { pane: 'ingest', key: 'strm_paths', label: 'STRM 本地目录', control: 'textarea', rows: 5, placeholder: '/media/strm-movies\n/media/strm-tv' },
    { pane: 'ingest', key: 'strm_auto_search', label: 'STRM 变化后自动搜索', control: 'switch' },
    { pane: 'search', key: 'online_providers', label: '启用字幕源', control: 'multi', options: [
      { title: 'SubHD', value: 'subhd' }, { title: 'Zimuku', value: 'zimuku' }, { title: 'ASSRT', value: 'assrt' }, { title: 'OpenSubtitles', value: 'opensubtitles' },
    ] },
    { pane: 'search', key: 'prefer_season_pack', label: '整季包优先', control: 'switch' },
    { pane: 'search', key: 'online_use_proxy', label: '搜索走代理', control: 'switch' },
    { pane: 'search', key: 'assrt_api_key', label: 'ASSRT API Key', control: 'password' },
    { pane: 'search', key: 'opensubtitles_api_key', label: 'OpenSubtitles API Key', control: 'password' },
    { pane: 'export', key: 'export_preset', label: '导出预设', control: 'preset' },
    { pane: 'export', key: 'target_languages', label: '目标语种（有序，1–3）', control: 'lang-order' },
    { pane: 'export', key: 'lang_stack', label: '叠行位置', control: 'select', options: [
      { title: '主下小上', value: 'main_bottom' }, { title: '主上小下', value: 'main_top' },
    ] },
    { pane: 'export', key: 'export_layouts', label: '版式', control: 'multi', options: [
      { title: '单语', value: 'mono' }, { title: '叠行', value: 'stacked' }, { title: '分轨', value: 'split' },
    ] },
    { pane: 'export', key: 'export_formats', label: '输出格式', control: 'multi', options: [
      { title: 'SRT', value: 'srt' }, { title: 'ASS', value: 'ass' }, { title: 'VTT', value: 'vtt' },
    ] },
    { pane: 'export', key: 'mark_default', label: '主中文标 default', control: 'switch' },
    { pane: 'export', key: 'encoding', label: '文件编码', control: 'select', options: [
      { title: 'UTF-8', value: 'utf-8' }, { title: 'UTF-8 BOM', value: 'utf-8-sig' }, { title: 'GB18030', value: 'gb18030' },
    ] },
    { pane: 'export', key: 'overwrite_policy', label: '覆盖策略', control: 'select', options: [
      { title: '跳过', value: 'skip' }, { title: '备份后替换', value: 'backup' }, { title: '直接覆盖', value: 'overwrite' },
    ] },
    { pane: 'export', key: 'enable_sdh', label: '写出 SDH 轨', control: 'switch' },
    { pane: 'export', key: 'ass_style', label: '字幕样式', control: 'style-editor', purpose: 'ASS 对白和顶注的字体、颜色、主/次字号。', after: '改完即时预览。字体颜色只写入 ASS。' },
    { pane: 'effects', key: 'effects_enabled', label: '支持特效字幕', control: 'switch' },
    { pane: 'effects', key: 'effects_keep_downloaded', label: '保留下载的解说 / 屏字', control: 'switch' },
    { pane: 'effects', key: 'effects_character_briefs', label: '人物 / 专名背景顶注', control: 'switch' },
    { pane: 'effects', key: 'effects_keyword_notes', label: '关键词检索补充注释轨', control: 'switch' },
    { pane: 'effects', key: 'effects_briefing', label: '写入 Briefing 词条', control: 'switch' },
    { pane: 'effects', key: 'effects_density_minutes', label: '注释密度（分钟/条）', control: 'number' },
    { pane: 'asr', key: 'enable_asr', label: '允许 ASR', control: 'switch' },
    { pane: 'asr', key: 'save_asr_track', label: '保存 ASR 原轨', control: 'switch' },
    { pane: 'asr', key: 'whisper_model', label: 'Whisper 模型', control: 'select', options: ['tiny', 'base', 'small', 'medium', 'large-v3', 'large-v3-turbo'].map(item => ({ title: item, value: item })) },
    { pane: 'asr', key: 'min_file_mb', label: '最小文件体积（MB）', control: 'number' },
    { pane: 'model', key: 'translate_enabled', label: '启用翻译', control: 'switch' },
    { pane: 'model', key: 'translate_backend', label: '对白翻译后端', control: 'select', options: [
      { title: '免费引擎优先', value: 'free_first' }, { title: '仅免费引擎', value: 'free_only' }, { title: '仅大模型', value: 'llm_only' },
    ] },
    { pane: 'model', key: 'mt_engines', label: '免费引擎顺序', control: 'multi', options: [
      { title: 'Edge', value: 'edge' }, { title: 'Google GTX', value: 'gtx' }, { title: 'DeepLX', value: 'deeplx' },
    ] },
    { pane: 'model', key: 'openai_fallback_enabled', label: '自动 fallback', control: 'switch' },
    { pane: 'model', key: 'openai_endpoints', label: 'API 线路', control: 'endpoints' },
    { pane: 'model', key: 'enable_batch', label: '启用批量翻译', control: 'switch' },
    { pane: 'model', key: 'batch_size', label: '每批句数', control: 'number' },
    { pane: 'model', key: 'parallel_workers', label: '并发线程', control: 'number' },
    { pane: 'model', key: 'context_window', label: '上下文窗口', control: 'number' },
    { pane: 'quality', key: 'format_repair_enabled', label: '自动修复模型输出格式', control: 'switch' },
    { pane: 'quality', key: 'asr_repair_enabled', label: '自动修复 ASR 轴和分段', control: 'switch' },
    { pane: 'quality', key: 'repair_partial_accept', label: '好句先收下', control: 'switch' },
    { pane: 'quality', key: 'abort_on_high_failure', label: '失败率过高则整片不写', control: 'switch' },
    { pane: 'quality', key: 'write_failure_placeholder', label: '失败句写入「[翻译失败]」', control: 'switch' },
    { pane: 'queue', key: 'preview_enabled', label: '工作台在线预览', control: 'switch' },
    { pane: 'queue', key: 'preview_source', label: '预览画面来源', control: 'select', options: [
      { title: '本地文件优先', value: 'local_first' }, { title: '媒体服务器', value: 'mediaserver' }, { title: '仅字幕黑板', value: 'blackboard' },
    ] },
    { pane: 'queue', key: 'queue_offpeak_enabled', label: '错峰只跑 P2', control: 'switch' },
    { pane: 'queue', key: 'queue_offpeak_window', label: '错峰时间窗', control: 'text', placeholder: '02:00-07:00' },
    { pane: 'queue', key: 'workflow_enabled', label: '开放工作流动作', control: 'switch' },
  ]
}

const files = computed(() => exportPreview(config.value));

return (_ctx, _cache) => {
  const _component_VBtn = _resolveComponent("VBtn");
  const _component_VBtnToggle = _resolveComponent("VBtnToggle");
  const _component_VExpansionPanel = _resolveComponent("VExpansionPanel");
  const _component_VExpansionPanels = _resolveComponent("VExpansionPanels");
  const _component_VSwitch = _resolveComponent("VSwitch");
  const _component_VTextarea = _resolveComponent("VTextarea");
  const _component_VTextField = _resolveComponent("VTextField");
  const _component_VSelect = _resolveComponent("VSelect");
  const _component_VChip = _resolveComponent("VChip");
  const _component_VCardTitle = _resolveComponent("VCardTitle");
  const _component_VCardText = _resolveComponent("VCardText");
  const _component_VCard = _resolveComponent("VCard");
  const _component_VCol = _resolveComponent("VCol");
  const _component_VRow = _resolveComponent("VRow");
  const _component_VSpacer = _resolveComponent("VSpacer");
  const _component_VAlert = _resolveComponent("VAlert");

  return (_openBlock(), _createElementBlock("div", _hoisted_1, [
    (!__props.mobile)
      ? (_openBlock(), _createBlock(_component_VBtnToggle, {
          key: 0,
          modelValue: pane.value,
          "onUpdate:modelValue": _cache[0] || (_cache[0] = $event => ((pane).value = $event)),
          mandatory: "",
          density: "comfortable",
          class: "mb-4 flex-wrap"
        }, {
          default: _withCtx(() => [
            (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(_unref(PANES), (item) => {
              return (_openBlock(), _createBlock(_component_VBtn, {
                key: item[0],
                value: item[0],
                class: "ss-touch"
              }, {
                default: _withCtx(() => [
                  _createTextVNode(_toDisplayString(item[1]), 1)
                ]),
                _: 2
              }, 1032, ["value"]))
            }), 128))
          ]),
          _: 1
        }, 8, ["modelValue"]))
      : (_openBlock(), _createBlock(_component_VExpansionPanels, {
          key: 1,
          modelValue: pane.value,
          "onUpdate:modelValue": _cache[1] || (_cache[1] = $event => ((pane).value = $event))
        }, {
          default: _withCtx(() => [
            (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(_unref(PANES), (item) => {
              return (_openBlock(), _createBlock(_component_VExpansionPanel, {
                key: item[0],
                value: item[0],
                title: item[1]
              }, null, 8, ["value", "title"]))
            }), 128))
          ]),
          _: 1
        }, 8, ["modelValue"])),
    _createElementVNode("div", _hoisted_2, [
      (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(grouped.value, (field) => {
        return (_openBlock(), _createElementBlock("div", {
          key: field.key,
          class: "mb-5"
        }, [
          (field.control === 'switch')
            ? (_openBlock(), _createBlock(_component_VSwitch, {
                key: 0,
                "model-value": config.value[field.key],
                label: field.label,
                color: "primary",
                "hide-details": "",
                "onUpdate:modelValue": $event => (setField(field.key, $event))
              }, null, 8, ["model-value", "label", "onUpdate:modelValue"]))
            : (field.control === 'textarea')
              ? (_openBlock(), _createBlock(_component_VTextarea, {
                  key: 1,
                  "model-value": config.value[field.key],
                  label: field.label,
                  placeholder: field.placeholder,
                  "auto-grow": "",
                  rows: field.rows || 4,
                  "onUpdate:modelValue": $event => (setField(field.key, $event))
                }, null, 8, ["model-value", "label", "placeholder", "rows", "onUpdate:modelValue"]))
              : (field.control === 'password')
                ? (_openBlock(), _createBlock(_component_VTextField, {
                    key: 2,
                    "model-value": config.value[field.key],
                    label: field.label,
                    type: "password",
                    "onUpdate:modelValue": $event => (setField(field.key, $event))
                  }, null, 8, ["model-value", "label", "onUpdate:modelValue"]))
                : (field.control === 'number')
                  ? (_openBlock(), _createBlock(_component_VTextField, {
                      key: 3,
                      "model-value": config.value[field.key],
                      label: field.label,
                      type: "number",
                      "onUpdate:modelValue": $event => (setField(field.key, Number($event)))
                    }, null, 8, ["model-value", "label", "onUpdate:modelValue"]))
                  : (field.control === 'text')
                    ? (_openBlock(), _createBlock(_component_VTextField, {
                        key: 4,
                        "model-value": config.value[field.key],
                        label: field.label,
                        placeholder: field.placeholder,
                        "onUpdate:modelValue": $event => (setField(field.key, $event))
                      }, null, 8, ["model-value", "label", "placeholder", "onUpdate:modelValue"]))
                    : (field.control === 'select')
                      ? (_openBlock(), _createBlock(_component_VSelect, {
                          key: 5,
                          "model-value": config.value[field.key],
                          label: field.label,
                          items: field.options,
                          "onUpdate:modelValue": $event => (setField(field.key, $event))
                        }, null, 8, ["model-value", "label", "items", "onUpdate:modelValue"]))
                      : (field.control === 'multi')
                        ? (_openBlock(), _createElementBlock("div", _hoisted_3, [
                            _createElementVNode("div", _hoisted_4, _toDisplayString(field.label), 1),
                            (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(field.options, (option) => {
                              return (_openBlock(), _createBlock(_component_VChip, {
                                key: option.value,
                                class: "ma-1 ss-touch",
                                color: (config.value[field.key] || []).includes(option.value) ? 'primary' : undefined,
                                filter: "",
                                onClick: $event => (toggleList(field.key, option.value))
                              }, {
                                default: _withCtx(() => [
                                  _createTextVNode(_toDisplayString(option.title), 1)
                                ]),
                                _: 2
                              }, 1032, ["color", "onClick"]))
                            }), 128))
                          ]))
                        : (field.control === 'preset')
                          ? (_openBlock(), _createElementBlock("div", _hoisted_5, [
                              _createElementVNode("div", _hoisted_6, _toDisplayString(field.label), 1),
                              _createVNode(_component_VRow, { dense: "" }, {
                                default: _withCtx(() => [
                                  (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(_unref(PRESETS), (item) => {
                                    return (_openBlock(), _createBlock(_component_VCol, {
                                      key: item.value,
                                      cols: "12",
                                      md: "4"
                                    }, {
                                      default: _withCtx(() => [
                                        _createVNode(_component_VCard, {
                                          class: "ss-card",
                                          color: config.value.export_preset === item.value ? 'primary' : undefined,
                                          variant: "tonal",
                                          onClick: $event => (usePreset(item.value))
                                        }, {
                                          default: _withCtx(() => [
                                            _createVNode(_component_VCardTitle, { class: "text-subtitle-1" }, {
                                              default: _withCtx(() => [
                                                _createTextVNode(_toDisplayString(item.title), 1)
                                              ]),
                                              _: 2
                                            }, 1024),
                                            _createVNode(_component_VCardText, null, {
                                              default: _withCtx(() => [
                                                _createTextVNode(_toDisplayString(item.hint), 1)
                                              ]),
                                              _: 2
                                            }, 1024)
                                          ]),
                                          _: 2
                                        }, 1032, ["color", "onClick"])
                                      ]),
                                      _: 2
                                    }, 1024))
                                  }), 128))
                                ]),
                                _: 1
                              })
                            ]))
                          : (field.control === 'lang-order')
                            ? (_openBlock(), _createElementBlock("div", _hoisted_7, [
                                _createElementVNode("div", _hoisted_8, _toDisplayString(field.label), 1),
                                (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(config.value.target_languages || [], (lang, index) => {
                                  return (_openBlock(), _createElementBlock("div", {
                                    key: lang,
                                    class: "d-flex align-center ga-2 mb-2"
                                  }, [
                                    _createVNode(_component_VChip, { color: "primary" }, {
                                      default: _withCtx(() => [
                                        _createTextVNode(_toDisplayString(index + 1) + " · " + _toDisplayString(_unref(langLabel)(lang)) + " · " + _toDisplayString(_unref(sizeForRank)(index, assStyle.value)) + "px", 1)
                                      ]),
                                      _: 2
                                    }, 1024),
                                    _createVNode(_component_VBtn, {
                                      icon: "mdi-arrow-up",
                                      size: "small",
                                      class: "ss-touch",
                                      onClick: $event => (moveLang(index, -1))
                                    }, null, 8, ["onClick"]),
                                    _createVNode(_component_VBtn, {
                                      icon: "mdi-arrow-down",
                                      size: "small",
                                      class: "ss-touch",
                                      onClick: $event => (moveLang(index, 1))
                                    }, null, 8, ["onClick"]),
                                    _createVNode(_component_VBtn, {
                                      icon: "mdi-close",
                                      size: "small",
                                      class: "ss-touch",
                                      onClick: $event => (toggleList('target_languages', lang))
                                    }, null, 8, ["onClick"])
                                  ]))
                                }), 128)),
                                (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(_unref(LANGS).filter(lang => !(config.value.target_languages || []).includes(lang.value)), (item) => {
                                  return (_openBlock(), _createBlock(_component_VChip, {
                                    key: item.value,
                                    class: "ma-1 ss-touch",
                                    onClick: $event => (toggleList('target_languages', item.value))
                                  }, {
                                    default: _withCtx(() => [
                                      _createTextVNode(" 加 " + _toDisplayString(item.title), 1)
                                    ]),
                                    _: 2
                                  }, 1032, ["onClick"]))
                                }), 128))
                              ]))
                            : (field.control === 'endpoints')
                              ? (_openBlock(), _createElementBlock("div", _hoisted_9, [
                                  _createElementVNode("div", _hoisted_10, [
                                    _createElementVNode("div", _hoisted_11, _toDisplayString(field.label), 1),
                                    _createVNode(_component_VSpacer),
                                    _createVNode(_component_VBtn, {
                                      size: "small",
                                      class: "ss-touch",
                                      onClick: addEndpoint
                                    }, {
                                      default: _withCtx(() => [...(_cache[16] || (_cache[16] = [
                                        _createTextVNode("加线路", -1)
                                      ]))]),
                                      _: 1
                                    })
                                  ]),
                                  (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(config.value.openai_endpoints || [], (endpoint, index) => {
                                    return (_openBlock(), _createBlock(_component_VCard, {
                                      key: endpoint.endpoint_id,
                                      class: "ss-card mb-3 pa-3"
                                    }, {
                                      default: _withCtx(() => [
                                        _createVNode(_component_VTextField, {
                                          modelValue: endpoint.name,
                                          "onUpdate:modelValue": [$event => ((endpoint.name) = $event), $event => (updateEndpoint(index, { name: $event }))],
                                          label: "名称"
                                        }, null, 8, ["modelValue", "onUpdate:modelValue"]),
                                        _createVNode(_component_VTextField, {
                                          modelValue: endpoint.api_url,
                                          "onUpdate:modelValue": [$event => ((endpoint.api_url) = $event), $event => (updateEndpoint(index, { api_url: $event }))],
                                          label: "API URL"
                                        }, null, 8, ["modelValue", "onUpdate:modelValue"]),
                                        _createVNode(_component_VTextField, {
                                          modelValue: endpoint.api_key,
                                          "onUpdate:modelValue": [$event => ((endpoint.api_key) = $event), $event => (updateEndpoint(index, { api_key: $event }))],
                                          label: "API Key",
                                          type: "password"
                                        }, null, 8, ["modelValue", "onUpdate:modelValue"]),
                                        _createVNode(_component_VTextField, {
                                          modelValue: endpoint.model,
                                          "onUpdate:modelValue": [$event => ((endpoint.model) = $event), $event => (updateEndpoint(index, { model: $event }))],
                                          label: "模型"
                                        }, null, 8, ["modelValue", "onUpdate:modelValue"]),
                                        _createElementVNode("div", _hoisted_12, [
                                          _createVNode(_component_VSwitch, {
                                            "model-value": endpoint.enabled,
                                            label: "启用",
                                            "hide-details": "",
                                            "onUpdate:modelValue": $event => (updateEndpoint(index, { enabled: $event }))
                                          }, null, 8, ["model-value", "onUpdate:modelValue"]),
                                          _createVNode(_component_VSwitch, {
                                            "model-value": endpoint.primary,
                                            label: "主线路",
                                            "hide-details": "",
                                            "onUpdate:modelValue": $event => (updateEndpoint(index, { primary: $event }))
                                          }, null, 8, ["model-value", "onUpdate:modelValue"]),
                                          _createVNode(_component_VSwitch, {
                                            "model-value": endpoint.use_proxy,
                                            label: "该线路使用代理",
                                            "hide-details": "",
                                            "onUpdate:modelValue": $event => (updateEndpoint(index, { use_proxy: $event }))
                                          }, null, 8, ["model-value", "onUpdate:modelValue"]),
                                          _createVNode(_component_VSwitch, {
                                            "model-value": endpoint.compatible,
                                            label: "兼容模式",
                                            "hide-details": "",
                                            "onUpdate:modelValue": $event => (updateEndpoint(index, { compatible: $event }))
                                          }, null, 8, ["model-value", "onUpdate:modelValue"])
                                        ]),
                                        _createElementVNode("div", _hoisted_13, [
                                          _createVNode(_component_VBtn, {
                                            size: "small",
                                            class: "ss-touch",
                                            onClick: $event => (_ctx.$emit('test-endpoint', endpoint))
                                          }, {
                                            default: _withCtx(() => [...(_cache[17] || (_cache[17] = [
                                              _createTextVNode("测连通", -1)
                                            ]))]),
                                            _: 1
                                          }, 8, ["onClick"]),
                                          _createVNode(_component_VBtn, {
                                            size: "small",
                                            class: "ss-touch",
                                            onClick: $event => (_ctx.$emit('list-models', endpoint))
                                          }, {
                                            default: _withCtx(() => [...(_cache[18] || (_cache[18] = [
                                              _createTextVNode("拉模型", -1)
                                            ]))]),
                                            _: 1
                                          }, 8, ["onClick"])
                                        ])
                                      ]),
                                      _: 2
                                    }, 1024))
                                  }), 128))
                                ]))
                              : (field.control === 'style-editor')
                                ? (_openBlock(), _createElementBlock("div", _hoisted_14, [
                                    _createElementVNode("div", _hoisted_15, _toDisplayString(field.label), 1),
                                    _createElementVNode("div", _hoisted_16, [
                                      (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(_unref(STYLE_LOOKS), (look) => {
                                        return (_openBlock(), _createBlock(_component_VChip, {
                                          key: look.title,
                                          class: "ss-touch",
                                          onClick: $event => (useStyleLook(look))
                                        }, {
                                          default: _withCtx(() => [
                                            _createTextVNode(_toDisplayString(look.title), 1)
                                          ]),
                                          _: 2
                                        }, 1032, ["onClick"]))
                                      }), 128))
                                    ]),
                                    _createElementVNode("div", {
                                      class: _normalizeClass(["ss-style-preview mb-4", { 'ss-style-preview--top': stylePreview.value.stack === 'main_top' }])
                                    }, [
                                      _createElementVNode("div", {
                                        class: "ss-style-preview__note",
                                        style: _normalizeStyle({ ...stylePreview.value.note, top: stylePreview.value.noteTop })
                                      }, "钢铁侠 / 托尼·斯塔克", 4),
                                      (stylePreview.value.langs[2])
                                        ? (_openBlock(), _createElementBlock("div", {
                                            key: 0,
                                            class: "ss-style-preview__line",
                                            style: _normalizeStyle({ ...stylePreview.value.third, bottom: stylePreview.value.stack === 'main_top' ? 'auto' : stylePreview.value.thirdBottom, top: stylePreview.value.stack === 'main_top' ? stylePreview.value.mainBottom : 'auto' })
                                          }, " 三语小字 ", 4))
                                        : _createCommentVNode("", true),
                                      (stylePreview.value.langs[1])
                                        ? (_openBlock(), _createElementBlock("div", {
                                            key: 1,
                                            class: "ss-style-preview__line",
                                            style: _normalizeStyle({ ...stylePreview.value.second, bottom: stylePreview.value.stack === 'main_top' ? 'auto' : stylePreview.value.secondBottom, top: stylePreview.value.stack === 'main_top' ? stylePreview.value.secondBottom : 'auto' })
                                          }, " Secondary line ", 4))
                                        : _createCommentVNode("", true),
                                      _createElementVNode("div", {
                                        class: "ss-style-preview__line",
                                        style: _normalizeStyle({ ...stylePreview.value.main, bottom: stylePreview.value.stack === 'main_top' ? 'auto' : stylePreview.value.mainBottom, top: stylePreview.value.stack === 'main_top' ? stylePreview.value.thirdBottom : 'auto' })
                                      }, " 这是主字幕 ", 4)
                                    ], 2),
                                    _createVNode(_component_VRow, { dense: "" }, {
                                      default: _withCtx(() => [
                                        _createVNode(_component_VCol, {
                                          cols: "12",
                                          md: "6"
                                        }, {
                                          default: _withCtx(() => [
                                            _createVNode(_component_VSelect, {
                                              "model-value": assStyle.value.font_name,
                                              items: _unref(STYLE_FONTS),
                                              label: "字体",
                                              "onUpdate:modelValue": _cache[2] || (_cache[2] = $event => (setStyle({ font_name: $event })))
                                            }, null, 8, ["model-value", "items"])
                                          ]),
                                          _: 1
                                        }),
                                        _createVNode(_component_VCol, {
                                          cols: "12",
                                          md: "6"
                                        }, {
                                          default: _withCtx(() => [
                                            _createVNode(_component_VTextField, {
                                              "model-value": assStyle.value.font_name,
                                              label: "自定义字体名",
                                              "onUpdate:modelValue": _cache[3] || (_cache[3] = $event => (setStyle({ font_name: $event })))
                                            }, null, 8, ["model-value"])
                                          ]),
                                          _: 1
                                        }),
                                        _createVNode(_component_VCol, {
                                          cols: "6",
                                          md: "4"
                                        }, {
                                          default: _withCtx(() => [
                                            _createElementVNode("label", _hoisted_17, [
                                              _cache[19] || (_cache[19] = _createElementVNode("span", null, "对白颜色", -1)),
                                              _createElementVNode("input", {
                                                type: "color",
                                                class: "ss-color-input",
                                                value: assStyle.value.primary_color,
                                                onInput: _cache[4] || (_cache[4] = $event => (setStyle({ primary_color: $event.target.value })))
                                              }, null, 40, _hoisted_18)
                                            ])
                                          ]),
                                          _: 1
                                        }),
                                        _createVNode(_component_VCol, {
                                          cols: "6",
                                          md: "4"
                                        }, {
                                          default: _withCtx(() => [
                                            _createElementVNode("label", _hoisted_19, [
                                              _cache[20] || (_cache[20] = _createElementVNode("span", null, "描边颜色", -1)),
                                              _createElementVNode("input", {
                                                type: "color",
                                                class: "ss-color-input",
                                                value: assStyle.value.outline_color,
                                                onInput: _cache[5] || (_cache[5] = $event => (setStyle({ outline_color: $event.target.value })))
                                              }, null, 40, _hoisted_20)
                                            ])
                                          ]),
                                          _: 1
                                        }),
                                        _createVNode(_component_VCol, {
                                          cols: "6",
                                          md: "4"
                                        }, {
                                          default: _withCtx(() => [
                                            _createElementVNode("label", _hoisted_21, [
                                              _cache[21] || (_cache[21] = _createElementVNode("span", null, "顶注颜色", -1)),
                                              _createElementVNode("input", {
                                                type: "color",
                                                class: "ss-color-input",
                                                value: assStyle.value.note_color,
                                                onInput: _cache[6] || (_cache[6] = $event => (setStyle({ note_color: $event.target.value })))
                                              }, null, 40, _hoisted_22)
                                            ])
                                          ]),
                                          _: 1
                                        }),
                                        _createVNode(_component_VCol, { cols: "4" }, {
                                          default: _withCtx(() => [
                                            _createVNode(_component_VTextField, {
                                              "model-value": assStyle.value.sizes[0],
                                              type: "number",
                                              label: "主字号",
                                              "onUpdate:modelValue": _cache[7] || (_cache[7] = $event => (setStyleSize(0, $event)))
                                            }, null, 8, ["model-value"])
                                          ]),
                                          _: 1
                                        }),
                                        _createVNode(_component_VCol, { cols: "4" }, {
                                          default: _withCtx(() => [
                                            _createVNode(_component_VTextField, {
                                              "model-value": assStyle.value.sizes[1],
                                              type: "number",
                                              label: "次行小字",
                                              "onUpdate:modelValue": _cache[8] || (_cache[8] = $event => (setStyleSize(1, $event)))
                                            }, null, 8, ["model-value"])
                                          ]),
                                          _: 1
                                        }),
                                        _createVNode(_component_VCol, { cols: "4" }, {
                                          default: _withCtx(() => [
                                            _createVNode(_component_VTextField, {
                                              "model-value": assStyle.value.sizes[2],
                                              type: "number",
                                              label: "第三行",
                                              "onUpdate:modelValue": _cache[9] || (_cache[9] = $event => (setStyleSize(2, $event)))
                                            }, null, 8, ["model-value"])
                                          ]),
                                          _: 1
                                        }),
                                        _createVNode(_component_VCol, {
                                          cols: "6",
                                          md: "3"
                                        }, {
                                          default: _withCtx(() => [
                                            _createVNode(_component_VTextField, {
                                              "model-value": assStyle.value.note_size,
                                              type: "number",
                                              label: "顶注字号",
                                              "onUpdate:modelValue": _cache[10] || (_cache[10] = $event => (setStyle({ note_size: Number($event) })))
                                            }, null, 8, ["model-value"])
                                          ]),
                                          _: 1
                                        }),
                                        _createVNode(_component_VCol, {
                                          cols: "6",
                                          md: "3"
                                        }, {
                                          default: _withCtx(() => [
                                            _createVNode(_component_VTextField, {
                                              "model-value": assStyle.value.outline,
                                              type: "number",
                                              label: "描边",
                                              "onUpdate:modelValue": _cache[11] || (_cache[11] = $event => (setStyle({ outline: Number($event) })))
                                            }, null, 8, ["model-value"])
                                          ]),
                                          _: 1
                                        }),
                                        _createVNode(_component_VCol, {
                                          cols: "6",
                                          md: "3"
                                        }, {
                                          default: _withCtx(() => [
                                            _createVNode(_component_VTextField, {
                                              "model-value": assStyle.value.shadow,
                                              type: "number",
                                              label: "阴影",
                                              "onUpdate:modelValue": _cache[12] || (_cache[12] = $event => (setStyle({ shadow: Number($event) })))
                                            }, null, 8, ["model-value"])
                                          ]),
                                          _: 1
                                        }),
                                        _createVNode(_component_VCol, {
                                          cols: "6",
                                          md: "3"
                                        }, {
                                          default: _withCtx(() => [
                                            _createVNode(_component_VTextField, {
                                              "model-value": assStyle.value.margin_v,
                                              type: "number",
                                              label: "底边距",
                                              "onUpdate:modelValue": _cache[13] || (_cache[13] = $event => (setStyle({ margin_v: Number($event) })))
                                            }, null, 8, ["model-value"])
                                          ]),
                                          _: 1
                                        })
                                      ]),
                                      _: 1
                                    }),
                                    _createElementVNode("div", _hoisted_23, [
                                      _createVNode(_component_VSwitch, {
                                        "model-value": assStyle.value.bold,
                                        label: "粗体",
                                        "hide-details": "",
                                        "onUpdate:modelValue": _cache[14] || (_cache[14] = $event => (setStyle({ bold: $event })))
                                      }, null, 8, ["model-value"]),
                                      _createVNode(_component_VSwitch, {
                                        "model-value": assStyle.value.italic,
                                        label: "斜体",
                                        "hide-details": "",
                                        "onUpdate:modelValue": _cache[15] || (_cache[15] = $event => (setStyle({ italic: $event })))
                                      }, null, 8, ["model-value"])
                                    ]),
                                    (!(config.value.export_formats || []).includes('ass'))
                                      ? (_openBlock(), _createBlock(_component_VAlert, {
                                          key: 0,
                                          type: "warning",
                                          variant: "tonal",
                                          class: "mt-3"
                                        }, {
                                          default: _withCtx(() => [...(_cache[22] || (_cache[22] = [
                                            _createTextVNode(" 字体和颜色只写入 ASS。当前没勾 ASS，播放器会用自己的字体。 ", -1)
                                          ]))]),
                                          _: 1
                                        }))
                                      : _createCommentVNode("", true),
                                    (!(config.value.export_formats || []).includes('ass'))
                                      ? (_openBlock(), _createBlock(_component_VBtn, {
                                          key: 1,
                                          size: "small",
                                          class: "ss-touch mt-2",
                                          onClick: ensureAssFormat
                                        }, {
                                          default: _withCtx(() => [...(_cache[23] || (_cache[23] = [
                                            _createTextVNode("同时写出 ASS", -1)
                                          ]))]),
                                          _: 1
                                        }))
                                      : _createCommentVNode("", true)
                                  ]))
                                : _createCommentVNode("", true),
          _createElementVNode("div", _hoisted_24, _toDisplayString(field.purpose) + " " + _toDisplayString(field.after), 1)
        ]))
      }), 128))
    ]),
    (pane.value === 'export')
      ? (_openBlock(), _createBlock(_component_VAlert, {
          key: 2,
          type: "info",
          variant: "tonal",
          class: "mt-2"
        }, {
          default: _withCtx(() => [
            _createTextVNode(" 将写出：" + _toDisplayString(files.value.join(' · ') || '请至少勾一种语种、版式和格式'), 1)
          ]),
          _: 1
        }))
      : _createCommentVNode("", true)
  ]))
}
}

};

export { _sfc_main as _, useMobileViewport as a, langLabel as l, overlayCss as o, useHostInjects as u };
