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
  { value: 'library_zh', title: '媒体库中文', hint: 'zh-Hans + default · 默认勾 SRT' },
  { value: 'plex', title: 'Plex', hint: 'chi · 不标 default' },
  { value: 'web', title: '网页 / Infuse', hint: '再勾 VTT' },
  { value: 'legacy', title: '兼容旧库', hint: 'chi / chi&eng' },
];

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

function sizeForRank(index) {
  return [22, 17, 14][index] || 14
}

function applyPreset(config, preset) {
  const next = { ...config, export_preset: preset };
  if (preset === 'plex') next.mark_default = false;
  if (preset === 'library_zh' || preset === 'web') next.mark_default = true;
  const formats = new Set(next.export_formats || ['srt']);
  formats.add('srt');
  if (preset === 'web') formats.add('vtt');
  next.export_formats = Array.from(formats);
  return next
}

function exportPreview(config, stem = 'Movie') {
  const langs = (config.target_languages || ['zh-Hans']).slice(0, 3);
  const layouts = config.export_layouts || ['mono'];
  const formats = config.export_formats || ['srt'];
  const files = [];
  for (const layout of layouts) {
    for (const fmt of formats) {
      if (layout === 'mono') files.push(`${stem}.${langs[0]}.${fmt}`);
      if (layout === 'stacked' && langs.length > 1) files.push(`${stem}.${langs.join('.')}.${fmt}`);
      if (layout === 'split') langs.forEach(lang => files.push(`${stem}.${lang}.${fmt}`));
    }
  }
  if (config.effects_enabled) files.push(`${stem}.${langs[0]}.notes.ass`);
  if (config.enable_sdh) files.push(`${stem}.${langs[0]}.sdh.srt`);
  if (config.save_asr_track) files.push(`${stem}.en.asr.srt`);
  return files
}

const {unref:_unref,renderList:_renderList,Fragment:_Fragment,openBlock:_openBlock,createElementBlock:_createElementBlock,toDisplayString:_toDisplayString,createTextVNode:_createTextVNode,resolveComponent:_resolveComponent,withCtx:_withCtx,createBlock:_createBlock,createCommentVNode:_createCommentVNode,createElementVNode:_createElementVNode,createVNode:_createVNode} = await importShared('vue');


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
const _hoisted_14 = { class: "ss-help" };

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
  return rows.filter(item => item.pane === pane.value && item.mock !== 'skip')
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
    { pane: 'basic', key: 'send_notify', label: '任务完成通知', control: 'switch', purpose: '走 MoviePilot 通知。', after: '默认关，避免剧集刷屏。' },
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
                                      md: "3"
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
                                        _createTextVNode(_toDisplayString(index + 1) + " · " + _toDisplayString(_unref(langLabel)(lang)) + " · " + _toDisplayString(_unref(sizeForRank)(index)) + "px", 1)
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
                                      default: _withCtx(() => [...(_cache[2] || (_cache[2] = [
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
                                            default: _withCtx(() => [...(_cache[3] || (_cache[3] = [
                                              _createTextVNode("测连通", -1)
                                            ]))]),
                                            _: 1
                                          }, 8, ["onClick"]),
                                          _createVNode(_component_VBtn, {
                                            size: "small",
                                            class: "ss-touch",
                                            onClick: $event => (_ctx.$emit('list-models', endpoint))
                                          }, {
                                            default: _withCtx(() => [...(_cache[4] || (_cache[4] = [
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
                              : _createCommentVNode("", true),
          _createElementVNode("div", _hoisted_14, _toDisplayString(field.purpose) + " " + _toDisplayString(field.after), 1)
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

export { _sfc_main as _, useMobileViewport as a, langLabel as l, useHostInjects as u };
