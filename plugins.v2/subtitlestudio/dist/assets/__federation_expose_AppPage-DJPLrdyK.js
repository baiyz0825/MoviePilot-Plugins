import { importShared } from './__federation_fn_import-JrT3xvdd.js';
import { a as useMobileViewport, u as useHostInjects, _ as _sfc_main$2, l as langLabel } from './SettingsForm-CGWuXqQW.js';
import { c as createStudioApi } from './studioApi-CSoBhxaG.js';

const {openBlock:_openBlock$1,createElementBlock:_createElementBlock$1,createCommentVNode:_createCommentVNode$1,createElementVNode:_createElementVNode$1,renderList:_renderList$1,Fragment:_Fragment$1,toDisplayString:_toDisplayString$1,normalizeClass:_normalizeClass,normalizeStyle:_normalizeStyle$1} = await importShared('vue');


const _hoisted_1$1 = {
  key: 0,
  class: "ss-player"
};
const _hoisted_2$1 = ["src"];
const _hoisted_3$1 = {
  key: 1,
  class: "ss-board pa-6 text-center"
};

const {computed: computed$1} = await importShared('vue');



const _sfc_main$1 = {
  __name: 'PreviewPlayer',
  props: {
  graph: { type: Object, default: () => ({ cues: [], notes: [] }) },
  currentMs: { type: Number, default: 0 },
  langs: { type: Array, default: () => ['zh-Hans'] },
  tracks: { type: Array, default: () => ['dialogue', 'notes'] },
  stack: { type: String, default: 'main_bottom' },
  videoUrl: { type: String, default: '' },
  enabled: { type: Boolean, default: true },
},
  emits: ['time'],
  setup(__props, { emit: __emit }) {

const props = __props;

const emit = __emit;

const active = computed$1(() => {
  const cues = [...(props.graph?.cues || []), ...(props.graph?.notes || [])];
  return cues.filter(item => item.start_ms <= props.currentMs && props.currentMs < item.end_ms)
});

const dialogue = computed$1(() => active.value.filter(item => item.kind !== 'note'));
const notes = computed$1(() => active.value.filter(item => item.kind === 'note'));

function textOf(cue, lang) {
  return cue?.texts?.[lang] || cue?.texts?.source || Object.values(cue?.texts || {})[0] || ''
}

function onTime(event) {
  emit('time', Math.floor((event.target.currentTime || 0) * 1000));
}

return (_ctx, _cache) => {
  return (__props.enabled)
    ? (_openBlock$1(), _createElementBlock$1("div", _hoisted_1$1, [
        (__props.videoUrl)
          ? (_openBlock$1(), _createElementBlock$1("video", {
              key: 0,
              src: __props.videoUrl,
              controls: "",
              onTimeupdate: onTime
            }, null, 40, _hoisted_2$1))
          : (_openBlock$1(), _createElementBlock$1("div", _hoisted_3$1, [...(_cache[0] || (_cache[0] = [
              _createElementVNode$1("div", { class: "text-medium-emphasis" }, "字幕黑板 · 当前没有可播原片", -1)
            ]))])),
        (__props.tracks.includes('notes'))
          ? (_openBlock$1(true), _createElementBlock$1(_Fragment$1, { key: 2 }, _renderList$1(notes.value, (note) => {
              return (_openBlock$1(), _createElementBlock$1("div", {
                key: note.cue_id,
                class: "ss-overlay note"
              }, _toDisplayString$1(textOf(note)), 1))
            }), 128))
          : _createCommentVNode$1("", true),
        (__props.tracks.includes('dialogue'))
          ? (_openBlock$1(true), _createElementBlock$1(_Fragment$1, { key: 3 }, _renderList$1(__props.langs, (lang, index) => {
              return (_openBlock$1(), _createElementBlock$1("div", {
                key: lang,
                class: _normalizeClass(["ss-overlay dialogue", [`rank${index + 1}`]]),
                style: _normalizeStyle$1(__props.stack === 'main_top' && index === 0 ? { top: '72px', bottom: 'auto' } : {})
              }, _toDisplayString$1(dialogue.value.map(item => textOf(item, lang)).filter(Boolean).join(' ')), 7))
            }), 128))
          : _createCommentVNode$1("", true)
      ]))
    : _createCommentVNode$1("", true)
}
}

};

const {openBlock:_openBlock,createElementBlock:_createElementBlock,createCommentVNode:_createCommentVNode,renderList:_renderList,Fragment:_Fragment,resolveComponent:_resolveComponent,createVNode:_createVNode,toDisplayString:_toDisplayString,createTextVNode:_createTextVNode,withCtx:_withCtx,createElementVNode:_createElementVNode,createBlock:_createBlock,withKeys:_withKeys,createSlots:_createSlots,unref:_unref,withModifiers:_withModifiers,normalizeStyle:_normalizeStyle} = await importShared('vue');


const _hoisted_1 = { class: "plugin-root" };
const _hoisted_2 = { class: "ss-toolbar pa-3" };
const _hoisted_3 = {
  key: 0,
  class: "text-h6 mb-2"
};
const _hoisted_4 = { class: "pa-3" };
const _hoisted_5 = {
  key: 0,
  class: "mb-3"
};
const _hoisted_6 = { class: "d-flex align-center mb-3" };
const _hoisted_7 = { class: "ms-2" };
const _hoisted_8 = { class: "text-medium-emphasis mb-2" };
const _hoisted_9 = { class: "d-flex ga-2 mb-3" };
const _hoisted_10 = {
  class: "d-flex ga-2 mb-3",
  style: {"overflow-x":"auto"}
};
const _hoisted_11 = { class: "text-medium-emphasis mb-3" };
const _hoisted_12 = ["src"];
const _hoisted_13 = {
  class: "d-flex ga-2 mb-3",
  style: {"overflow-x":"auto"}
};
const _hoisted_14 = {
  key: 0,
  class: "text-medium-emphasis"
};
const _hoisted_15 = { class: "mb-2" };
const _hoisted_16 = { class: "d-flex ga-2 my-3" };
const _hoisted_17 = ["onClick"];
const _hoisted_18 = {
  key: 0,
  class: "ss-savebar pa-3 d-flex ga-2"
};
const _hoisted_19 = { class: "text-medium-emphasis mb-2" };

const {computed,onMounted,reactive,ref} = await importShared('vue');


const _sfc_main = {
  __name: 'AppPage',
  props: {
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  navKey: { type: String, default: 'main' },
  hideTitle: { type: Boolean, default: false },
},
  emits: ['action'],
  setup(__props, { expose: __expose, emit: __emit }) {

const props = __props;
const isMobile = useMobileViewport();
const { toast, confirm } = useHostInjects();
const pluginBase = computed(() => `plugin/${props.pluginId || 'SubtitleStudio'}`);
const pluginApi = computed(() => createStudioApi(props.api, pluginBase));

const nav = ref('media');
const loading = ref(false);
const dirty = ref(false);
const mediaQuery = ref('');
const mediaType = ref('');
const mediaGroups = ref([]);
const mediaCounts = ref({ files: 0, groups: 0 });
const mediaDetail = ref(null);
const selectedPaths = ref({});
const submitting = ref(false);
const jobsQuery = ref('');
const jobStatus = ref('');
const jobs = ref([]);
const jobSheet = ref(false);
const activeJob = ref(null);
const graph = ref({ cues: [], notes: [], briefing: [] });
const currentMs = ref(0);
const previewTracks = ref(['dialogue', 'notes']);
const cueSheet = ref(false);
const editingCue = ref(null);
const config = ref({});
const fields = ref([]);
const enqueueSheet = ref(false);
const enqueueForm = reactive({ strategy: 'search_then_translate', priority: 'P0', force: true, items: [] });
const selectedFiles = computed(() => (mediaDetail.value?.files || []).filter(item => selectedPaths.value[item.path]));

const tabs = [
  { value: 'media', title: '媒体', icon: 'mdi-filmstrip' },
  { value: 'jobs', title: '队列', icon: 'mdi-playlist-play' },
  { value: 'desk', title: '工作台', icon: 'mdi-subtitles-outline' },
  { value: 'settings', title: '设置', icon: 'mdi-cog-outline' },
];

const duration = computed(() => Math.max(graph.value.duration_ms || 1, ...(graph.value.cues || []).map(item => item.end_ms || 0), 1));

async function reload() {
  loading.value = true;
  try {
    const [cfg, fieldData] = await Promise.all([
      pluginApi.value.config().catch(() => ({})),
      pluginApi.value.fields().catch(() => ({ fields: [] })),
    ]);
    config.value = { ...cfg };
    fields.value = fieldData.fields || [];
    previewTracks.value = cfg.preview_tracks_default || ['dialogue', 'notes'];
    await Promise.all([loadMedia(), loadJobs()]);
  } finally {
    loading.value = false;
  }
}

async function loadMedia() {
  const data = await pluginApi.value.media(mediaQuery.value, mediaType.value);
  mediaGroups.value = data?.groups || [];
  mediaCounts.value = data?.counts || { files: 0, groups: 0 };
}

async function refreshLibrary() {
  loading.value = true;
  try {
    const data = await pluginApi.value.refreshMedia();
    mediaGroups.value = data?.groups || [];
    mediaCounts.value = data?.counts || { files: 0, groups: 0 };
    toast.success?.(`已拉取 ${mediaCounts.value.files || 0} 个媒体文件`);
  } catch (error) {
    toast.error?.(error?.message || '拉取媒体库失败');
  } finally {
    loading.value = false;
  }
}

function openGroup(group) {
  mediaDetail.value = group;
  selectedPaths.value = Object.fromEntries((group.files || []).map(item => [item.path, true]));
}

function toggleFile(path, value) {
  selectedPaths.value = { ...selectedPaths.value, [path]: value };
}

function toggleAll(value) {
  selectedPaths.value = Object.fromEntries((mediaDetail.value?.files || []).map(item => [item.path, value]));
}

function fileLabel(item) {
  if (item.type === 'tv' && item.season && item.episode) {
    const season = String(item.season).padStart(2, '0');
    const episode = String(item.episode).padStart(2, '0');
    return `S${season}E${episode} · ${item.filename || item.path}`
  }
  return item.filename || item.path
}

async function loadJobs() {
  const data = await pluginApi.value.jobs(jobsQuery.value, jobStatus.value);
  jobs.value = data?.items || [];
}

async function saveConfig() {
  await pluginApi.value.saveConfig(config.value);
  dirty.value = false;
  toast.success?.('已保存');
}

async function enqueue(item) {
  enqueueForm.items = item?.path ? [item] : selectedFiles.value;
  enqueueForm.path = item?.path || enqueueForm.items[0]?.path;
  enqueueForm.title = item?.title || mediaDetail.value?.title;
  enqueueForm.media_source = item?.media_source;
  enqueueForm.media_id = item?.media_id;
  enqueueForm.tmdbid = item?.tmdbid;
  enqueueForm.doubanid = item?.doubanid;
  enqueueForm.force = true;
  enqueueSheet.value = true;
}

async function submitSelected() {
  if (!selectedFiles.value.length) {
    toast.error?.('先勾选要识别的文件');
    return
  }
  enqueueForm.items = selectedFiles.value;
  enqueueForm.force = true;
  enqueueSheet.value = true;
}

async function confirmEnqueue() {
  submitting.value = true;
  try {
    const items = enqueueForm.items?.length ? enqueueForm.items : [enqueueForm];
    await pluginApi.value.createJobs({
      items,
      strategy: enqueueForm.strategy,
      priority: enqueueForm.priority,
      force: enqueueForm.force,
    });
    enqueueSheet.value = false;
    nav.value = 'jobs';
    await loadJobs();
  } catch (error) {
    toast.error?.(error?.message || '入队失败');
  } finally {
    submitting.value = false;
  }
}

async function openJob(job) {
  activeJob.value = job;
  graph.value = (await pluginApi.value.cues(job.job_id)) || { cues: [], notes: [] };
  nav.value = 'desk';
}

async function cutIn(job) {
  await pluginApi.value.cutIn(job.job_id);
  await loadJobs();
}

async function changePriority(job, priority) {
  await pluginApi.value.setPriority(job.job_id, priority);
  await loadJobs();
}

async function cancelJob(job) {
  if (!(await confirm('取消这个任务？'))) return
  await pluginApi.value.cancel(job.job_id);
  await loadJobs();
}

function openCue(cue) {
  editingCue.value = { ...cue, texts: { ...(cue.texts || {}) } };
  cueSheet.value = true;
}

async function saveCue() {
  if (!(await confirm('写回这一句到 CueGraph？'))) return
  await pluginApi.value.saveCue(activeJob.value.job_id, editingCue.value.cue_id, editingCue.value);
  cueSheet.value = false;
  graph.value = await pluginApi.value.cues(activeJob.value.job_id);
}

function cueLeft(cue) {
  return `${(cue.start_ms / duration.value) * 100}%`
}

function cueWidth(cue) {
  return `${Math.max(4, ((cue.end_ms - cue.start_ms) / duration.value) * 100)}%`
}

function seekTimeline(event) {
  const box = event.currentTarget.getBoundingClientRect();
  currentMs.value = Math.floor(((event.clientX - box.left) / box.width) * duration.value);
}

function moreJob(job) {
  activeJob.value = job;
  jobSheet.value = true;
}

onMounted(reload);
__expose({ reload, loadStatus: reload });

return (_ctx, _cache) => {
  const _component_VIcon = _resolveComponent("VIcon");
  const _component_VBtn = _resolveComponent("VBtn");
  const _component_VBtnToggle = _resolveComponent("VBtnToggle");
  const _component_VCheckbox = _resolveComponent("VCheckbox");
  const _component_VListItemTitle = _resolveComponent("VListItemTitle");
  const _component_VListItemSubtitle = _resolveComponent("VListItemSubtitle");
  const _component_VListItem = _resolveComponent("VListItem");
  const _component_VList = _resolveComponent("VList");
  const _component_VTextField = _resolveComponent("VTextField");
  const _component_VChip = _resolveComponent("VChip");
  const _component_VAlert = _resolveComponent("VAlert");
  const _component_VCardTitle = _resolveComponent("VCardTitle");
  const _component_VCard = _resolveComponent("VCard");
  const _component_VBottomSheet = _resolveComponent("VBottomSheet");
  const _component_VSelect = _resolveComponent("VSelect");
  const _component_VSwitch = _resolveComponent("VSwitch");

  return (_openBlock(), _createElementBlock("div", _hoisted_1, [
    _createElementVNode("div", _hoisted_2, [
      (!__props.hideTitle)
        ? (_openBlock(), _createElementBlock("div", _hoisted_3, "字幕工坊"))
        : _createCommentVNode("", true),
      _createVNode(_component_VBtnToggle, {
        modelValue: nav.value,
        "onUpdate:modelValue": _cache[0] || (_cache[0] = $event => ((nav).value = $event)),
        mandatory: "",
        density: "comfortable",
        class: "w-100"
      }, {
        default: _withCtx(() => [
          (_openBlock(), _createElementBlock(_Fragment, null, _renderList(tabs, (tab) => {
            return _createVNode(_component_VBtn, {
              key: tab.value,
              value: tab.value,
              class: "ss-touch flex-grow-1"
            }, {
              default: _withCtx(() => [
                _createVNode(_component_VIcon, {
                  start: "",
                  icon: tab.icon
                }, null, 8, ["icon"]),
                _createTextVNode(" " + _toDisplayString(tab.title), 1)
              ]),
              _: 2
            }, 1032, ["value"])
          }), 64))
        ]),
        _: 1
      }, 8, ["modelValue"])
    ]),
    _createElementVNode("div", _hoisted_4, [
      (nav.value === 'media')
        ? (_openBlock(), _createElementBlock(_Fragment, { key: 0 }, [
            (mediaDetail.value)
              ? (_openBlock(), _createElementBlock("div", _hoisted_5, [
                  _createElementVNode("div", _hoisted_6, [
                    _createVNode(_component_VBtn, {
                      icon: "mdi-arrow-left",
                      class: "ss-touch",
                      onClick: _cache[1] || (_cache[1] = $event => (mediaDetail.value = null))
                    }),
                    _createElementVNode("strong", _hoisted_7, _toDisplayString(mediaDetail.value.title) + _toDisplayString(mediaDetail.value.year ? ` (${mediaDetail.value.year})` : ''), 1)
                  ]),
                  _createElementVNode("div", _hoisted_8, _toDisplayString(mediaDetail.value.library_name || 'MoviePilot 整理记录') + " · " + _toDisplayString(mediaDetail.value.file_count || mediaDetail.value.files?.length || 0) + " 个文件", 1),
                  _createElementVNode("div", _hoisted_9, [
                    _createVNode(_component_VBtn, {
                      size: "small",
                      variant: "text",
                      class: "ss-touch",
                      onClick: _cache[2] || (_cache[2] = $event => (toggleAll(true)))
                    }, {
                      default: _withCtx(() => [...(_cache[28] || (_cache[28] = [
                        _createTextVNode("全选", -1)
                      ]))]),
                      _: 1
                    }),
                    _createVNode(_component_VBtn, {
                      size: "small",
                      variant: "text",
                      class: "ss-touch",
                      onClick: _cache[3] || (_cache[3] = $event => (toggleAll(false)))
                    }, {
                      default: _withCtx(() => [...(_cache[29] || (_cache[29] = [
                        _createTextVNode("清空", -1)
                      ]))]),
                      _: 1
                    })
                  ]),
                  _createVNode(_component_VList, null, {
                    default: _withCtx(() => [
                      (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(mediaDetail.value.files || [], (item) => {
                        return (_openBlock(), _createBlock(_component_VListItem, {
                          key: item.id || item.path
                        }, {
                          prepend: _withCtx(() => [
                            _createVNode(_component_VCheckbox, {
                              "model-value": !!selectedPaths.value[item.path],
                              "hide-details": "",
                              "onUpdate:modelValue": $event => (toggleFile(item.path, $event))
                            }, null, 8, ["model-value", "onUpdate:modelValue"])
                          ]),
                          default: _withCtx(() => [
                            _createVNode(_component_VListItemTitle, null, {
                              default: _withCtx(() => [
                                _createTextVNode(_toDisplayString(fileLabel(item)), 1)
                              ]),
                              _: 2
                            }, 1024),
                            _createVNode(_component_VListItemSubtitle, null, {
                              default: _withCtx(() => [
                                _createTextVNode(_toDisplayString(item.sidecars?.length || 0) + " 条外挂" + _toDisplayString(item.is_strm ? ' · STRM' : ''), 1)
                              ]),
                              _: 2
                            }, 1024)
                          ]),
                          _: 2
                        }, 1024))
                      }), 128))
                    ]),
                    _: 1
                  }),
                  _createVNode(_component_VBtn, {
                    color: "primary",
                    block: "",
                    class: "ss-touch mt-3",
                    disabled: !selectedFiles.value.length,
                    onClick: submitSelected
                  }, {
                    default: _withCtx(() => [
                      _createTextVNode(" 提交识别（" + _toDisplayString(selectedFiles.value.length) + "） ", 1)
                    ]),
                    _: 1
                  }, 8, ["disabled"])
                ]))
              : (_openBlock(), _createElementBlock(_Fragment, { key: 1 }, [
                  _createVNode(_component_VTextField, {
                    modelValue: mediaQuery.value,
                    "onUpdate:modelValue": _cache[4] || (_cache[4] = $event => ((mediaQuery).value = $event)),
                    label: "搜索标题或文件名",
                    "prepend-inner-icon": "mdi-magnify",
                    class: "mb-3",
                    onKeyup: _withKeys(loadMedia, ["enter"])
                  }, null, 8, ["modelValue"]),
                  _createElementVNode("div", _hoisted_10, [
                    _createVNode(_component_VChip, {
                      color: !mediaType.value ? 'primary' : undefined,
                      onClick: _cache[5] || (_cache[5] = $event => {mediaType.value = ''; loadMedia();})
                    }, {
                      default: _withCtx(() => [...(_cache[30] || (_cache[30] = [
                        _createTextVNode("全部", -1)
                      ]))]),
                      _: 1
                    }, 8, ["color"]),
                    _createVNode(_component_VChip, {
                      color: mediaType.value === 'movie' ? 'primary' : undefined,
                      onClick: _cache[6] || (_cache[6] = $event => {mediaType.value = 'movie'; loadMedia();})
                    }, {
                      default: _withCtx(() => [...(_cache[31] || (_cache[31] = [
                        _createTextVNode("电影", -1)
                      ]))]),
                      _: 1
                    }, 8, ["color"]),
                    _createVNode(_component_VChip, {
                      color: mediaType.value === 'tv' ? 'primary' : undefined,
                      onClick: _cache[7] || (_cache[7] = $event => {mediaType.value = 'tv'; loadMedia();})
                    }, {
                      default: _withCtx(() => [...(_cache[32] || (_cache[32] = [
                        _createTextVNode("剧集", -1)
                      ]))]),
                      _: 1
                    }, 8, ["color"]),
                    _createVNode(_component_VBtn, {
                      size: "small",
                      class: "ss-touch",
                      loading: loading.value,
                      onClick: refreshLibrary
                    }, {
                      default: _withCtx(() => [...(_cache[33] || (_cache[33] = [
                        _createTextVNode("拉取媒体库", -1)
                      ]))]),
                      _: 1
                    }, 8, ["loading"])
                  ]),
                  _createElementVNode("div", _hoisted_11, "整理记录 " + _toDisplayString(mediaCounts.value.groups || 0) + " 部 · " + _toDisplayString(mediaCounts.value.files || 0) + " 个文件", 1),
                  (!mediaGroups.value.length)
                    ? (_openBlock(), _createBlock(_component_VAlert, {
                        key: 0,
                        type: "info",
                        variant: "tonal",
                        class: "mb-3"
                      }, {
                        default: _withCtx(() => [...(_cache[34] || (_cache[34] = [
                          _createTextVNode(" 没有本地媒体。点「拉取媒体库」读取 MoviePilot 整理记录；先在 MoviePilot 里整理入库后才会出现。这里不是 Emby/Jellyfin 在线目录。 ", -1)
                        ]))]),
                        _: 1
                      }))
                    : _createCommentVNode("", true),
                  _createVNode(_component_VList, null, {
                    default: _withCtx(() => [
                      (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(mediaGroups.value, (item) => {
                        return (_openBlock(), _createBlock(_component_VListItem, {
                          key: item.id,
                          class: "ss-card mb-2",
                          onClick: $event => (openGroup(item))
                        }, _createSlots({
                          append: _withCtx(() => [
                            _createVNode(_component_VIcon, { icon: "mdi-chevron-right" })
                          ]),
                          default: _withCtx(() => [
                            _createVNode(_component_VListItemTitle, null, {
                              default: _withCtx(() => [
                                _createTextVNode(_toDisplayString(item.title) + _toDisplayString(item.year ? ` (${item.year})` : ''), 1)
                              ]),
                              _: 2
                            }, 1024),
                            _createVNode(_component_VListItemSubtitle, null, {
                              default: _withCtx(() => [
                                _createTextVNode(_toDisplayString(item.type === 'tv' ? '剧集' : '电影') + " · " + _toDisplayString(item.file_count || item.files?.length || 0) + " 个文件 · " + _toDisplayString(item.library_name), 1)
                              ]),
                              _: 2
                            }, 1024)
                          ]),
                          _: 2
                        }, [
                          (item.poster)
                            ? {
                                name: "prepend",
                                fn: _withCtx(() => [
                                  _createElementVNode("img", {
                                    src: item.poster,
                                    alt: "",
                                    width: "40",
                                    height: "56",
                                    style: {"object-fit":"cover","border-radius":"4px"}
                                  }, null, 8, _hoisted_12)
                                ]),
                                key: "0"
                              }
                            : undefined
                        ]), 1032, ["onClick"]))
                      }), 128))
                    ]),
                    _: 1
                  })
                ], 64))
          ], 64))
        : (nav.value === 'jobs')
          ? (_openBlock(), _createElementBlock(_Fragment, { key: 1 }, [
              _createVNode(_component_VTextField, {
                modelValue: jobsQuery.value,
                "onUpdate:modelValue": _cache[8] || (_cache[8] = $event => ((jobsQuery).value = $event)),
                label: "搜索标题",
                "prepend-inner-icon": "mdi-magnify",
                class: "mb-3",
                onKeyup: _withKeys(loadJobs, ["enter"])
              }, null, 8, ["modelValue"]),
              _createElementVNode("div", _hoisted_13, [
                (_openBlock(), _createElementBlock(_Fragment, null, _renderList(['', 'pending', 'running', 'failed', 'success'], (item) => {
                  return _createVNode(_component_VChip, {
                    key: item || 'all',
                    color: jobStatus.value === item ? 'primary' : undefined,
                    onClick: $event => {jobStatus.value = item; loadJobs();}
                  }, {
                    default: _withCtx(() => [
                      _createTextVNode(_toDisplayString(item || '全部'), 1)
                    ]),
                    _: 2
                  }, 1032, ["color", "onClick"])
                }), 64))
              ]),
              _createVNode(_component_VList, null, {
                default: _withCtx(() => [
                  (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(jobs.value, (job) => {
                    return (_openBlock(), _createBlock(_component_VListItem, {
                      key: job.job_id,
                      class: "ss-card mb-2"
                    }, {
                      append: _withCtx(() => [
                        (!_unref(isMobile) && job.status === 'pending')
                          ? (_openBlock(), _createBlock(_component_VBtn, {
                              key: 0,
                              size: "small",
                              class: "ss-touch",
                              onClick: $event => (cutIn(job))
                            }, {
                              default: _withCtx(() => [...(_cache[35] || (_cache[35] = [
                                _createTextVNode("插队", -1)
                              ]))]),
                              _: 1
                            }, 8, ["onClick"]))
                          : _createCommentVNode("", true),
                        (!_unref(isMobile))
                          ? (_openBlock(), _createBlock(_component_VBtn, {
                              key: 1,
                              size: "small",
                              variant: "text",
                              class: "ss-touch",
                              onClick: $event => (openJob(job))
                            }, {
                              default: _withCtx(() => [...(_cache[36] || (_cache[36] = [
                                _createTextVNode("工作台", -1)
                              ]))]),
                              _: 1
                            }, 8, ["onClick"]))
                          : _createCommentVNode("", true),
                        _createVNode(_component_VBtn, {
                          icon: "mdi-dots-horizontal",
                          class: "ss-touch",
                          onClick: $event => (moreJob(job))
                        }, null, 8, ["onClick"])
                      ]),
                      default: _withCtx(() => [
                        _createVNode(_component_VListItemTitle, null, {
                          default: _withCtx(() => [
                            _createTextVNode(_toDisplayString(job.title), 1)
                          ]),
                          _: 2
                        }, 1024),
                        _createVNode(_component_VListItemSubtitle, null, {
                          default: _withCtx(() => [
                            _createTextVNode(_toDisplayString(job.priority) + " · " + _toDisplayString(job.status) + " · " + _toDisplayString(job.trigger), 1)
                          ]),
                          _: 2
                        }, 1024)
                      ]),
                      _: 2
                    }, 1024))
                  }), 128))
                ]),
                _: 1
              })
            ], 64))
          : (nav.value === 'desk')
            ? (_openBlock(), _createElementBlock(_Fragment, { key: 2 }, [
                (!activeJob.value)
                  ? (_openBlock(), _createElementBlock("div", _hoisted_14, "从队列打开一个任务。"))
                  : (_openBlock(), _createElementBlock(_Fragment, { key: 1 }, [
                      _createElementVNode("div", _hoisted_15, _toDisplayString(activeJob.value.title), 1),
                      _createVNode(_sfc_main$1, {
                        graph: graph.value,
                        "current-ms": currentMs.value,
                        langs: config.value.target_languages || ['zh-Hans'],
                        tracks: previewTracks.value,
                        stack: config.value.lang_stack || 'main_bottom',
                        "video-url": pluginApi.value.previewVideoUrl(activeJob.value.job_id),
                        enabled: config.value.preview_enabled !== false,
                        onTime: _cache[9] || (_cache[9] = $event => (currentMs.value = $event))
                      }, null, 8, ["graph", "current-ms", "langs", "tracks", "stack", "video-url", "enabled"]),
                      _createElementVNode("div", _hoisted_16, [
                        (_openBlock(), _createElementBlock(_Fragment, null, _renderList(['dialogue', 'notes', 'stacked', 'sdh'], (track) => {
                          return _createVNode(_component_VChip, {
                            key: track,
                            color: previewTracks.value.includes(track) ? 'primary' : undefined,
                            onClick: $event => (previewTracks.value.includes(track) ? previewTracks.value = previewTracks.value.filter(item => item !== track) : previewTracks.value = [...previewTracks.value, track])
                          }, {
                            default: _withCtx(() => [
                              _createTextVNode(_toDisplayString(track), 1)
                            ]),
                            _: 2
                          }, 1032, ["color", "onClick"])
                        }), 64))
                      ]),
                      _cache[37] || (_cache[37] = _createElementVNode("div", { class: "text-caption mb-1" }, "按时间", -1)),
                      _createElementVNode("div", {
                        class: "ss-timeline mb-4",
                        onClick: seekTimeline
                      }, [
                        (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(graph.value.cues || [], (cue) => {
                          return (_openBlock(), _createElementBlock("button", {
                            key: cue.cue_id,
                            type: "button",
                            class: "ss-cue",
                            style: _normalizeStyle({ left: cueLeft(cue), width: cueWidth(cue) }),
                            onClick: _withModifiers($event => {openCue(cue); currentMs.value = cue.start_ms;}, ["stop"])
                          }, _toDisplayString(Object.values(cue.texts || {})[0]), 13, _hoisted_17))
                        }), 128))
                      ]),
                      _createVNode(_component_VList, null, {
                        default: _withCtx(() => [
                          (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(graph.value.cues || [], (cue) => {
                            return (_openBlock(), _createBlock(_component_VListItem, {
                              key: cue.cue_id,
                              onClick: $event => (openCue(cue))
                            }, {
                              default: _withCtx(() => [
                                _createVNode(_component_VListItemTitle, null, {
                                  default: _withCtx(() => [
                                    _createTextVNode(_toDisplayString(Object.values(cue.texts || {})[0]), 1)
                                  ]),
                                  _: 2
                                }, 1024),
                                _createVNode(_component_VListItemSubtitle, null, {
                                  default: _withCtx(() => [
                                    _createTextVNode(_toDisplayString(cue.start_ms) + " – " + _toDisplayString(cue.end_ms), 1)
                                  ]),
                                  _: 2
                                }, 1024)
                              ]),
                              _: 2
                            }, 1032, ["onClick"]))
                          }), 128))
                        ]),
                        _: 1
                      })
                    ], 64))
              ], 64))
            : (_openBlock(), _createElementBlock(_Fragment, { key: 3 }, [
                _createVNode(_sfc_main$2, {
                  modelValue: config.value,
                  "onUpdate:modelValue": _cache[10] || (_cache[10] = $event => ((config).value = $event)),
                  fields: fields.value,
                  mobile: _unref(isMobile),
                  onDirty: _cache[11] || (_cache[11] = $event => (dirty.value = true)),
                  onTestEndpoint: _cache[12] || (_cache[12] = $event => (pluginApi.value.testEndpoint($event))),
                  onListModels: _cache[13] || (_cache[13] = $event => (pluginApi.value.listModels($event)))
                }, null, 8, ["modelValue", "fields", "mobile"]),
                (dirty.value)
                  ? (_openBlock(), _createElementBlock("div", _hoisted_18, [
                      _createVNode(_component_VBtn, {
                        color: "primary",
                        class: "ss-touch",
                        onClick: saveConfig
                      }, {
                        default: _withCtx(() => [...(_cache[38] || (_cache[38] = [
                          _createTextVNode("保存", -1)
                        ]))]),
                        _: 1
                      }),
                      _createVNode(_component_VBtn, {
                        variant: "text",
                        class: "ss-touch",
                        onClick: _cache[14] || (_cache[14] = $event => (reload()))
                      }, {
                        default: _withCtx(() => [...(_cache[39] || (_cache[39] = [
                          _createTextVNode("恢复", -1)
                        ]))]),
                        _: 1
                      })
                    ]))
                  : _createCommentVNode("", true)
              ], 64))
    ]),
    _createVNode(_component_VBottomSheet, {
      modelValue: jobSheet.value,
      "onUpdate:modelValue": _cache[22] || (_cache[22] = $event => ((jobSheet).value = $event)),
      inset: "",
      rounded: "t-xl"
    }, {
      default: _withCtx(() => [
        _createVNode(_component_VCard, null, {
          default: _withCtx(() => [
            _createVNode(_component_VCardTitle, null, {
              default: _withCtx(() => [
                _createTextVNode(_toDisplayString(activeJob.value?.title), 1)
              ]),
              _: 1
            }),
            _createVNode(_component_VList, null, {
              default: _withCtx(() => [
                (activeJob.value?.job_id)
                  ? (_openBlock(), _createBlock(_component_VListItem, {
                      key: 0,
                      title: "打开工作台",
                      onClick: _cache[15] || (_cache[15] = $event => {openJob(activeJob.value); jobSheet.value = false;})
                    }))
                  : _createCommentVNode("", true),
                (activeJob.value?.status === 'pending')
                  ? (_openBlock(), _createBlock(_component_VListItem, {
                      key: 1,
                      title: "插队",
                      onClick: _cache[16] || (_cache[16] = $event => {cutIn(activeJob.value); jobSheet.value = false;})
                    }))
                  : _createCommentVNode("", true),
                _createVNode(_component_VListItem, {
                  title: "改成 P0",
                  onClick: _cache[17] || (_cache[17] = $event => {changePriority(activeJob.value, 'P0'); jobSheet.value = false;})
                }),
                _createVNode(_component_VListItem, {
                  title: "改成 P1",
                  onClick: _cache[18] || (_cache[18] = $event => {changePriority(activeJob.value, 'P1'); jobSheet.value = false;})
                }),
                _createVNode(_component_VListItem, {
                  title: "改成 P2",
                  onClick: _cache[19] || (_cache[19] = $event => {changePriority(activeJob.value, 'P2'); jobSheet.value = false;})
                }),
                _createVNode(_component_VListItem, {
                  title: "取消",
                  onClick: _cache[20] || (_cache[20] = $event => {cancelJob(activeJob.value); jobSheet.value = false;})
                }),
                (activeJob.value?.path && !activeJob.value?.job_id)
                  ? (_openBlock(), _createBlock(_component_VListItem, {
                      key: 2,
                      title: "入队",
                      onClick: _cache[21] || (_cache[21] = $event => {enqueue(activeJob.value); jobSheet.value = false;})
                    }))
                  : _createCommentVNode("", true)
              ]),
              _: 1
            })
          ]),
          _: 1
        })
      ]),
      _: 1
    }, 8, ["modelValue"]),
    _createVNode(_component_VBottomSheet, {
      modelValue: enqueueSheet.value,
      "onUpdate:modelValue": _cache[26] || (_cache[26] = $event => ((enqueueSheet).value = $event)),
      inset: "",
      rounded: "t-xl"
    }, {
      default: _withCtx(() => [
        _createVNode(_component_VCard, { class: "pa-4" }, {
          default: _withCtx(() => [
            _createVNode(_component_VCardTitle, null, {
              default: _withCtx(() => [...(_cache[40] || (_cache[40] = [
                _createTextVNode("手动提交识别", -1)
              ]))]),
              _: 1
            }),
            _createElementVNode("div", _hoisted_19, "将提交 " + _toDisplayString(enqueueForm.items?.length || 1) + " 个文件", 1),
            _createVNode(_component_VSelect, {
              modelValue: enqueueForm.strategy,
              "onUpdate:modelValue": _cache[23] || (_cache[23] = $event => ((enqueueForm.strategy) = $event)),
              label: "本次策略",
              items: [
          { title: '先搜后译', value: 'search_then_translate' },
          { title: '只搜索', value: 'search_only' },
          { title: '只识别翻译', value: 'translate_only' },
        ]
            }, null, 8, ["modelValue"]),
            _createVNode(_component_VSelect, {
              modelValue: enqueueForm.priority,
              "onUpdate:modelValue": _cache[24] || (_cache[24] = $event => ((enqueueForm.priority) = $event)),
              label: "优先级",
              items: ['P0', 'P1', 'P2']
            }, null, 8, ["modelValue"]),
            _createVNode(_component_VSwitch, {
              modelValue: enqueueForm.force,
              "onUpdate:modelValue": _cache[25] || (_cache[25] = $event => ((enqueueForm.force) = $event)),
              label: "强制入队（忽略已有中字等门禁）",
              "hide-details": "",
              class: "mb-2"
            }, null, 8, ["modelValue"]),
            _createVNode(_component_VBtn, {
              color: "primary",
              block: "",
              class: "ss-touch mt-2",
              loading: submitting.value,
              onClick: confirmEnqueue
            }, {
              default: _withCtx(() => [...(_cache[41] || (_cache[41] = [
                _createTextVNode("确认提交", -1)
              ]))]),
              _: 1
            }, 8, ["loading"])
          ]),
          _: 1
        })
      ]),
      _: 1
    }, 8, ["modelValue"]),
    _createVNode(_component_VBottomSheet, {
      modelValue: cueSheet.value,
      "onUpdate:modelValue": _cache[27] || (_cache[27] = $event => ((cueSheet).value = $event)),
      inset: "",
      rounded: "t-xl"
    }, {
      default: _withCtx(() => [
        (editingCue.value)
          ? (_openBlock(), _createBlock(_component_VCard, {
              key: 0,
              class: "pa-4"
            }, {
              default: _withCtx(() => [
                _createVNode(_component_VCardTitle, null, {
                  default: _withCtx(() => [...(_cache[42] || (_cache[42] = [
                    _createTextVNode("编辑句子", -1)
                  ]))]),
                  _: 1
                }),
                (_openBlock(true), _createElementBlock(_Fragment, null, _renderList((config.value.target_languages || ['zh-Hans']), (lang) => {
                  return (_openBlock(), _createBlock(_component_VTextField, {
                    key: lang,
                    "model-value": editingCue.value.texts[lang],
                    label: _unref(langLabel)(lang),
                    "onUpdate:modelValue": $event => (editingCue.value.texts[lang] = $event)
                  }, null, 8, ["model-value", "label", "onUpdate:modelValue"]))
                }), 128)),
                _createVNode(_component_VBtn, {
                  color: "primary",
                  block: "",
                  class: "ss-touch",
                  onClick: saveCue
                }, {
                  default: _withCtx(() => [...(_cache[43] || (_cache[43] = [
                    _createTextVNode("写回", -1)
                  ]))]),
                  _: 1
                })
              ]),
              _: 1
            }))
          : _createCommentVNode("", true)
      ]),
      _: 1
    }, 8, ["modelValue"])
  ]))
}
}

};

export { _sfc_main as default };
