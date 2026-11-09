import { importShared } from './__federation_fn_import-JrT3xvdd.js';
import { A as ASS_REFERENCE_HEIGHT, o as overlayCss, n as normalizeStyle, a as useMobileViewport, u as useHostInjects, _ as _sfc_main$2, l as langLabel } from './SettingsForm-B-o0RdLw.js';
import { c as createStudioApi } from './studioApi-CUfZ3-lv.js';

const {openBlock:_openBlock$1,createElementBlock:_createElementBlock$1,createCommentVNode:_createCommentVNode$1,createElementVNode:_createElementVNode$1,renderList:_renderList$1,Fragment:_Fragment$1,toDisplayString:_toDisplayString$1,normalizeStyle:_normalizeStyle$1,normalizeClass:_normalizeClass} = await importShared('vue');


const _hoisted_1$1 = ["src"];
const _hoisted_2$1 = {
  key: 1,
  class: "ss-board pa-6 text-center"
};
const _hoisted_3$1 = { key: 3 };

const {computed: computed$1,nextTick,onBeforeUnmount: onBeforeUnmount$1,onMounted: onMounted$1,ref: ref$1,watch: watch$1} = await importShared('vue');


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
  styleConfig: { type: Object, default: () => ({}) },
},
  emits: ['time'],
  setup(__props, { emit: __emit }) {

const props = __props;

const emit = __emit;

const root = ref$1(null);
const frameHeight = ref$1(0);
let observer;

function pictureNode() {
  return root.value?.querySelector('video, .ss-board') || root.value
}

function measure() {
  frameHeight.value = pictureNode()?.clientHeight || 0;
}

function watchPicture() {
  observer?.disconnect();
  observer = null;
  const node = pictureNode();
  if (!node || typeof ResizeObserver === 'undefined') return
  observer = new ResizeObserver(() => measure());
  observer.observe(node);
  measure();
}

onMounted$1(() => watchPicture());

watch$1(() => props.videoUrl, async () => {
  await nextTick();
  watchPicture();
});

onBeforeUnmount$1(() => observer?.disconnect());

const fontScale = computed$1(() => {
  const height = frameHeight.value || root.value?.clientHeight || 0;
  return (height > 0 ? height : 180) / ASS_REFERENCE_HEIGHT
});

const active = computed$1(() => {
  const cues = [...(props.graph?.cues || []), ...(props.graph?.notes || [])];
  return cues.filter(item => item.start_ms <= props.currentMs && props.currentMs < item.end_ms)
});

const dialogue = computed$1(() => {
  const rows = active.value.filter(item => item.kind !== 'note' && item.kind !== 'sdh');
  if (props.tracks.includes('sdh')) {
    return active.value.filter(item => item.kind === 'sdh').concat(rows)
  }
  return rows
});
const notes = computed$1(() => active.value.filter(item => item.kind === 'note'));
const showDialogue = computed$1(() => props.tracks.includes('dialogue') || props.tracks.includes('stacked') || props.tracks.includes('sdh'));

function sourceText(cue) {
  const texts = cue?.texts || {};
  const sourceLang = props.graph?.source_lang;
  return (sourceLang && texts[sourceLang]) || texts.source || Object.values(texts)[0] || ''
}

function textOf(cue, lang) {
  if (!lang) return sourceText(cue)
  const exact = cue?.texts?.[lang] || '';
  if (exact) return exact
  // A single-language preview may show the source as an explicit convenience.
  // Multi-language rows are strict: a missing target stays missing.
  return (props.langs?.length || 0) <= 1 ? sourceText(cue) : ''
}

function dialogueText(lang) {
  return dialogue.value.map(item => textOf(item, lang)).filter(Boolean).join(' ')
}

const dialogueLines = computed$1(() => {
  const seen = new Set();
  return (props.langs || []).flatMap((lang) => {
    const text = dialogueText(lang);
    const identity = text.replace(/\s+/g, '').toLocaleLowerCase();
    if (!text || seen.has(identity)) return []
    seen.add(identity);
    return [{ lang, text }]
  })
});

function lineStyle(index, kind = 'dialogue') {
  const css = overlayCss(props.styleConfig, index, kind, fontScale.value);
  const spec = normalizeStyle(props.styleConfig);
  const margin = Math.round(spec.margin_v * fontScale.value);
  const gap = Math.round(spec.line_gap * fontScale.value);
  if (kind === 'note') {
    return { ...css, top: `${Math.max(8, margin)}px`, bottom: 'auto' }
  }
  const visibleCount = dialogueLines.value?.length || 1;
  const rank = props.stack === 'main_top' ? Math.max(0, visibleCount - index - 1) : index;
  return { ...css, bottom: `${margin + rank * gap}px`, top: 'auto', whiteSpace: 'pre-line' }
}

function onTime(event) {
  emit('time', Math.floor((event.target.currentTime || 0) * 1000));
}

return (_ctx, _cache) => {
  return (__props.enabled)
    ? (_openBlock$1(), _createElementBlock$1("div", {
        key: 0,
        ref_key: "root",
        ref: root,
        class: "ss-player"
      }, [
        (__props.videoUrl)
          ? (_openBlock$1(), _createElementBlock$1("video", {
              key: 0,
              src: __props.videoUrl,
              controls: "",
              playsinline: "",
              onLoadedmetadata: measure,
              onTimeupdate: onTime
            }, null, 40, _hoisted_1$1))
          : (_openBlock$1(), _createElementBlock$1("div", _hoisted_2$1, [...(_cache[0] || (_cache[0] = [
              _createElementVNode$1("div", { class: "text-medium-emphasis" }, "字幕黑板 · 当前没有可播原片（STRM 或需重新打开任务拉取预览）", -1)
            ]))])),
        (__props.tracks.includes('notes'))
          ? (_openBlock$1(true), _createElementBlock$1(_Fragment$1, { key: 2 }, _renderList$1(notes.value, (note) => {
              return (_openBlock$1(), _createElementBlock$1("div", {
                key: note.cue_id,
                class: "ss-overlay note",
                style: _normalizeStyle$1(lineStyle(0, 'note'))
              }, _toDisplayString$1(textOf(note)), 5))
            }), 128))
          : _createCommentVNode$1("", true),
        (showDialogue.value)
          ? (_openBlock$1(), _createElementBlock$1("template", _hoisted_3$1, [
              (_openBlock$1(true), _createElementBlock$1(_Fragment$1, null, _renderList$1(dialogueLines.value, (line, index) => {
                return (_openBlock$1(), _createElementBlock$1("div", {
                  key: line.lang,
                  class: _normalizeClass(["ss-overlay dialogue", [`rank${index + 1}`]]),
                  style: _normalizeStyle$1(lineStyle(index))
                }, _toDisplayString$1(line.text), 7))
              }), 128))
            ]))
          : _createCommentVNode$1("", true)
      ], 512))
    : _createCommentVNode$1("", true)
}
}

};

const {openBlock:_openBlock,createElementBlock:_createElementBlock,createCommentVNode:_createCommentVNode,renderList:_renderList,Fragment:_Fragment,resolveComponent:_resolveComponent,createVNode:_createVNode,toDisplayString:_toDisplayString,createTextVNode:_createTextVNode,withCtx:_withCtx,createElementVNode:_createElementVNode,withModifiers:_withModifiers,createBlock:_createBlock,withKeys:_withKeys,withDirectives:_withDirectives,unref:_unref,normalizeStyle:_normalizeStyle} = await importShared('vue');


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
const _hoisted_10 = { class: "d-flex ga-2 mb-3" };
const _hoisted_11 = {
  class: "d-flex ga-2 mb-3",
  style: {"overflow-x":"auto"}
};
const _hoisted_12 = {
  class: "d-flex ga-2 mb-3",
  style: {"overflow-x":"auto"}
};
const _hoisted_13 = { class: "d-flex ga-2 mb-3" };
const _hoisted_14 = { class: "text-medium-emphasis mb-3" };
const _hoisted_15 = {
  class: "ss-media-poster",
  "aria-hidden": "true"
};
const _hoisted_16 = ["src", "onError"];
const _hoisted_17 = { class: "d-flex ga-2 mb-3" };
const _hoisted_18 = {
  class: "d-flex ga-2 mb-3",
  style: {"overflow-x":"auto"}
};
const _hoisted_19 = {
  key: 0,
  class: "mt-1 w-100"
};
const _hoisted_20 = { class: "text-caption text-medium-emphasis" };
const _hoisted_21 = {
  key: 1,
  class: "mt-2 pa-2 rounded bg-grey-lighten-4 text-caption"
};
const _hoisted_22 = {
  key: 0,
  class: "text-error"
};
const _hoisted_23 = {
  key: 0,
  class: "text-medium-emphasis"
};
const _hoisted_24 = { class: "d-flex align-center justify-space-between mb-2 flex-wrap ga-2" };
const _hoisted_25 = { class: "d-flex ga-2 flex-wrap" };
const _hoisted_26 = { class: "d-flex ga-2 my-3" };
const _hoisted_27 = ["aria-label", "onClick"];
const _hoisted_28 = {
  key: 0,
  class: "ss-savebar pa-3 d-flex ga-2"
};
const _hoisted_29 = { class: "text-medium-emphasis mb-2" };
const _hoisted_30 = { class: "text-medium-emphasis mb-3" };

const {computed,onBeforeUnmount,onMounted,reactive,ref,watch} = await importShared('vue');


const _sfc_main = {
  __name: 'AppPage',
  props: {
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  sourcePluginId: { type: String, default: 'SubtitleStudio' },
  navKey: { type: String, default: 'main' },
  hideTitle: { type: Boolean, default: false },
},
  emits: ['action'],
  setup(__props, { expose: __expose, emit: __emit }) {

const props = __props;
const isMobile = useMobileViewport();
const { toast, confirm } = useHostInjects();
const pluginBase = computed(() => `plugin/${props.pluginId || props.sourcePluginId || 'SubtitleStudio'}`);
const pluginApi = computed(() => createStudioApi(props.api, pluginBase));

const nav = ref('media');
const loading = ref(false);
const dirty = ref(false);
const mediaQuery = ref('');
const mediaType = ref('');
const mediaGroups = ref([]);
const failedPosterUrls = ref(new Set());
const posterObjectUrls = ref({});
const loadingPosterUrls = ref(new Set());
let posterObserver;
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
const previewObjectUrl = ref('');
const cueSheet = ref(false);
const editingCue = ref(null);
const config = ref({});
const fields = ref([]);
const modelPicker = ref({ open: false, endpointId: '', items: [] });
const enqueueSheet = ref(false);
const enqueueForm = reactive({ strategy: 'search_then_translate', priority: 'P0', force: false, overwrite: false, items: [] });
const hideSheet = ref(false);
const hiding = ref(false);
const hideForm = reactive({ paths: [], title: '', deleteSubtitles: false });
const selectedFiles = computed(() => (mediaDetail.value?.files || []).filter(item => selectedPaths.value[item.path]));
const enqueueHasStrm = computed(() => (enqueueForm.items || []).some(item => item?.is_strm || String(item?.path || '').toLowerCase().endsWith('.strm')));

const STATUS_LABELS = { pending: '等待', running: '运行中', success: '成功', failed: '失败', skipped: '跳过', cancelled: '已取消' };
const TRIGGER_LABELS = { event: '事件', watch: '监控', strm: 'STRM', manual: '手动', workflow: '工作流', command: '命令' };
const TRACK_LABELS = { dialogue: '对白', notes: '顶注', stacked: '叠行', sdh: 'SDH' };
const statusFilters = [
  { value: '', label: '全部' },
  { value: 'pending', label: '等待' },
  { value: 'running', label: '运行中' },
  { value: 'failed', label: '失败' },
  { value: 'success', label: '成功' },
  { value: 'skipped', label: '跳过' },
  { value: 'cancelled', label: '已取消' },
];
const expandedJob = ref(null);
function statusLabel(s) { return STATUS_LABELS[s] || s }
function triggerLabel(t) { return TRIGGER_LABELS[t] || t }
function trackLabel(t) { return TRACK_LABELS[t] || t }
function markPosterFailed(url) {
  failedPosterUrls.value = new Set(failedPosterUrls.value).add(url);
}
function isProxyPoster(url) {
  if (typeof url !== 'string' || !/^https?:\/\//i.test(url)) return false
  try {
    const host = new URL(url).hostname.toLowerCase();
    return host === 'doubanio.com' || host.endsWith('.doubanio.com') || host === 'tmdb.org' || host.endsWith('.tmdb.org')
  } catch {
    return false
  }
}
async function loadPoster(url) {
  if (!url || posterObjectUrls.value[url] || loadingPosterUrls.value.has(url) || failedPosterUrls.value.has(url)) return
  loadingPosterUrls.value = new Set(loadingPosterUrls.value).add(url);
  try {
    const response = await props.api.get('system/img/0', {
      params: { imgurl: url },
      responseType: 'blob',
      feedback: 'silent',
    });
    const blob = response instanceof Blob ? response : response?.data;
    if (!(blob instanceof Blob) || !blob.size) throw new Error('图片代理未返回图片')
    posterObjectUrls.value = { ...posterObjectUrls.value, [url]: URL.createObjectURL(blob) };
  } catch {
    markPosterFailed(url);
  } finally {
    const next = new Set(loadingPosterUrls.value);
    next.delete(url);
    loadingPosterUrls.value = next;
  }
}
const vPoster = {
  mounted(el, binding) {
    const url = binding.value;
    if (!isProxyPoster(url)) return
    posterObserver ||= new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return
        const imageUrl = entry.target.dataset.posterUrl;
        posterObserver.unobserve(entry.target);
        loadPoster(imageUrl);
      });
    }, { rootMargin: '120px' });
    el.dataset.posterUrl = url;
    posterObserver.observe(el);
  },
  unmounted(el) {
    posterObserver?.unobserve(el);
  },
};
function toggleJobDetail(job) { expandedJob.value = expandedJob.value === job.job_id ? null : job.job_id; }
function formatMs(ms) {
  const total = Math.max(0, Math.floor(Number(ms) || 0) / 1000);
  const m = Math.floor(total / 60);
  const s = Math.floor(total % 60);
  const frac = Math.floor((Number(ms) || 0) % 1000);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(frac).padStart(3, '0')}`
}

const tabs = [
  { value: 'media', title: '媒体', icon: 'mdi-filmstrip' },
  { value: 'jobs', title: '队列', icon: 'mdi-playlist-play' },
  { value: 'desk', title: '工作台', icon: 'mdi-subtitles-outline' },
  { value: 'settings', title: '设置', icon: 'mdi-cog-outline' },
];

const duration = computed(() => Math.max(graph.value.duration_ms || 1, ...(graph.value.cues || []).map(item => item.end_ms || 0), 1));
const subtitleFilter = ref('');
const selectedGroupIds = ref({});

function fileHasSubtitle(file) {
  return !!(file?.sidecars?.length || file?.already_processed)
}

const visibleGroups = computed(() => (mediaGroups.value || []).map(group => {
  const files = (group.files || []).filter(file => {
    if (subtitleFilter.value === 'has') return fileHasSubtitle(file)
    if (subtitleFilter.value === 'missing') return !fileHasSubtitle(file)
    return true
  });
  return files.length ? { ...group, files, file_count: files.length } : null
}).filter(Boolean));

const selectedGroupFiles = computed(() => visibleGroups.value
  .filter(group => selectedGroupIds.value[group.id])
  .flatMap(group => group.files || []));

watch(nav, async (value, prev) => {
  if (prev === 'settings' && value !== 'settings' && dirty.value) {
    if (!(await confirm('设置尚未保存，确定离开？'))) {
      nav.value = 'settings';
    }
  }
});

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
  try {
    const data = await pluginApi.value.media(mediaQuery.value, mediaType.value);
    mediaGroups.value = data?.groups || [];
    mediaCounts.value = data?.counts || { files: 0, groups: 0 };
  } catch (error) {
    toast.error?.(error?.message || '加载媒体失败');
  }
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

function toggleGroup(id, value) {
  selectedGroupIds.value = { ...selectedGroupIds.value, [id]: value };
}

function toggleVisibleGroups(value) {
  selectedGroupIds.value = Object.fromEntries(visibleGroups.value.map(group => [group.id, value]));
}

async function submitGroups() {
  const files = selectedGroupFiles.value;
  if (!files.length) {
    toast.error?.('先勾选要提交的影片');
    return
  }
  const completedCount = files.filter(item => item.already_processed).length;
  if (completedCount && !(await confirm(`${completedCount} 个文件已有成功处理记录。再次提交会重新执行并消耗模型资源，仍要继续吗？`))) return
  prepareEnqueue(files);
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

let pollTimer = null;
async function loadJobs() {
  try {
    const data = await pluginApi.value.jobs(jobsQuery.value, jobStatus.value);
    jobs.value = data?.items || [];
  } catch (error) {
    toast.error?.(error?.message || '加载队列失败');
  }
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  if (jobs.value.some(j => j.status === 'running' || j.status === 'pending')) {
    pollTimer = setInterval(() => { loadJobs(); }, 1500);
  }
}

async function saveConfig() {
  try {
    const saved = await pluginApi.value.saveConfig(config.value);
    config.value = { ...(saved || config.value) };
    dirty.value = false;
    toast.success?.('已保存');
  } catch (error) {
    toast.error?.(error?.message || '保存失败');
  }
}

function revokePreview() {
  if (previewObjectUrl.value) {
    URL.revokeObjectURL(previewObjectUrl.value);
    previewObjectUrl.value = '';
  }
}

async function loadPreview(job) {
  revokePreview();
  if (!job?.job_id || job.is_strm) return
  try {
    const blob = await pluginApi.value.fetchPreviewVideo(job.job_id);
    if (blob) previewObjectUrl.value = URL.createObjectURL(blob);
  } catch {
    previewObjectUrl.value = '';
  }
}

function prepareEnqueue(items) {
  enqueueForm.items = items;
  enqueueForm.force = false;
  enqueueForm.overwrite = items.some(item => item.already_processed);
  enqueueSheet.value = true;
}

async function submitSelected() {
  if (!selectedFiles.value.length) {
    toast.error?.('先勾选要处理的文件');
    return
  }
  const completedCount = selectedFiles.value.filter(item => item.already_processed).length;
  if (completedCount && !(await confirm(`${completedCount} 个文件已有成功处理记录。再次提交会重新执行并消耗模型资源，仍要继续吗？`))) return
  prepareEnqueue(selectedFiles.value);
}

async function confirmEnqueue() {
  submitting.value = true;
  try {
    const items = enqueueForm.items?.length ? enqueueForm.items : [enqueueForm];
    const data = await pluginApi.value.createJobs({
      items,
      priority: enqueueForm.priority,
      force: enqueueForm.force,
      overwrite: enqueueForm.overwrite,
    });
    const created = data?.count || data?.items?.length || 0;
    const skipped = data?.skipped?.length || 0;
    if (skipped) {
      const reasons = (data.skipped || []).slice(0, 3).map(item => item.reason || '跳过').join('；');
      toast.success?.(`已入队 ${created}，跳过 ${skipped}${reasons ? `：${reasons}` : ''}`);
    } else {
      toast.success?.(`已入队 ${created} 条`);
    }
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
  try {
    activeJob.value = job;
    graph.value = (await pluginApi.value.cues(job.job_id)) || { cues: [], notes: [], briefing: [] };
    await loadPreview(job);
    nav.value = 'desk';
  } catch (error) {
    toast.error?.(error?.message || '打开工作台失败');
  }
}

async function cutIn(job) {
  if (!(await confirm('把这个任务插到队首？'))) return
  try {
    await pluginApi.value.cutIn(job.job_id);
    await loadJobs();
  } catch (error) {
    toast.error?.(error?.message || '插队失败');
  }
}

async function changePriority(job, priority) {
  if (!(await confirm(`改成 ${priority}？`))) return
  try {
    await pluginApi.value.setPriority(job.job_id, priority);
    await loadJobs();
  } catch (error) {
    toast.error?.(error?.message || '改优先级失败');
  }
}

async function cancelJob(job) {
  if (!(await confirm('取消这个任务？'))) return
  try {
    await pluginApi.value.cancel(job.job_id);
    toast.success?.('已取消');
    await loadJobs();
  } catch (error) {
    toast.error?.(error?.message || '取消失败');
  }
}

function askHide(paths, title) {
  const list = [...new Set((paths || []).filter(Boolean))];
  if (!list.length) {
    toast.error?.('先勾选要从列表移除的文件');
    return
  }
  hideForm.paths = list;
  hideForm.title = title || '';
  hideForm.deleteSubtitles = false;
  hideSheet.value = true;
}

async function confirmHide() {
  hiding.value = true;
  try {
    const data = await pluginApi.value.hideMedia({
      paths: hideForm.paths,
      delete_subtitles: hideForm.deleteSubtitles,
    });
    const removed = data?.deleted_files?.length || 0;
    toast.success?.(hideForm.deleteSubtitles ? `已从列表移除，并删除 ${removed} 个字幕文件` : '已从列表移除');
    hideSheet.value = false;
    if (mediaDetail.value) {
      const gone = new Set(hideForm.paths);
      const files = (mediaDetail.value.files || []).filter(item => !gone.has(item.path));
      mediaDetail.value = files.length ? { ...mediaDetail.value, files, file_count: files.length } : null;
    }
    await loadMedia();
  } catch (error) {
    toast.error?.(error?.message || '移除失败');
  } finally {
    hiding.value = false;
  }
}

async function retryJob(job, overwrite = false) {
  try {
    await pluginApi.value.retry(job.job_id, { overwrite: !!overwrite });
    toast.success?.(overwrite ? '已重新入队，将覆盖已有字幕' : '已重新入队');
    jobSheet.value = false;
    nav.value = 'jobs';
    await loadJobs();
  } catch (error) {
    toast.error?.(error?.message || '重跑失败');
  }
}

async function searchAgain(job) {
  try {
    const data = await pluginApi.value.searchJob(job.job_id);
    toast.success?.(`搜索到 ${(data?.items || []).length} 条结果`);
  } catch (error) {
    toast.error?.(error?.message || '搜索失败');
  }
}

async function exportCurrent() {
  if (!activeJob.value?.job_id) return
  if (!(await confirm('按当前 CueGraph 写盘？会遵循设置里的覆盖策略。'))) return
  try {
    const data = await pluginApi.value.exportJob(activeJob.value.job_id);
    const written = (data?.items || []).filter(item => item.written).length;
    const skipped = (data?.items || []).filter(item => !item.written).length;
    toast.success?.(`已写 ${written} 个文件${skipped ? `，跳过 ${skipped}` : ''}`);
  } catch (error) {
    toast.error?.(error?.message || '写盘失败');
  }
}

async function downloadAssPreview() {
  if (!activeJob.value?.job_id) return
  try {
    const text = await pluginApi.value.fetchPreviewAss(activeJob.value.job_id);
    const blob = new Blob([text || ''], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${activeJob.value.title || 'preview'}.ass`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    toast.error?.(error?.message || '预览 ASS 失败');
  }
}

function openCue(cue) {
  editingCue.value = {
    ...cue,
    texts: { ...(cue.texts || {}) },
    start_ms: cue.start_ms,
    end_ms: cue.end_ms,
  };
  cueSheet.value = true;
}

async function saveCue() {
  try {
    await pluginApi.value.saveCue(activeJob.value.job_id, editingCue.value.cue_id, {
      texts: editingCue.value.texts,
      start_ms: Number(editingCue.value.start_ms),
      end_ms: Number(editingCue.value.end_ms),
    });
    cueSheet.value = false;
    graph.value = await pluginApi.value.cues(activeJob.value.job_id);
    toast.success?.('已改句');
  } catch (error) {
    toast.error?.(error?.message || '保存句子失败');
  }
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

async function onListModels(endpoint) {
  try {
    const data = await pluginApi.value.listModels(endpoint);
    const items = data?.items || [];
    if (!items.length) {
      toast.error?.('没有拉到模型');
      return
    }
    modelPicker.value = { open: true, endpointId: endpoint.endpoint_id, items };
  } catch (error) {
    toast.error?.(error?.message || '拉模型失败');
  }
}

function applyModel(model) {
  const rows = (config.value.openai_endpoints || []).map(item => (
    item.endpoint_id === modelPicker.value.endpointId ? { ...item, model } : item
  ));
  config.value = { ...config.value, openai_endpoints: rows };
  dirty.value = true;
  modelPicker.value.open = false;
  toast.success?.(`已选模型 ${model}`);
}

onMounted(reload);
onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer);
  revokePreview();
  posterObserver?.disconnect();
  Object.values(posterObjectUrls.value).forEach(url => URL.revokeObjectURL(url));
});
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
  const _component_VProgressLinear = _resolveComponent("VProgressLinear");
  const _component_VCardTitle = _resolveComponent("VCardTitle");
  const _component_VCard = _resolveComponent("VCard");
  const _component_VBottomSheet = _resolveComponent("VBottomSheet");
  const _component_VSelect = _resolveComponent("VSelect");
  const _component_VSwitch = _resolveComponent("VSwitch");
  const _component_VDialog = _resolveComponent("VDialog");

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
                      "aria-label": "返回",
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
                      default: _withCtx(() => [...(_cache[42] || (_cache[42] = [
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
                      default: _withCtx(() => [...(_cache[43] || (_cache[43] = [
                        _createTextVNode("清空", -1)
                      ]))]),
                      _: 1
                    }),
                    _createVNode(_component_VBtn, {
                      size: "small",
                      variant: "text",
                      class: "ss-touch",
                      disabled: !selectedFiles.value.length,
                      onClick: _cache[4] || (_cache[4] = $event => (askHide(selectedFiles.value.map(item => item.path), mediaDetail.value.title)))
                    }, {
                      default: _withCtx(() => [...(_cache[44] || (_cache[44] = [
                        _createTextVNode("移除所选", -1)
                      ]))]),
                      _: 1
                    }, 8, ["disabled"])
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
                          append: _withCtx(() => [
                            _createVNode(_component_VBtn, {
                              icon: "mdi-close",
                              class: "ss-touch",
                              "aria-label": "从列表移除",
                              onClick: _withModifiers($event => (askHide([item.path], fileLabel(item))), ["stop"])
                            }, null, 8, ["onClick"])
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
                                _createTextVNode(_toDisplayString(item.sidecars?.length || 0) + " 条外挂" + _toDisplayString(item.is_strm ? ' · STRM' : '') + _toDisplayString(item.already_processed ? ' · 已成功处理' : ''), 1)
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
                      _createTextVNode(" 提交处理（" + _toDisplayString(selectedFiles.value.length) + "） ", 1)
                    ]),
                    _: 1
                  }, 8, ["disabled"])
                ]))
              : (_openBlock(), _createElementBlock(_Fragment, { key: 1 }, [
                  _createElementVNode("div", _hoisted_10, [
                    _createVNode(_component_VTextField, {
                      modelValue: mediaQuery.value,
                      "onUpdate:modelValue": _cache[5] || (_cache[5] = $event => ((mediaQuery).value = $event)),
                      label: "搜索标题或文件名",
                      "prepend-inner-icon": "mdi-magnify",
                      "hide-details": "",
                      class: "flex-grow-1",
                      onKeyup: _withKeys(loadMedia, ["enter"])
                    }, null, 8, ["modelValue"]),
                    _createVNode(_component_VBtn, {
                      class: "ss-touch",
                      loading: loading.value,
                      onClick: loadMedia
                    }, {
                      default: _withCtx(() => [...(_cache[45] || (_cache[45] = [
                        _createTextVNode("搜索", -1)
                      ]))]),
                      _: 1
                    }, 8, ["loading"])
                  ]),
                  _createElementVNode("div", _hoisted_11, [
                    _createVNode(_component_VChip, {
                      color: !mediaType.value ? 'primary' : undefined,
                      onClick: _cache[6] || (_cache[6] = $event => {mediaType.value = ''; loadMedia();})
                    }, {
                      default: _withCtx(() => [...(_cache[46] || (_cache[46] = [
                        _createTextVNode("全部", -1)
                      ]))]),
                      _: 1
                    }, 8, ["color"]),
                    _createVNode(_component_VChip, {
                      color: mediaType.value === 'movie' ? 'primary' : undefined,
                      onClick: _cache[7] || (_cache[7] = $event => {mediaType.value = 'movie'; loadMedia();})
                    }, {
                      default: _withCtx(() => [...(_cache[47] || (_cache[47] = [
                        _createTextVNode("电影", -1)
                      ]))]),
                      _: 1
                    }, 8, ["color"]),
                    _createVNode(_component_VChip, {
                      color: mediaType.value === 'tv' ? 'primary' : undefined,
                      onClick: _cache[8] || (_cache[8] = $event => {mediaType.value = 'tv'; loadMedia();})
                    }, {
                      default: _withCtx(() => [...(_cache[48] || (_cache[48] = [
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
                      default: _withCtx(() => [...(_cache[49] || (_cache[49] = [
                        _createTextVNode("拉取媒体库", -1)
                      ]))]),
                      _: 1
                    }, 8, ["loading"])
                  ]),
                  _createElementVNode("div", _hoisted_12, [
                    _createVNode(_component_VChip, {
                      color: !subtitleFilter.value ? 'primary' : undefined,
                      onClick: _cache[9] || (_cache[9] = $event => (subtitleFilter.value = ''))
                    }, {
                      default: _withCtx(() => [...(_cache[50] || (_cache[50] = [
                        _createTextVNode("字幕不限", -1)
                      ]))]),
                      _: 1
                    }, 8, ["color"]),
                    _createVNode(_component_VChip, {
                      color: subtitleFilter.value === 'has' ? 'primary' : undefined,
                      onClick: _cache[10] || (_cache[10] = $event => (subtitleFilter.value = 'has'))
                    }, {
                      default: _withCtx(() => [...(_cache[51] || (_cache[51] = [
                        _createTextVNode("已有字幕", -1)
                      ]))]),
                      _: 1
                    }, 8, ["color"]),
                    _createVNode(_component_VChip, {
                      color: subtitleFilter.value === 'missing' ? 'primary' : undefined,
                      onClick: _cache[11] || (_cache[11] = $event => (subtitleFilter.value = 'missing'))
                    }, {
                      default: _withCtx(() => [...(_cache[52] || (_cache[52] = [
                        _createTextVNode("没有字幕", -1)
                      ]))]),
                      _: 1
                    }, 8, ["color"])
                  ]),
                  _createElementVNode("div", _hoisted_13, [
                    _createVNode(_component_VBtn, {
                      size: "small",
                      variant: "text",
                      class: "ss-touch",
                      onClick: _cache[12] || (_cache[12] = $event => (toggleVisibleGroups(true)))
                    }, {
                      default: _withCtx(() => [...(_cache[53] || (_cache[53] = [
                        _createTextVNode("全选", -1)
                      ]))]),
                      _: 1
                    }),
                    _createVNode(_component_VBtn, {
                      size: "small",
                      variant: "text",
                      class: "ss-touch",
                      onClick: _cache[13] || (_cache[13] = $event => (toggleVisibleGroups(false)))
                    }, {
                      default: _withCtx(() => [...(_cache[54] || (_cache[54] = [
                        _createTextVNode("清空", -1)
                      ]))]),
                      _: 1
                    }),
                    _createVNode(_component_VBtn, {
                      size: "small",
                      color: "primary",
                      class: "ss-touch",
                      disabled: !selectedGroupFiles.value.length,
                      onClick: submitGroups
                    }, {
                      default: _withCtx(() => [
                        _createTextVNode("提交所选（" + _toDisplayString(selectedGroupFiles.value.length) + "）", 1)
                      ]),
                      _: 1
                    }, 8, ["disabled"])
                  ]),
                  _createElementVNode("div", _hoisted_14, "整理记录 " + _toDisplayString(visibleGroups.value.length) + " 部 · " + _toDisplayString(visibleGroups.value.reduce((sum, group) => sum + (group.file_count || 0), 0)) + " 个文件", 1),
                  (!visibleGroups.value.length)
                    ? (_openBlock(), _createBlock(_component_VAlert, {
                        key: 0,
                        type: "info",
                        variant: "tonal",
                        class: "mb-3"
                      }, {
                        default: _withCtx(() => [
                          _createTextVNode(_toDisplayString(subtitleFilter.value ? '这个筛选下没有影片。' : '没有本地媒体。点「拉取媒体库」读取 MoviePilot 整理记录；先在 MoviePilot 里整理入库后才会出现。这里不是 Emby/Jellyfin 在线目录。'), 1)
                        ]),
                        _: 1
                      }))
                    : _createCommentVNode("", true),
                  _createVNode(_component_VList, null, {
                    default: _withCtx(() => [
                      (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(visibleGroups.value, (item) => {
                        return (_openBlock(), _createBlock(_component_VListItem, {
                          key: item.id,
                          class: "ss-card mb-2",
                          onClick: $event => (openGroup(item))
                        }, {
                          prepend: _withCtx(() => [
                            _createVNode(_component_VCheckbox, {
                              "model-value": !!selectedGroupIds.value[item.id],
                              "hide-details": "",
                              onClick: _cache[14] || (_cache[14] = _withModifiers(() => {}, ["stop"])),
                              "onUpdate:modelValue": $event => (toggleGroup(item.id, $event))
                            }, null, 8, ["model-value", "onUpdate:modelValue"]),
                            _createElementVNode("div", _hoisted_15, [
                              (item.poster && !failedPosterUrls.value.has(item.poster))
                                ? _withDirectives((_openBlock(), _createElementBlock("img", {
                                    key: 0,
                                    src: isProxyPoster(item.poster) ? posterObjectUrls.value[item.poster] : item.poster,
                                    alt: "",
                                    loading: "lazy",
                                    decoding: "async",
                                    onError: $event => (markPosterFailed(item.poster))
                                  }, null, 40, _hoisted_16)), [
                                    [vPoster, item.poster]
                                  ])
                                : (_openBlock(), _createBlock(_component_VIcon, {
                                    key: 1,
                                    icon: "mdi-movie-open-outline"
                                  }))
                            ])
                          ]),
                          append: _withCtx(() => [
                            _createVNode(_component_VBtn, {
                              icon: "mdi-close",
                              class: "ss-touch",
                              "aria-label": "从列表移除",
                              onClick: _withModifiers($event => (askHide((item.files || []).map(file => file.path), item.title)), ["stop"])
                            }, null, 8, ["onClick"]),
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
                                _createTextVNode(_toDisplayString(item.type === 'tv' ? '剧集' : '电影') + " · " + _toDisplayString(item.file_count || item.files?.length || 0) + " 个文件 · " + _toDisplayString(item.library_name) + " ", 1),
                                (item.processed_count)
                                  ? (_openBlock(), _createElementBlock(_Fragment, { key: 0 }, [
                                      _createTextVNode(" · 已处理 " + _toDisplayString(item.processed_count) + "/" + _toDisplayString(item.file_count || item.files?.length || 0), 1)
                                    ], 64))
                                  : _createCommentVNode("", true)
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
        : (nav.value === 'jobs')
          ? (_openBlock(), _createElementBlock(_Fragment, { key: 1 }, [
              _createElementVNode("div", _hoisted_17, [
                _createVNode(_component_VTextField, {
                  modelValue: jobsQuery.value,
                  "onUpdate:modelValue": _cache[15] || (_cache[15] = $event => ((jobsQuery).value = $event)),
                  label: "搜索标题",
                  "prepend-inner-icon": "mdi-magnify",
                  "hide-details": "",
                  class: "flex-grow-1",
                  onKeyup: _withKeys(loadJobs, ["enter"])
                }, null, 8, ["modelValue"]),
                _createVNode(_component_VBtn, {
                  class: "ss-touch",
                  onClick: loadJobs
                }, {
                  default: _withCtx(() => [...(_cache[55] || (_cache[55] = [
                    _createTextVNode("搜索", -1)
                  ]))]),
                  _: 1
                })
              ]),
              _createElementVNode("div", _hoisted_18, [
                (_openBlock(), _createElementBlock(_Fragment, null, _renderList(statusFilters, (item) => {
                  return _createVNode(_component_VChip, {
                    key: item.value,
                    color: jobStatus.value === item.value ? 'primary' : undefined,
                    onClick: $event => {jobStatus.value = item.value; loadJobs();}
                  }, {
                    default: _withCtx(() => [
                      _createTextVNode(_toDisplayString(item.label), 1)
                    ]),
                    _: 2
                  }, 1032, ["color", "onClick"])
                }), 64))
              ]),
              (!jobs.value.length)
                ? (_openBlock(), _createBlock(_component_VAlert, {
                    key: 0,
                    type: "info",
                    variant: "tonal",
                    class: "mb-3"
                  }, {
                    default: _withCtx(() => [...(_cache[56] || (_cache[56] = [
                      _createTextVNode("队列是空的。从「媒体」勾选文件提交处理。", -1)
                    ]))]),
                    _: 1
                  }))
                : _createCommentVNode("", true),
              _createVNode(_component_VList, null, {
                default: _withCtx(() => [
                  (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(jobs.value, (job) => {
                    return (_openBlock(), _createBlock(_component_VListItem, {
                      key: job.job_id,
                      class: "ss-card mb-2",
                      onClick: $event => (toggleJobDetail(job))
                    }, {
                      append: _withCtx(() => [
                        (!_unref(isMobile) && job.status === 'pending')
                          ? (_openBlock(), _createBlock(_component_VBtn, {
                              key: 0,
                              size: "small",
                              class: "ss-touch",
                              onClick: _withModifiers($event => (cutIn(job)), ["stop"])
                            }, {
                              default: _withCtx(() => [...(_cache[57] || (_cache[57] = [
                                _createTextVNode("插队", -1)
                              ]))]),
                              _: 1
                            }, 8, ["onClick"]))
                          : _createCommentVNode("", true),
                        (!_unref(isMobile) && job.status === 'success')
                          ? (_openBlock(), _createBlock(_component_VBtn, {
                              key: 1,
                              size: "small",
                              class: "ss-touch",
                              onClick: _withModifiers($event => (retryJob(job, true)), ["stop"])
                            }, {
                              default: _withCtx(() => [...(_cache[58] || (_cache[58] = [
                                _createTextVNode("覆盖重译", -1)
                              ]))]),
                              _: 1
                            }, 8, ["onClick"]))
                          : _createCommentVNode("", true),
                        (!_unref(isMobile) && (job.status === 'failed' || job.status === 'cancelled'))
                          ? (_openBlock(), _createBlock(_component_VBtn, {
                              key: 2,
                              size: "small",
                              class: "ss-touch",
                              onClick: _withModifiers($event => (retryJob(job)), ["stop"])
                            }, {
                              default: _withCtx(() => [...(_cache[59] || (_cache[59] = [
                                _createTextVNode("重跑", -1)
                              ]))]),
                              _: 1
                            }, 8, ["onClick"]))
                          : _createCommentVNode("", true),
                        (!_unref(isMobile))
                          ? (_openBlock(), _createBlock(_component_VBtn, {
                              key: 3,
                              size: "small",
                              variant: "text",
                              class: "ss-touch",
                              onClick: _withModifiers($event => (openJob(job)), ["stop"])
                            }, {
                              default: _withCtx(() => [...(_cache[60] || (_cache[60] = [
                                _createTextVNode("工作台", -1)
                              ]))]),
                              _: 1
                            }, 8, ["onClick"]))
                          : _createCommentVNode("", true),
                        _createVNode(_component_VBtn, {
                          icon: "mdi-dots-horizontal",
                          class: "ss-touch",
                          "aria-label": "更多",
                          onClick: _withModifiers($event => (moreJob(job)), ["stop"])
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
                            _createTextVNode(_toDisplayString(job.priority) + " · " + _toDisplayString(statusLabel(job.status)) + " · " + _toDisplayString(triggerLabel(job.trigger)) + _toDisplayString(job.is_strm ? ' · STRM' : ''), 1)
                          ]),
                          _: 2
                        }, 1024),
                        (job.status === 'running' || job.progress > 0)
                          ? (_openBlock(), _createElementBlock("div", _hoisted_19, [
                              _createVNode(_component_VProgressLinear, {
                                "model-value": job.progress,
                                height: "6",
                                rounded: "",
                                color: "primary"
                              }, null, 8, ["model-value"]),
                              _createElementVNode("div", _hoisted_20, _toDisplayString(job.stage_detail) + " · " + _toDisplayString(job.progress) + "%", 1)
                            ]))
                          : _createCommentVNode("", true),
                        (expandedJob.value === job.job_id)
                          ? (_openBlock(), _createElementBlock("div", _hoisted_21, [
                              _createElementVNode("div", null, "阶段：" + _toDisplayString(job.stage_detail || '—'), 1),
                              _createElementVNode("div", null, "进度：" + _toDisplayString(job.progress) + "%", 1),
                              _createElementVNode("div", null, "路径：" + _toDisplayString(job.path), 1),
                              (job.error)
                                ? (_openBlock(), _createElementBlock("div", _hoisted_22, "错误：" + _toDisplayString(job.error), 1))
                                : _createCommentVNode("", true)
                            ]))
                          : _createCommentVNode("", true)
                      ]),
                      _: 2
                    }, 1032, ["onClick"]))
                  }), 128))
                ]),
                _: 1
              })
            ], 64))
          : (nav.value === 'desk')
            ? (_openBlock(), _createElementBlock(_Fragment, { key: 2 }, [
                (!activeJob.value)
                  ? (_openBlock(), _createElementBlock("div", _hoisted_23, "从队列打开一个任务。"))
                  : (_openBlock(), _createElementBlock(_Fragment, { key: 1 }, [
                      _createElementVNode("div", _hoisted_24, [
                        _createElementVNode("div", null, _toDisplayString(activeJob.value.title), 1),
                        _createElementVNode("div", _hoisted_25, [
                          _createVNode(_component_VBtn, {
                            size: "small",
                            color: "primary",
                            class: "ss-touch",
                            onClick: exportCurrent
                          }, {
                            default: _withCtx(() => [...(_cache[61] || (_cache[61] = [
                              _createTextVNode("按当前图写盘", -1)
                            ]))]),
                            _: 1
                          }),
                          _createVNode(_component_VBtn, {
                            size: "small",
                            variant: "tonal",
                            class: "ss-touch",
                            onClick: _cache[16] || (_cache[16] = $event => (searchAgain(activeJob.value)))
                          }, {
                            default: _withCtx(() => [...(_cache[62] || (_cache[62] = [
                              _createTextVNode("重新搜索", -1)
                            ]))]),
                            _: 1
                          }),
                          _createVNode(_component_VBtn, {
                            size: "small",
                            variant: "text",
                            class: "ss-touch",
                            onClick: downloadAssPreview
                          }, {
                            default: _withCtx(() => [...(_cache[63] || (_cache[63] = [
                              _createTextVNode("预览 ASS", -1)
                            ]))]),
                            _: 1
                          })
                        ])
                      ]),
                      _createVNode(_sfc_main$1, {
                        graph: graph.value,
                        "current-ms": currentMs.value,
                        langs: config.value.target_languages || ['zh-Hans'],
                        tracks: previewTracks.value,
                        stack: config.value.lang_stack || 'main_bottom',
                        "style-config": config.value.ass_style || {},
                        "video-url": previewObjectUrl.value,
                        enabled: config.value.preview_enabled !== false,
                        onTime: _cache[17] || (_cache[17] = $event => (currentMs.value = $event))
                      }, null, 8, ["graph", "current-ms", "langs", "tracks", "stack", "style-config", "video-url", "enabled"]),
                      _createElementVNode("div", _hoisted_26, [
                        (_openBlock(), _createElementBlock(_Fragment, null, _renderList(['dialogue', 'notes', 'stacked', 'sdh'], (track) => {
                          return _createVNode(_component_VChip, {
                            key: track,
                            color: previewTracks.value.includes(track) ? 'primary' : undefined,
                            onClick: $event => (previewTracks.value.includes(track) ? previewTracks.value = previewTracks.value.filter(item => item !== track) : previewTracks.value = [...previewTracks.value, track])
                          }, {
                            default: _withCtx(() => [
                              _createTextVNode(_toDisplayString(trackLabel(track)), 1)
                            ]),
                            _: 2
                          }, 1032, ["color", "onClick"])
                        }), 64))
                      ]),
                      _cache[65] || (_cache[65] = _createElementVNode("div", { class: "text-caption mb-1" }, "按时间", -1)),
                      _createElementVNode("div", {
                        class: "ss-timeline mb-4",
                        onClick: seekTimeline
                      }, [
                        (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(graph.value.cues || [], (cue) => {
                          return (_openBlock(), _createElementBlock("button", {
                            key: cue.cue_id,
                            type: "button",
                            class: "ss-cue",
                            "aria-label": Object.values(cue.texts || {})[0] || '字幕句',
                            style: _normalizeStyle({ left: cueLeft(cue), width: cueWidth(cue) }),
                            onClick: _withModifiers($event => {openCue(cue); currentMs.value = cue.start_ms;}, ["stop"])
                          }, _toDisplayString(Object.values(cue.texts || {})[0]), 13, _hoisted_27))
                        }), 128))
                      ]),
                      (!(graph.value.cues || []).length)
                        ? (_openBlock(), _createBlock(_component_VAlert, {
                            key: 0,
                            type: "info",
                            variant: "tonal",
                            class: "mb-3"
                          }, {
                            default: _withCtx(() => [...(_cache[64] || (_cache[64] = [
                              _createTextVNode("这个任务还没有 CueGraph。等任务跑完或先搜索/识别。", -1)
                            ]))]),
                            _: 1
                          }))
                        : _createCommentVNode("", true),
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
                                    _createTextVNode(_toDisplayString(formatMs(cue.start_ms)) + " – " + _toDisplayString(formatMs(cue.end_ms)), 1)
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
                  "onUpdate:modelValue": _cache[18] || (_cache[18] = $event => ((config).value = $event)),
                  fields: fields.value,
                  mobile: _unref(isMobile),
                  onDirty: _cache[19] || (_cache[19] = $event => (dirty.value = true)),
                  onTestEndpoint: _cache[20] || (_cache[20] = $event => (pluginApi.value.testEndpoint($event))),
                  onListModels: onListModels
                }, null, 8, ["modelValue", "fields", "mobile"]),
                (dirty.value)
                  ? (_openBlock(), _createElementBlock("div", _hoisted_28, [
                      _createVNode(_component_VBtn, {
                        color: "primary",
                        class: "ss-touch",
                        onClick: saveConfig
                      }, {
                        default: _withCtx(() => [...(_cache[66] || (_cache[66] = [
                          _createTextVNode("保存", -1)
                        ]))]),
                        _: 1
                      }),
                      _createVNode(_component_VBtn, {
                        variant: "text",
                        class: "ss-touch",
                        onClick: _cache[21] || (_cache[21] = $event => (reload()))
                      }, {
                        default: _withCtx(() => [...(_cache[67] || (_cache[67] = [
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
      "onUpdate:modelValue": _cache[31] || (_cache[31] = $event => ((jobSheet).value = $event)),
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
                      onClick: _cache[22] || (_cache[22] = $event => {openJob(activeJob.value); jobSheet.value = false;})
                    }))
                  : _createCommentVNode("", true),
                (activeJob.value?.status === 'pending')
                  ? (_openBlock(), _createBlock(_component_VListItem, {
                      key: 1,
                      title: "插队",
                      onClick: _cache[23] || (_cache[23] = $event => {cutIn(activeJob.value); jobSheet.value = false;})
                    }))
                  : _createCommentVNode("", true),
                (activeJob.value?.status === 'success')
                  ? (_openBlock(), _createBlock(_component_VListItem, {
                      key: 2,
                      title: "再次翻译并覆盖字幕",
                      onClick: _cache[24] || (_cache[24] = $event => (retryJob(activeJob.value, true)))
                    }))
                  : _createCommentVNode("", true),
                (activeJob.value?.status === 'failed' || activeJob.value?.status === 'cancelled' || activeJob.value?.status === 'skipped')
                  ? (_openBlock(), _createBlock(_component_VListItem, {
                      key: 3,
                      title: "重跑",
                      onClick: _cache[25] || (_cache[25] = $event => (retryJob(activeJob.value)))
                    }))
                  : _createCommentVNode("", true),
                (activeJob.value?.job_id)
                  ? (_openBlock(), _createBlock(_component_VListItem, {
                      key: 4,
                      title: "重新搜索",
                      onClick: _cache[26] || (_cache[26] = $event => {searchAgain(activeJob.value); jobSheet.value = false;})
                    }))
                  : _createCommentVNode("", true),
                _createVNode(_component_VListItem, {
                  title: "改成 P0",
                  onClick: _cache[27] || (_cache[27] = $event => {changePriority(activeJob.value, 'P0'); jobSheet.value = false;})
                }),
                _createVNode(_component_VListItem, {
                  title: "改成 P1",
                  onClick: _cache[28] || (_cache[28] = $event => {changePriority(activeJob.value, 'P1'); jobSheet.value = false;})
                }),
                _createVNode(_component_VListItem, {
                  title: "改成 P2",
                  onClick: _cache[29] || (_cache[29] = $event => {changePriority(activeJob.value, 'P2'); jobSheet.value = false;})
                }),
                _createVNode(_component_VListItem, {
                  title: "取消",
                  onClick: _cache[30] || (_cache[30] = $event => {cancelJob(activeJob.value); jobSheet.value = false;})
                })
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
      "onUpdate:modelValue": _cache[35] || (_cache[35] = $event => ((enqueueSheet).value = $event)),
      inset: "",
      rounded: "t-xl"
    }, {
      default: _withCtx(() => [
        _createVNode(_component_VCard, { class: "pa-4" }, {
          default: _withCtx(() => [
            _createVNode(_component_VCardTitle, null, {
              default: _withCtx(() => [...(_cache[68] || (_cache[68] = [
                _createTextVNode("提交处理", -1)
              ]))]),
              _: 1
            }),
            _createElementVNode("div", _hoisted_29, "将提交 " + _toDisplayString(enqueueForm.items?.length || 1) + " 个文件", 1),
            (enqueueForm.items?.some(item => item.already_processed))
              ? (_openBlock(), _createBlock(_component_VAlert, {
                  key: 0,
                  type: "warning",
                  variant: "tonal",
                  class: "mb-3"
                }, {
                  default: _withCtx(() => [...(_cache[69] || (_cache[69] = [
                    _createTextVNode(" 所选里有已经翻译过的文件。覆盖开关打开时，这次会直接替换已生成的字幕。 ", -1)
                  ]))]),
                  _: 1
                }))
              : _createCommentVNode("", true),
            (enqueueHasStrm.value)
              ? (_openBlock(), _createBlock(_component_VAlert, {
                  key: 1,
                  type: "warning",
                  variant: "tonal",
                  class: "mb-3"
                }, {
                  default: _withCtx(() => [...(_cache[70] || (_cache[70] = [
                    _createTextVNode(" 勾选里含 STRM：只搜外挂 / 翻译 / 导出，不做 ASR 和调轴。 ", -1)
                  ]))]),
                  _: 1
                }))
              : _createCommentVNode("", true),
            _createVNode(_component_VSelect, {
              modelValue: enqueueForm.priority,
              "onUpdate:modelValue": _cache[32] || (_cache[32] = $event => ((enqueueForm.priority) = $event)),
              label: "优先级",
              items: ['P0', 'P1', 'P2']
            }, null, 8, ["modelValue"]),
            _createVNode(_component_VSwitch, {
              modelValue: enqueueForm.overwrite,
              "onUpdate:modelValue": _cache[33] || (_cache[33] = $event => ((enqueueForm.overwrite) = $event)),
              label: "再次翻译时覆盖已有字幕",
              "hide-details": "",
              class: "mb-1"
            }, null, 8, ["modelValue"]),
            _cache[72] || (_cache[72] = _createElementVNode("div", { class: "text-caption text-medium-emphasis mb-2" }, "打开后这次直接替换目标字幕，不再受设置里「跳过已有文件」的影响。已有中字也不会拦住这次入队。", -1)),
            _createVNode(_component_VSwitch, {
              modelValue: enqueueForm.force,
              "onUpdate:modelValue": _cache[34] || (_cache[34] = $event => ((enqueueForm.force) = $event)),
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
              default: _withCtx(() => [...(_cache[71] || (_cache[71] = [
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
      modelValue: hideSheet.value,
      "onUpdate:modelValue": _cache[37] || (_cache[37] = $event => ((hideSheet).value = $event)),
      inset: "",
      rounded: "t-xl"
    }, {
      default: _withCtx(() => [
        _createVNode(_component_VCard, { class: "pa-4" }, {
          default: _withCtx(() => [
            _createVNode(_component_VCardTitle, null, {
              default: _withCtx(() => [...(_cache[73] || (_cache[73] = [
                _createTextVNode("从列表移除", -1)
              ]))]),
              _: 1
            }),
            _createElementVNode("div", _hoisted_30, _toDisplayString(hideForm.title ? `${hideForm.title} · ` : '') + _toDisplayString(hideForm.paths.length) + " 个文件。只从字幕工坊的媒体列表里隐藏，不会删除视频，也不会改 MoviePilot 的整理记录。 ", 1),
            _createVNode(_component_VSwitch, {
              modelValue: hideForm.deleteSubtitles,
              "onUpdate:modelValue": _cache[36] || (_cache[36] = $event => ((hideForm.deleteSubtitles) = $event)),
              label: "同时删除已生成的字幕文件",
              "hide-details": "",
              class: "mb-1"
            }, null, 8, ["modelValue"]),
            _cache[75] || (_cache[75] = _createElementVNode("div", { class: "text-caption text-medium-emphasis mb-3" }, "默认不删文件。勾选后只删除本插件按当前导出设置写出的字幕和备份，原片和其他外挂会留下。", -1)),
            _createVNode(_component_VBtn, {
              color: "primary",
              block: "",
              class: "ss-touch",
              loading: hiding.value,
              onClick: confirmHide
            }, {
              default: _withCtx(() => [...(_cache[74] || (_cache[74] = [
                _createTextVNode("确认移除", -1)
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
      "onUpdate:modelValue": _cache[40] || (_cache[40] = $event => ((cueSheet).value = $event)),
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
                  default: _withCtx(() => [...(_cache[76] || (_cache[76] = [
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
                _createVNode(_component_VTextField, {
                  modelValue: editingCue.value.start_ms,
                  "onUpdate:modelValue": _cache[38] || (_cache[38] = $event => ((editingCue.value.start_ms) = $event)),
                  modelModifiers: { number: true },
                  label: "开始（毫秒）",
                  type: "number"
                }, null, 8, ["modelValue"]),
                _createVNode(_component_VTextField, {
                  modelValue: editingCue.value.end_ms,
                  "onUpdate:modelValue": _cache[39] || (_cache[39] = $event => ((editingCue.value.end_ms) = $event)),
                  modelModifiers: { number: true },
                  label: "结束（毫秒）",
                  type: "number"
                }, null, 8, ["modelValue"]),
                _createVNode(_component_VBtn, {
                  color: "primary",
                  block: "",
                  class: "ss-touch",
                  onClick: saveCue
                }, {
                  default: _withCtx(() => [...(_cache[77] || (_cache[77] = [
                    _createTextVNode("保存到 CueGraph", -1)
                  ]))]),
                  _: 1
                })
              ]),
              _: 1
            }))
          : _createCommentVNode("", true)
      ]),
      _: 1
    }, 8, ["modelValue"]),
    _createVNode(_component_VDialog, {
      modelValue: modelPicker.value.open,
      "onUpdate:modelValue": _cache[41] || (_cache[41] = $event => ((modelPicker.value.open) = $event)),
      "max-width": "420"
    }, {
      default: _withCtx(() => [
        _createVNode(_component_VCard, null, {
          default: _withCtx(() => [
            _createVNode(_component_VCardTitle, null, {
              default: _withCtx(() => [...(_cache[78] || (_cache[78] = [
                _createTextVNode("选择模型", -1)
              ]))]),
              _: 1
            }),
            _createVNode(_component_VList, null, {
              default: _withCtx(() => [
                (_openBlock(true), _createElementBlock(_Fragment, null, _renderList(modelPicker.value.items, (item) => {
                  return (_openBlock(), _createBlock(_component_VListItem, {
                    key: item,
                    title: item,
                    onClick: $event => (applyModel(item))
                  }, null, 8, ["title", "onClick"]))
                }), 128))
              ]),
              _: 1
            })
          ]),
          _: 1
        })
      ]),
      _: 1
    }, 8, ["modelValue"])
  ]))
}
}

};

export { _sfc_main as default };
