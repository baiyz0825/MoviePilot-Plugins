<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import '../styles/theme.css'
import { createStudioApi } from '../api/studioApi'
import { useHostInjects } from '../composables/useHostInjects'
import { useMobileViewport } from '../composables/useMobileViewport'
import { langLabel } from '../composables/fields'
import PreviewPlayer from './shared/PreviewPlayer.vue'
import SettingsForm from './shared/SettingsForm.vue'

const props = defineProps({
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  sourcePluginId: { type: String, default: 'SubtitleStudio' },
  navKey: { type: String, default: 'main' },
  hideTitle: { type: Boolean, default: false },
})

const emit = defineEmits(['action'])
const isMobile = useMobileViewport()
const { toast, confirm } = useHostInjects()
const pluginBase = computed(() => `plugin/${props.pluginId || props.sourcePluginId || 'SubtitleStudio'}`)
const pluginApi = computed(() => createStudioApi(props.api, pluginBase))

const nav = ref('media')
const loading = ref(false)
const dirty = ref(false)
const mediaQuery = ref('')
const mediaType = ref('')
const mediaGroups = ref([])
const failedPosterUrls = ref(new Set())
const posterObjectUrls = ref({})
const loadingPosterUrls = ref(new Set())
let posterObserver
const mediaCounts = ref({ files: 0, groups: 0 })
const stockScan = ref({ status: 'idle', scanned: 0, total: 0, queued: 0, skipped: 0, failed: 0 })
const stockScanStarting = ref(false)
let stockScanTimer = null
const mediaDetail = ref(null)
const selectedPaths = ref({})
const submitting = ref(false)
const jobsQuery = ref('')
const jobStatus = ref('')
const jobs = ref([])
const jobSheet = ref(false)
const activeJob = ref(null)
const graph = ref({ cues: [], notes: [], briefing: [] })
const currentMs = ref(0)
const previewTracks = ref(['dialogue', 'notes'])
const previewObjectUrl = ref('')
const cueSheet = ref(false)
const editingCue = ref(null)
const config = ref({})
const fields = ref([])
const modelPicker = ref({ open: false, endpointId: '', items: [] })
const enqueueSheet = ref(false)
const enqueueForm = reactive({ strategy: 'search_then_translate', priority: 'P0', force: false, overwrite: false, items: [] })
const hideSheet = ref(false)
const hiding = ref(false)
const hideForm = reactive({ paths: [], title: '', deleteSubtitles: false })
const selectedFiles = computed(() => (mediaDetail.value?.files || []).filter(item => selectedPaths.value[item.path]))
const enqueueHasStrm = computed(() => (enqueueForm.items || []).some(item => item?.is_strm || String(item?.path || '').toLowerCase().endsWith('.strm')))

const STATUS_LABELS = { pending: '等待', running: '运行中', success: '成功', failed: '失败', skipped: '跳过', cancelled: '已取消' }
const TRIGGER_LABELS = { event: '事件', watch: '监控', strm: 'STRM', manual: '手动', stock_scan: '存量扫描', workflow: '工作流', command: '命令' }
const TRACK_LABELS = { dialogue: '对白', notes: '顶注', stacked: '叠行', sdh: 'SDH' }
const statusFilters = [
  { value: '', label: '全部' },
  { value: 'pending', label: '等待' },
  { value: 'running', label: '运行中' },
  { value: 'failed', label: '失败' },
  { value: 'success', label: '成功' },
  { value: 'skipped', label: '跳过' },
  { value: 'cancelled', label: '已取消' },
]
const expandedJob = ref(null)
function statusLabel(s) { return STATUS_LABELS[s] || s }
function triggerLabel(t) { return TRIGGER_LABELS[t] || t }
function trackLabel(t) { return TRACK_LABELS[t] || t }
function markPosterFailed(url) {
  failedPosterUrls.value = new Set(failedPosterUrls.value).add(url)
}
function isProxyPoster(url) {
  if (typeof url !== 'string' || !/^https?:\/\//i.test(url)) return false
  try {
    const host = new URL(url).hostname.toLowerCase()
    return host === 'doubanio.com' || host.endsWith('.doubanio.com') || host === 'tmdb.org' || host.endsWith('.tmdb.org')
  } catch {
    return false
  }
}
async function loadPoster(url) {
  if (!url || posterObjectUrls.value[url] || loadingPosterUrls.value.has(url) || failedPosterUrls.value.has(url)) return
  loadingPosterUrls.value = new Set(loadingPosterUrls.value).add(url)
  try {
    const response = await props.api.get('system/img/0', {
      params: { imgurl: url },
      responseType: 'blob',
      feedback: 'silent',
    })
    const blob = response instanceof Blob ? response : response?.data
    if (!(blob instanceof Blob) || !blob.size) throw new Error('图片代理未返回图片')
    posterObjectUrls.value = { ...posterObjectUrls.value, [url]: URL.createObjectURL(blob) }
  } catch {
    markPosterFailed(url)
  } finally {
    const next = new Set(loadingPosterUrls.value)
    next.delete(url)
    loadingPosterUrls.value = next
  }
}
const vPoster = {
  mounted(el, binding) {
    const url = binding.value
    if (!isProxyPoster(url)) return
    posterObserver ||= new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return
        const imageUrl = entry.target.dataset.posterUrl
        posterObserver.unobserve(entry.target)
        loadPoster(imageUrl)
      })
    }, { rootMargin: '120px' })
    el.dataset.posterUrl = url
    posterObserver.observe(el)
  },
  unmounted(el) {
    posterObserver?.unobserve(el)
  },
}
function toggleJobDetail(job) { expandedJob.value = expandedJob.value === job.job_id ? null : job.job_id }
function formatMs(ms) {
  const total = Math.max(0, Math.floor(Number(ms) || 0) / 1000)
  const m = Math.floor(total / 60)
  const s = Math.floor(total % 60)
  const frac = Math.floor((Number(ms) || 0) % 1000)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(frac).padStart(3, '0')}`
}

const tabs = [
  { value: 'media', title: '媒体', icon: 'mdi-filmstrip' },
  { value: 'jobs', title: '队列', icon: 'mdi-playlist-play' },
  { value: 'desk', title: '工作台', icon: 'mdi-subtitles-outline' },
  { value: 'settings', title: '设置', icon: 'mdi-cog-outline' },
]

const duration = computed(() => Math.max(graph.value.duration_ms || 1, ...(graph.value.cues || []).map(item => item.end_ms || 0), 1))
const subtitleFilter = ref('')
const selectedGroupIds = ref({})

function fileHasSubtitle(file) {
  return !!(file?.sidecars?.length || file?.already_processed)
}

const visibleGroups = computed(() => (mediaGroups.value || []).map(group => {
  const files = (group.files || []).filter(file => {
    if (subtitleFilter.value === 'has') return fileHasSubtitle(file)
    if (subtitleFilter.value === 'missing') return !fileHasSubtitle(file)
    return true
  })
  return files.length ? { ...group, files, file_count: files.length } : null
}).filter(Boolean))

const selectedGroupFiles = computed(() => visibleGroups.value
  .filter(group => selectedGroupIds.value[group.id])
  .flatMap(group => group.files || []))

watch(nav, async (value, prev) => {
  if (prev === 'settings' && value !== 'settings' && dirty.value) {
    if (!(await confirm('设置尚未保存，确定离开？'))) {
      nav.value = 'settings'
    }
  }
})

async function reload() {
  loading.value = true
  try {
    const [cfg, fieldData] = await Promise.all([
      pluginApi.value.config().catch(() => ({})),
      pluginApi.value.fields().catch(() => ({ fields: [] })),
    ])
    config.value = { ...cfg }
    fields.value = fieldData.fields || []
    previewTracks.value = cfg.preview_tracks_default || ['dialogue', 'notes']
    await Promise.all([loadMedia(), loadJobs(), loadStockScanStatus()])
  } finally {
    loading.value = false
  }
}

async function loadMedia() {
  try {
    const data = await pluginApi.value.media(mediaQuery.value, mediaType.value)
    mediaGroups.value = data?.groups || []
    mediaCounts.value = data?.counts || { files: 0, groups: 0 }
  } catch (error) {
    toast.error?.(error?.message || '加载媒体失败')
  }
}

async function refreshLibrary() {
  loading.value = true
  try {
    const data = await pluginApi.value.refreshMedia()
    mediaGroups.value = data?.groups || []
    mediaCounts.value = data?.counts || { files: 0, groups: 0 }
    toast.success?.(`已拉取 ${mediaCounts.value.files || 0} 个媒体文件`)
  } catch (error) {
    toast.error?.(error?.message || '拉取媒体库失败')
  } finally {
    loading.value = false
  }
}

function syncStockScanPolling() {
  if (stockScan.value?.status === 'running') {
    if (!stockScanTimer) stockScanTimer = setInterval(() => { loadStockScanStatus() }, 1600)
  } else if (stockScanTimer) {
    clearInterval(stockScanTimer)
    stockScanTimer = null
  }
}

async function loadStockScanStatus() {
  const wasRunning = stockScan.value?.status === 'running'
  try {
    stockScan.value = await pluginApi.value.stockScanStatus() || { status: 'idle' }
    syncStockScanPolling()
    if (wasRunning && stockScan.value?.status !== 'running') {
      await Promise.all([loadMedia(), loadJobs()])
      if (stockScan.value?.status === 'completed') {
        toast.success?.(`存量扫描完成：入队 ${stockScan.value.queued || 0} 条，跳过 ${stockScan.value.skipped || 0} 条`)
      }
    }
  } catch {
    // A status poll is silent; the next interval can recover from transient API failures.
  }
}

async function startStockScan() {
  if (stockScanStarting.value || stockScan.value?.status === 'running') return
  const targets = (config.value.target_languages || ['zh-Hans']).map(langLabel).join('、')
  if (!(await confirm(`将扫描已入库记录和配置的媒体库目录，检查外挂字幕及可识别的内嵌字幕。只有全部目标语言（${targets}）都已存在的媒体会跳过；其余会以 P1 加入队列。扫描可能持续一段时间，是否开始？`))) return
  stockScanStarting.value = true
  try {
    stockScan.value = await pluginApi.value.startStockScan() || { status: 'idle' }
    syncStockScanPolling()
    if (stockScan.value?.started) toast.success?.('存量字幕扫描已开始，可在此查看进度')
    else if (stockScan.value?.status === 'running') toast.success?.('存量字幕扫描正在运行')
  } catch (error) {
    toast.error?.(error?.message || '启动存量扫描失败')
  } finally {
    stockScanStarting.value = false
  }
}

function openGroup(group) {
  mediaDetail.value = group
  selectedPaths.value = Object.fromEntries((group.files || []).map(item => [item.path, true]))
}

function toggleGroup(id, value) {
  selectedGroupIds.value = { ...selectedGroupIds.value, [id]: value }
}

function toggleVisibleGroups(value) {
  selectedGroupIds.value = Object.fromEntries(visibleGroups.value.map(group => [group.id, value]))
}

async function submitGroups() {
  const files = selectedGroupFiles.value
  if (!files.length) {
    toast.error?.('先勾选要提交的影片')
    return
  }
  const completedCount = files.filter(item => item.already_processed).length
  if (completedCount && !(await confirm(`${completedCount} 个文件已有成功处理记录。再次提交会重新执行并消耗模型资源，仍要继续吗？`))) return
  prepareEnqueue(files)
}

function toggleFile(path, value) {
  selectedPaths.value = { ...selectedPaths.value, [path]: value }
}

function toggleAll(value) {
  selectedPaths.value = Object.fromEntries((mediaDetail.value?.files || []).map(item => [item.path, value]))
}

function fileLabel(item) {
  if (item.type === 'tv' && item.season && item.episode) {
    const season = String(item.season).padStart(2, '0')
    const episode = String(item.episode).padStart(2, '0')
    return `S${season}E${episode} · ${item.filename || item.path}`
  }
  return item.filename || item.path
}

let pollTimer = null
async function loadJobs() {
  try {
    const data = await pluginApi.value.jobs(jobsQuery.value, jobStatus.value)
    jobs.value = data?.items || []
  } catch (error) {
    toast.error?.(error?.message || '加载队列失败')
  }
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
  if (jobs.value.some(j => j.status === 'running' || j.status === 'pending')) {
    pollTimer = setInterval(() => { loadJobs() }, 1500)
  }
}

async function saveConfig() {
  try {
    const saved = await pluginApi.value.saveConfig(config.value)
    config.value = { ...(saved || config.value) }
    dirty.value = false
    toast.success?.('已保存')
  } catch (error) {
    toast.error?.(error?.message || '保存失败')
  }
}

function revokePreview() {
  if (previewObjectUrl.value) {
    URL.revokeObjectURL(previewObjectUrl.value)
    previewObjectUrl.value = ''
  }
}

async function loadPreview(job) {
  revokePreview()
  if (!job?.job_id || job.is_strm) return
  try {
    const blob = await pluginApi.value.fetchPreviewVideo(job.job_id)
    if (blob) previewObjectUrl.value = URL.createObjectURL(blob)
  } catch {
    previewObjectUrl.value = ''
  }
}

function prepareEnqueue(items) {
  enqueueForm.items = items
  enqueueForm.force = false
  enqueueForm.overwrite = items.some(item => item.already_processed)
  enqueueSheet.value = true
}

async function enqueue(item) {
  const items = item?.path ? [item] : selectedFiles.value
  enqueueForm.path = item?.path || items[0]?.path
  enqueueForm.title = item?.title || mediaDetail.value?.title
  enqueueForm.media_source = item?.media_source
  enqueueForm.media_id = item?.media_id
  enqueueForm.tmdbid = item?.tmdbid
  enqueueForm.doubanid = item?.doubanid
  prepareEnqueue(items)
}

async function submitSelected() {
  if (!selectedFiles.value.length) {
    toast.error?.('先勾选要处理的文件')
    return
  }
  const completedCount = selectedFiles.value.filter(item => item.already_processed).length
  if (completedCount && !(await confirm(`${completedCount} 个文件已有成功处理记录。再次提交会重新执行并消耗模型资源，仍要继续吗？`))) return
  prepareEnqueue(selectedFiles.value)
}

async function confirmEnqueue() {
  submitting.value = true
  try {
    const items = enqueueForm.items?.length ? enqueueForm.items : [enqueueForm]
    const data = await pluginApi.value.createJobs({
      items,
      priority: enqueueForm.priority,
      force: enqueueForm.force,
      overwrite: enqueueForm.overwrite,
    })
    const created = data?.count || data?.items?.length || 0
    const skipped = data?.skipped?.length || 0
    if (skipped) {
      const reasons = (data.skipped || []).slice(0, 3).map(item => item.reason || '跳过').join('；')
      toast.success?.(`已入队 ${created}，跳过 ${skipped}${reasons ? `：${reasons}` : ''}`)
    } else {
      toast.success?.(`已入队 ${created} 条`)
    }
    enqueueSheet.value = false
    nav.value = 'jobs'
    await loadJobs()
  } catch (error) {
    toast.error?.(error?.message || '入队失败')
  } finally {
    submitting.value = false
  }
}

async function openJob(job) {
  try {
    activeJob.value = job
    graph.value = (await pluginApi.value.cues(job.job_id)) || { cues: [], notes: [], briefing: [] }
    await loadPreview(job)
    nav.value = 'desk'
  } catch (error) {
    toast.error?.(error?.message || '打开工作台失败')
  }
}

async function cutIn(job) {
  if (!(await confirm('把这个任务插到队首？'))) return
  try {
    await pluginApi.value.cutIn(job.job_id)
    await loadJobs()
  } catch (error) {
    toast.error?.(error?.message || '插队失败')
  }
}

async function changePriority(job, priority) {
  if (!(await confirm(`改成 ${priority}？`))) return
  try {
    await pluginApi.value.setPriority(job.job_id, priority)
    await loadJobs()
  } catch (error) {
    toast.error?.(error?.message || '改优先级失败')
  }
}

async function cancelJob(job) {
  if (!(await confirm('取消这个任务？'))) return
  try {
    await pluginApi.value.cancel(job.job_id)
    toast.success?.('已取消')
    await loadJobs()
  } catch (error) {
    toast.error?.(error?.message || '取消失败')
  }
}

function askHide(paths, title) {
  const list = [...new Set((paths || []).filter(Boolean))]
  if (!list.length) {
    toast.error?.('先勾选要从列表移除的文件')
    return
  }
  hideForm.paths = list
  hideForm.title = title || ''
  hideForm.deleteSubtitles = false
  hideSheet.value = true
}

async function confirmHide() {
  hiding.value = true
  try {
    const data = await pluginApi.value.hideMedia({
      paths: hideForm.paths,
      delete_subtitles: hideForm.deleteSubtitles,
    })
    const removed = data?.deleted_files?.length || 0
    toast.success?.(hideForm.deleteSubtitles ? `已从列表移除，并删除 ${removed} 个字幕文件` : '已从列表移除')
    hideSheet.value = false
    if (mediaDetail.value) {
      const gone = new Set(hideForm.paths)
      const files = (mediaDetail.value.files || []).filter(item => !gone.has(item.path))
      mediaDetail.value = files.length ? { ...mediaDetail.value, files, file_count: files.length } : null
    }
    await loadMedia()
  } catch (error) {
    toast.error?.(error?.message || '移除失败')
  } finally {
    hiding.value = false
  }
}

async function retryJob(job, overwrite = false) {
  try {
    await pluginApi.value.retry(job.job_id, { overwrite: !!overwrite })
    toast.success?.(overwrite ? '已重新入队，将覆盖已有字幕' : '已重新入队')
    jobSheet.value = false
    nav.value = 'jobs'
    await loadJobs()
  } catch (error) {
    toast.error?.(error?.message || '重跑失败')
  }
}

async function searchAgain(job) {
  try {
    const data = await pluginApi.value.searchJob(job.job_id)
    toast.success?.(`搜索到 ${(data?.items || []).length} 条结果`)
  } catch (error) {
    toast.error?.(error?.message || '搜索失败')
  }
}

async function exportCurrent() {
  if (!activeJob.value?.job_id) return
  if (!(await confirm('按当前 CueGraph 写盘？会遵循设置里的覆盖策略。'))) return
  try {
    const data = await pluginApi.value.exportJob(activeJob.value.job_id)
    const written = (data?.items || []).filter(item => item.written).length
    const skipped = (data?.items || []).filter(item => !item.written).length
    toast.success?.(`已写 ${written} 个文件${skipped ? `，跳过 ${skipped}` : ''}`)
  } catch (error) {
    toast.error?.(error?.message || '写盘失败')
  }
}

async function downloadAssPreview() {
  if (!activeJob.value?.job_id) return
  try {
    const text = await pluginApi.value.fetchPreviewAss(activeJob.value.job_id)
    const blob = new Blob([text || ''], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${activeJob.value.title || 'preview'}.ass`
    a.click()
    URL.revokeObjectURL(url)
  } catch (error) {
    toast.error?.(error?.message || '预览 ASS 失败')
  }
}

function openCue(cue) {
  editingCue.value = {
    ...cue,
    texts: { ...(cue.texts || {}) },
    start_ms: cue.start_ms,
    end_ms: cue.end_ms,
  }
  cueSheet.value = true
}

async function saveCue() {
  try {
    await pluginApi.value.saveCue(activeJob.value.job_id, editingCue.value.cue_id, {
      texts: editingCue.value.texts,
      start_ms: Number(editingCue.value.start_ms),
      end_ms: Number(editingCue.value.end_ms),
    })
    cueSheet.value = false
    graph.value = await pluginApi.value.cues(activeJob.value.job_id)
    toast.success?.('已改句')
  } catch (error) {
    toast.error?.(error?.message || '保存句子失败')
  }
}

function cueLeft(cue) {
  return `${(cue.start_ms / duration.value) * 100}%`
}

function cueWidth(cue) {
  return `${Math.max(4, ((cue.end_ms - cue.start_ms) / duration.value) * 100)}%`
}

function seekTimeline(event) {
  const box = event.currentTarget.getBoundingClientRect()
  currentMs.value = Math.floor(((event.clientX - box.left) / box.width) * duration.value)
}

function moreJob(job) {
  activeJob.value = job
  jobSheet.value = true
}

async function onListModels(endpoint) {
  try {
    const data = await pluginApi.value.listModels(endpoint)
    const items = data?.items || []
    if (!items.length) {
      toast.error?.('没有拉到模型')
      return
    }
    modelPicker.value = { open: true, endpointId: endpoint.endpoint_id, items }
  } catch (error) {
    toast.error?.(error?.message || '拉模型失败')
  }
}

function applyModel(model) {
  const rows = (config.value.openai_endpoints || []).map(item => (
    item.endpoint_id === modelPicker.value.endpointId ? { ...item, model } : item
  ))
  config.value = { ...config.value, openai_endpoints: rows }
  dirty.value = true
  modelPicker.value.open = false
  toast.success?.(`已选模型 ${model}`)
}

onMounted(reload)
onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer)
  if (stockScanTimer) clearInterval(stockScanTimer)
  revokePreview()
  posterObserver?.disconnect()
  Object.values(posterObjectUrls.value).forEach(url => URL.revokeObjectURL(url))
})
defineExpose({ reload, loadStatus: reload })
</script>

<template>
  <div class="plugin-root">
    <div class="ss-toolbar pa-3">
      <div v-if="!hideTitle" class="text-h6 mb-2">字幕工坊</div>
      <VBtnToggle v-model="nav" mandatory density="comfortable" class="w-100">
        <VBtn v-for="tab in tabs" :key="tab.value" :value="tab.value" class="ss-touch flex-grow-1">
          <VIcon start :icon="tab.icon" />
          {{ tab.title }}
        </VBtn>
      </VBtnToggle>
    </div>

    <div class="pa-3">
      <template v-if="nav === 'media'">
        <div v-if="mediaDetail" class="mb-3">
          <div class="d-flex align-center mb-3">
            <VBtn icon="mdi-arrow-left" class="ss-touch" aria-label="返回" @click="mediaDetail = null" />
            <strong class="ms-2">{{ mediaDetail.title }}{{ mediaDetail.year ? ` (${mediaDetail.year})` : '' }}</strong>
          </div>
          <div class="text-medium-emphasis mb-2">{{ mediaDetail.library_name || 'MoviePilot 整理记录' }} · {{ mediaDetail.file_count || mediaDetail.files?.length || 0 }} 个文件</div>
          <div class="d-flex ga-2 mb-3">
            <VBtn size="small" variant="text" class="ss-touch" @click="toggleAll(true)">全选</VBtn>
            <VBtn size="small" variant="text" class="ss-touch" @click="toggleAll(false)">清空</VBtn>
            <VBtn size="small" variant="text" class="ss-touch" :disabled="!selectedFiles.length" @click="askHide(selectedFiles.map(item => item.path), mediaDetail.title)">移除所选</VBtn>
          </div>
          <VList>
            <VListItem v-for="item in mediaDetail.files || []" :key="item.id || item.path">
              <template #prepend>
                <VCheckbox
                  :model-value="!!selectedPaths[item.path]"
                  hide-details
                  @update:model-value="toggleFile(item.path, $event)"
                />
              </template>
              <VListItemTitle>{{ fileLabel(item) }}</VListItemTitle>
              <VListItemSubtitle>{{ item.sidecars?.length || 0 }} 条外挂{{ item.is_strm ? ' · STRM' : '' }}{{ item.already_processed ? ' · 已成功处理' : '' }}</VListItemSubtitle>
              <template #append>
                <VBtn icon="mdi-close" class="ss-touch" aria-label="从列表移除" @click.stop="askHide([item.path], fileLabel(item))" />
              </template>
            </VListItem>
          </VList>
          <VBtn color="primary" block class="ss-touch mt-3" :disabled="!selectedFiles.length" @click="submitSelected">
            提交处理（{{ selectedFiles.length }}）
          </VBtn>
        </div>
        <template v-else>
          <div class="d-flex ga-2 mb-3">
            <VTextField v-model="mediaQuery" label="搜索标题或文件名" prepend-inner-icon="mdi-magnify" hide-details class="flex-grow-1" @keyup.enter="loadMedia" />
            <VBtn class="ss-touch" :loading="loading" @click="loadMedia">搜索</VBtn>
          </div>
          <div class="d-flex ga-2 mb-3" style="overflow-x:auto">
            <VChip :color="!mediaType ? 'primary' : undefined" @click="mediaType = ''; loadMedia()">全部</VChip>
            <VChip :color="mediaType === 'movie' ? 'primary' : undefined" @click="mediaType = 'movie'; loadMedia()">电影</VChip>
            <VChip :color="mediaType === 'tv' ? 'primary' : undefined" @click="mediaType = 'tv'; loadMedia()">剧集</VChip>
            <VBtn size="small" class="ss-touch" :loading="loading" @click="refreshLibrary">拉取媒体库</VBtn>
          </div>
          <div class="d-flex ga-2 mb-3" style="overflow-x:auto">
            <VChip :color="!subtitleFilter ? 'primary' : undefined" @click="subtitleFilter = ''">字幕不限</VChip>
            <VChip :color="subtitleFilter === 'has' ? 'primary' : undefined" @click="subtitleFilter = 'has'">已有字幕</VChip>
            <VChip :color="subtitleFilter === 'missing' ? 'primary' : undefined" @click="subtitleFilter = 'missing'">没有字幕</VChip>
          </div>
          <div class="d-flex align-center ga-2 flex-wrap mb-3">
            <VBtn
              size="small"
              color="primary"
              variant="tonal"
              class="ss-touch"
              :disabled="stockScan.status === 'running'"
              :loading="stockScanStarting"
              @click="startStockScan"
            >
              <VIcon start icon="mdi-text-box-search-outline" />
              {{ stockScan.status === 'running' ? '正在扫描存量媒体' : '一次性扫描缺少目标字幕' }}
            </VBtn>
            <span class="text-caption text-medium-emphasis">命中全部目标语言的内嵌或外挂字幕才跳过</span>
          </div>
          <VAlert
            v-if="stockScan.status !== 'idle'"
            :type="stockScan.status === 'failed' ? 'error' : stockScan.status === 'completed' ? 'success' : stockScan.status === 'running' ? 'info' : 'warning'"
            variant="tonal"
            class="mb-3"
          >
            <div>
              存量扫描 {{ stockScan.status === 'running' ? '进行中' : stockScan.status === 'completed' ? '完成' : stockScan.status === 'failed' ? '失败' : '已停止' }}
              · 已检查 {{ stockScan.scanned || 0 }}/{{ stockScan.total || 0 }}
              · 已入队 {{ stockScan.queued || 0 }}
              · 已跳过 {{ stockScan.skipped || 0 }}
              <template v-if="stockScan.failed">· 检查失败 {{ stockScan.failed }}</template>
            </div>
            <div v-if="stockScan.current_title" class="text-caption mt-1">当前：{{ stockScan.current_title }}</div>
            <div v-if="stockScan.error" class="text-caption mt-1">{{ stockScan.error }}</div>
          </VAlert>
          <div class="d-flex ga-2 mb-3">
            <VBtn size="small" variant="text" class="ss-touch" @click="toggleVisibleGroups(true)">全选</VBtn>
            <VBtn size="small" variant="text" class="ss-touch" @click="toggleVisibleGroups(false)">清空</VBtn>
            <VBtn size="small" color="primary" class="ss-touch" :disabled="!selectedGroupFiles.length" @click="submitGroups">提交所选（{{ selectedGroupFiles.length }}）</VBtn>
          </div>
          <div class="text-medium-emphasis mb-3">整理记录 {{ visibleGroups.length }} 部 · {{ visibleGroups.reduce((sum, group) => sum + (group.file_count || 0), 0) }} 个文件</div>
          <VAlert v-if="!visibleGroups.length" type="info" variant="tonal" class="mb-3">
            {{ subtitleFilter ? '这个筛选下没有影片。' : '没有本地媒体。点「拉取媒体库」读取 MoviePilot 整理记录；先在 MoviePilot 里整理入库后才会出现。这里不是 Emby/Jellyfin 在线目录。' }}
          </VAlert>
          <VList>
            <VListItem
              v-for="item in visibleGroups"
              :key="item.id"
              class="ss-card mb-2"
              @click="openGroup(item)"
            >
              <template #prepend>
                <VCheckbox
                  :model-value="!!selectedGroupIds[item.id]"
                  hide-details
                  @click.stop
                  @update:model-value="toggleGroup(item.id, $event)"
                />
                <div class="ss-media-poster" aria-hidden="true">
                  <img
                    v-if="item.poster && !failedPosterUrls.has(item.poster)"
                    v-poster="item.poster"
                    :src="isProxyPoster(item.poster) ? posterObjectUrls[item.poster] : item.poster"
                    alt=""
                    loading="lazy"
                    decoding="async"
                    @error="markPosterFailed(item.poster)"
                  />
                  <VIcon v-else icon="mdi-movie-open-outline" />
                </div>
              </template>
              <VListItemTitle>{{ item.title }}{{ item.year ? ` (${item.year})` : '' }}</VListItemTitle>
              <VListItemSubtitle>
                {{ item.type === 'tv' ? '剧集' : '电影' }} · {{ item.file_count || item.files?.length || 0 }} 个文件 · {{ item.library_name }}
                <template v-if="item.processed_count"> · 已处理 {{ item.processed_count }}/{{ item.file_count || item.files?.length || 0 }}</template>
              </VListItemSubtitle>
              <template #append>
                <VBtn icon="mdi-close" class="ss-touch" aria-label="从列表移除" @click.stop="askHide((item.files || []).map(file => file.path), item.title)" />
                <VIcon icon="mdi-chevron-right" />
              </template>
            </VListItem>
          </VList>
        </template>
      </template>

      <template v-else-if="nav === 'jobs'">
        <div class="d-flex ga-2 mb-3">
          <VTextField v-model="jobsQuery" label="搜索标题" prepend-inner-icon="mdi-magnify" hide-details class="flex-grow-1" @keyup.enter="loadJobs" />
          <VBtn class="ss-touch" @click="loadJobs">搜索</VBtn>
        </div>
        <div class="d-flex ga-2 mb-3" style="overflow-x:auto">
          <VChip v-for="item in statusFilters" :key="item.value" :color="jobStatus === item.value ? 'primary' : undefined" @click="jobStatus = item.value; loadJobs()">
            {{ item.label }}
          </VChip>
        </div>
        <VAlert v-if="!jobs.length" type="info" variant="tonal" class="mb-3">队列是空的。从「媒体」勾选文件提交处理。</VAlert>
        <VList>
          <VListItem v-for="job in jobs" :key="job.job_id" class="ss-card mb-2" @click="toggleJobDetail(job)">
            <VListItemTitle>{{ job.title }}</VListItemTitle>
            <VListItemSubtitle>{{ job.priority }} · {{ statusLabel(job.status) }} · {{ triggerLabel(job.trigger) }}{{ job.is_strm ? ' · STRM' : '' }}</VListItemSubtitle>
            <div v-if="job.status === 'running' || job.progress > 0" class="mt-1 w-100">
              <VProgressLinear :model-value="job.progress" height="6" rounded color="primary" />
              <div class="text-caption text-medium-emphasis">{{ job.stage_detail }} · {{ job.progress }}%</div>
            </div>
            <div v-if="expandedJob === job.job_id" class="mt-2 pa-2 rounded bg-grey-lighten-4 text-caption">
              <div>阶段：{{ job.stage_detail || '—' }}</div>
              <div>进度：{{ job.progress }}%</div>
              <div>路径：{{ job.path }}</div>
              <div v-if="job.error" class="text-error">错误：{{ job.error }}</div>
            </div>
            <template #append>
              <VBtn v-if="!isMobile && job.status === 'pending'" size="small" class="ss-touch" @click.stop="cutIn(job)">插队</VBtn>
              <VBtn v-if="!isMobile && job.status === 'success'" size="small" class="ss-touch" @click.stop="retryJob(job, true)">覆盖重译</VBtn>
              <VBtn v-if="!isMobile && (job.status === 'failed' || job.status === 'cancelled')" size="small" class="ss-touch" @click.stop="retryJob(job)">重跑</VBtn>
              <VBtn v-if="!isMobile" size="small" variant="text" class="ss-touch" @click.stop="openJob(job)">工作台</VBtn>
              <VBtn icon="mdi-dots-horizontal" class="ss-touch" aria-label="更多" @click.stop="moreJob(job)" />
            </template>
          </VListItem>
        </VList>
      </template>

      <template v-else-if="nav === 'desk'">
        <div v-if="!activeJob" class="text-medium-emphasis">从队列打开一个任务。</div>
        <template v-else>
          <div class="d-flex align-center justify-space-between mb-2 flex-wrap ga-2">
            <div>{{ activeJob.title }}</div>
            <div class="d-flex ga-2 flex-wrap">
              <VBtn size="small" color="primary" class="ss-touch" @click="exportCurrent">按当前图写盘</VBtn>
              <VBtn size="small" variant="tonal" class="ss-touch" @click="searchAgain(activeJob)">重新搜索</VBtn>
              <VBtn size="small" variant="text" class="ss-touch" @click="downloadAssPreview">预览 ASS</VBtn>
            </div>
          </div>
          <PreviewPlayer
            :graph="graph"
            :current-ms="currentMs"
            :langs="config.target_languages || ['zh-Hans']"
            :tracks="previewTracks"
            :stack="config.lang_stack || 'main_bottom'"
            :style-config="config.ass_style || {}"
            :video-url="previewObjectUrl"
            :enabled="config.preview_enabled !== false"
            @time="currentMs = $event"
          />
          <div class="d-flex ga-2 my-3">
            <VChip v-for="track in ['dialogue', 'notes', 'stacked', 'sdh']" :key="track" :color="previewTracks.includes(track) ? 'primary' : undefined" @click="previewTracks.includes(track) ? previewTracks = previewTracks.filter(item => item !== track) : previewTracks = [...previewTracks, track]">
              {{ trackLabel(track) }}
            </VChip>
          </div>
          <div class="text-caption mb-1">按时间</div>
          <div class="ss-timeline mb-4" @click="seekTimeline">
            <button
              v-for="cue in graph.cues || []"
              :key="cue.cue_id"
              type="button"
              class="ss-cue"
              :aria-label="Object.values(cue.texts || {})[0] || '字幕句'"
              :style="{ left: cueLeft(cue), width: cueWidth(cue) }"
              @click.stop="openCue(cue); currentMs = cue.start_ms"
            >
              {{ Object.values(cue.texts || {})[0] }}
            </button>
          </div>
          <VAlert v-if="!(graph.cues || []).length" type="info" variant="tonal" class="mb-3">这个任务还没有 CueGraph。等任务跑完或先搜索/识别。</VAlert>
          <VList>
            <VListItem v-for="cue in graph.cues || []" :key="cue.cue_id" @click="openCue(cue)">
              <VListItemTitle>{{ Object.values(cue.texts || {})[0] }}</VListItemTitle>
              <VListItemSubtitle>{{ formatMs(cue.start_ms) }} – {{ formatMs(cue.end_ms) }}</VListItemSubtitle>
            </VListItem>
          </VList>
        </template>
      </template>

      <template v-else>
        <SettingsForm
          v-model="config"
          :fields="fields"
          :mobile="isMobile"
          @dirty="dirty = true"
          @test-endpoint="pluginApi.testEndpoint($event)"
          @list-models="onListModels"
        />
        <div v-if="dirty" class="ss-savebar pa-3 d-flex ga-2">
          <VBtn color="primary" class="ss-touch" @click="saveConfig">保存</VBtn>
          <VBtn variant="text" class="ss-touch" @click="reload()">恢复</VBtn>
        </div>
      </template>
    </div>

    <VBottomSheet v-model="jobSheet" inset rounded="t-xl">
      <VCard>
        <VCardTitle>{{ activeJob?.title }}</VCardTitle>
        <VList>
          <VListItem v-if="activeJob?.job_id" title="打开工作台" @click="openJob(activeJob); jobSheet = false" />
          <VListItem v-if="activeJob?.status === 'pending'" title="插队" @click="cutIn(activeJob); jobSheet = false" />
          <VListItem v-if="activeJob?.status === 'success'" title="再次翻译并覆盖字幕" @click="retryJob(activeJob, true)" />
          <VListItem v-if="activeJob?.status === 'failed' || activeJob?.status === 'cancelled' || activeJob?.status === 'skipped'" title="重跑" @click="retryJob(activeJob)" />
          <VListItem v-if="activeJob?.job_id" title="重新搜索" @click="searchAgain(activeJob); jobSheet = false" />
          <VListItem title="改成 P0" @click="changePriority(activeJob, 'P0'); jobSheet = false" />
          <VListItem title="改成 P1" @click="changePriority(activeJob, 'P1'); jobSheet = false" />
          <VListItem title="改成 P2" @click="changePriority(activeJob, 'P2'); jobSheet = false" />
          <VListItem title="取消" @click="cancelJob(activeJob); jobSheet = false" />
        </VList>
      </VCard>
    </VBottomSheet>

    <VBottomSheet v-model="enqueueSheet" inset rounded="t-xl">
      <VCard class="pa-4">
        <VCardTitle>提交处理</VCardTitle>
        <div class="text-medium-emphasis mb-2">将提交 {{ enqueueForm.items?.length || 1 }} 个文件</div>
        <VAlert
          v-if="enqueueForm.items?.some(item => item.already_processed)"
          type="warning"
          variant="tonal"
          class="mb-3"
        >
          所选里有已经翻译过的文件。覆盖开关打开时，这次会直接替换已生成的字幕。
        </VAlert>
        <VAlert v-if="enqueueHasStrm" type="warning" variant="tonal" class="mb-3">
          勾选里含 STRM：只搜外挂 / 翻译 / 导出，不做 ASR 和调轴。
        </VAlert>
        <VSelect v-model="enqueueForm.priority" label="优先级" :items="['P0', 'P1', 'P2']" />
        <VSwitch v-model="enqueueForm.overwrite" label="再次翻译时覆盖已有字幕" hide-details class="mb-1" />
        <div class="text-caption text-medium-emphasis mb-2">打开后这次直接替换目标字幕，不再受设置里「跳过已有文件」的影响。已有中字也不会拦住这次入队。</div>
        <VSwitch v-model="enqueueForm.force" label="强制入队（忽略已有中字等门禁）" hide-details class="mb-2" />
        <VBtn color="primary" block class="ss-touch mt-2" :loading="submitting" @click="confirmEnqueue">确认提交</VBtn>
      </VCard>
    </VBottomSheet>

    <VBottomSheet v-model="hideSheet" inset rounded="t-xl">
      <VCard class="pa-4">
        <VCardTitle>从列表移除</VCardTitle>
        <div class="text-medium-emphasis mb-3">
          {{ hideForm.title ? `${hideForm.title} · ` : '' }}{{ hideForm.paths.length }} 个文件。只从字幕工坊的媒体列表里隐藏，不会删除视频，也不会改 MoviePilot 的整理记录。
        </div>
        <VSwitch v-model="hideForm.deleteSubtitles" label="同时删除已生成的字幕文件" hide-details class="mb-1" />
        <div class="text-caption text-medium-emphasis mb-3">默认不删文件。勾选后只删除本插件按当前导出设置写出的字幕和备份，原片和其他外挂会留下。</div>
        <VBtn color="primary" block class="ss-touch" :loading="hiding" @click="confirmHide">确认移除</VBtn>
      </VCard>
    </VBottomSheet>

    <VBottomSheet v-model="cueSheet" inset rounded="t-xl">
      <VCard v-if="editingCue" class="pa-4">
        <VCardTitle>编辑句子</VCardTitle>
        <VTextField
          v-for="lang in (config.target_languages || ['zh-Hans'])"
          :key="lang"
          :model-value="editingCue.texts[lang]"
          :label="langLabel(lang)"
          @update:model-value="editingCue.texts[lang] = $event"
        />
        <VTextField v-model.number="editingCue.start_ms" label="开始（毫秒）" type="number" />
        <VTextField v-model.number="editingCue.end_ms" label="结束（毫秒）" type="number" />
        <VBtn color="primary" block class="ss-touch" @click="saveCue">保存到 CueGraph</VBtn>
      </VCard>
    </VBottomSheet>

    <VDialog v-model="modelPicker.open" max-width="420">
      <VCard>
        <VCardTitle>选择模型</VCardTitle>
        <VList>
          <VListItem v-for="item in modelPicker.items" :key="item" :title="item" @click="applyModel(item)" />
        </VList>
      </VCard>
    </VDialog>
  </div>
</template>
