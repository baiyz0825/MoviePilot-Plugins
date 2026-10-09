<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
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
  navKey: { type: String, default: 'main' },
  hideTitle: { type: Boolean, default: false },
})

const emit = defineEmits(['action'])
const isMobile = useMobileViewport()
const { toast, confirm } = useHostInjects()
const pluginBase = computed(() => `plugin/${props.pluginId || 'SubtitleStudio'}`)
const pluginApi = computed(() => createStudioApi(props.api, pluginBase))

const nav = ref('media')
const loading = ref(false)
const dirty = ref(false)
const mediaQuery = ref('')
const mediaType = ref('')
const mediaGroups = ref([])
const mediaCounts = ref({ files: 0, groups: 0 })
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
const cueSheet = ref(false)
const editingCue = ref(null)
const config = ref({})
const fields = ref([])
const enqueueSheet = ref(false)
const enqueueForm = reactive({ strategy: 'search_then_translate', priority: 'P0', force: true, items: [] })
const selectedFiles = computed(() => (mediaDetail.value?.files || []).filter(item => selectedPaths.value[item.path]))

const tabs = [
  { value: 'media', title: '媒体', icon: 'mdi-filmstrip' },
  { value: 'jobs', title: '队列', icon: 'mdi-playlist-play' },
  { value: 'desk', title: '工作台', icon: 'mdi-subtitles-outline' },
  { value: 'settings', title: '设置', icon: 'mdi-cog-outline' },
]

const duration = computed(() => Math.max(graph.value.duration_ms || 1, ...(graph.value.cues || []).map(item => item.end_ms || 0), 1))

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
    await Promise.all([loadMedia(), loadJobs()])
  } finally {
    loading.value = false
  }
}

async function loadMedia() {
  const data = await pluginApi.value.media(mediaQuery.value, mediaType.value)
  mediaGroups.value = data?.groups || []
  mediaCounts.value = data?.counts || { files: 0, groups: 0 }
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

function openGroup(group) {
  mediaDetail.value = group
  selectedPaths.value = Object.fromEntries((group.files || []).map(item => [item.path, true]))
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

async function loadJobs() {
  const data = await pluginApi.value.jobs(jobsQuery.value, jobStatus.value)
  jobs.value = data?.items || []
}

async function saveConfig() {
  await pluginApi.value.saveConfig(config.value)
  dirty.value = false
  toast.success?.('已保存')
}

async function enqueue(item) {
  enqueueForm.items = item?.path ? [item] : selectedFiles.value
  enqueueForm.path = item?.path || enqueueForm.items[0]?.path
  enqueueForm.title = item?.title || mediaDetail.value?.title
  enqueueForm.media_source = item?.media_source
  enqueueForm.media_id = item?.media_id
  enqueueForm.tmdbid = item?.tmdbid
  enqueueForm.doubanid = item?.doubanid
  enqueueForm.force = true
  enqueueSheet.value = true
}

async function submitSelected() {
  if (!selectedFiles.value.length) {
    toast.error?.('先勾选要识别的文件')
    return
  }
  enqueueForm.items = selectedFiles.value
  enqueueForm.force = true
  enqueueSheet.value = true
}

async function confirmEnqueue() {
  submitting.value = true
  try {
    const items = enqueueForm.items?.length ? enqueueForm.items : [enqueueForm]
    await pluginApi.value.createJobs({
      items,
      strategy: enqueueForm.strategy,
      priority: enqueueForm.priority,
      force: enqueueForm.force,
    })
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
  activeJob.value = job
  graph.value = (await pluginApi.value.cues(job.job_id)) || { cues: [], notes: [] }
  nav.value = 'desk'
}

async function cutIn(job) {
  await pluginApi.value.cutIn(job.job_id)
  await loadJobs()
}

async function changePriority(job, priority) {
  await pluginApi.value.setPriority(job.job_id, priority)
  await loadJobs()
}

async function cancelJob(job) {
  if (!(await confirm('取消这个任务？'))) return
  await pluginApi.value.cancel(job.job_id)
  await loadJobs()
}

function openCue(cue) {
  editingCue.value = { ...cue, texts: { ...(cue.texts || {}) } }
  cueSheet.value = true
}

async function saveCue() {
  if (!(await confirm('写回这一句到 CueGraph？'))) return
  await pluginApi.value.saveCue(activeJob.value.job_id, editingCue.value.cue_id, editingCue.value)
  cueSheet.value = false
  graph.value = await pluginApi.value.cues(activeJob.value.job_id)
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

onMounted(reload)
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
            <VBtn icon="mdi-arrow-left" class="ss-touch" @click="mediaDetail = null" />
            <strong class="ms-2">{{ mediaDetail.title }}{{ mediaDetail.year ? ` (${mediaDetail.year})` : '' }}</strong>
          </div>
          <div class="text-medium-emphasis mb-2">{{ mediaDetail.library_name || 'MoviePilot 整理记录' }} · {{ mediaDetail.file_count || mediaDetail.files?.length || 0 }} 个文件</div>
          <div class="d-flex ga-2 mb-3">
            <VBtn size="small" variant="text" class="ss-touch" @click="toggleAll(true)">全选</VBtn>
            <VBtn size="small" variant="text" class="ss-touch" @click="toggleAll(false)">清空</VBtn>
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
              <VListItemSubtitle>{{ item.sidecars?.length || 0 }} 条外挂{{ item.is_strm ? ' · STRM' : '' }}</VListItemSubtitle>
            </VListItem>
          </VList>
          <VBtn color="primary" block class="ss-touch mt-3" :disabled="!selectedFiles.length" @click="submitSelected">
            提交识别（{{ selectedFiles.length }}）
          </VBtn>
        </div>
        <template v-else>
          <VTextField v-model="mediaQuery" label="搜索标题或文件名" prepend-inner-icon="mdi-magnify" class="mb-3" @keyup.enter="loadMedia" />
          <div class="d-flex ga-2 mb-3" style="overflow-x:auto">
            <VChip :color="!mediaType ? 'primary' : undefined" @click="mediaType = ''; loadMedia()">全部</VChip>
            <VChip :color="mediaType === 'movie' ? 'primary' : undefined" @click="mediaType = 'movie'; loadMedia()">电影</VChip>
            <VChip :color="mediaType === 'tv' ? 'primary' : undefined" @click="mediaType = 'tv'; loadMedia()">剧集</VChip>
            <VBtn size="small" class="ss-touch" :loading="loading" @click="refreshLibrary">拉取媒体库</VBtn>
          </div>
          <div class="text-medium-emphasis mb-3">整理记录 {{ mediaCounts.groups || 0 }} 部 · {{ mediaCounts.files || 0 }} 个文件</div>
          <VAlert v-if="!mediaGroups.length" type="info" variant="tonal" class="mb-3">
            没有本地媒体。点「拉取媒体库」读取 MoviePilot 整理记录；先在 MoviePilot 里整理入库后才会出现。这里不是 Emby/Jellyfin 在线目录。
          </VAlert>
          <VList>
            <VListItem
              v-for="item in mediaGroups"
              :key="item.id"
              class="ss-card mb-2"
              @click="openGroup(item)"
            >
              <template v-if="item.poster" #prepend>
                <img :src="item.poster" alt="" width="40" height="56" style="object-fit:cover;border-radius:4px" />
              </template>
              <VListItemTitle>{{ item.title }}{{ item.year ? ` (${item.year})` : '' }}</VListItemTitle>
              <VListItemSubtitle>{{ item.type === 'tv' ? '剧集' : '电影' }} · {{ item.file_count || item.files?.length || 0 }} 个文件 · {{ item.library_name }}</VListItemSubtitle>
              <template #append><VIcon icon="mdi-chevron-right" /></template>
            </VListItem>
          </VList>
        </template>
      </template>

      <template v-else-if="nav === 'jobs'">
        <VTextField v-model="jobsQuery" label="搜索标题" prepend-inner-icon="mdi-magnify" class="mb-3" @keyup.enter="loadJobs" />
        <div class="d-flex ga-2 mb-3" style="overflow-x:auto">
          <VChip v-for="item in ['', 'pending', 'running', 'failed', 'success']" :key="item || 'all'" :color="jobStatus === item ? 'primary' : undefined" @click="jobStatus = item; loadJobs()">
            {{ item || '全部' }}
          </VChip>
        </div>
        <VList>
          <VListItem v-for="job in jobs" :key="job.job_id" class="ss-card mb-2">
            <VListItemTitle>{{ job.title }}</VListItemTitle>
            <VListItemSubtitle>{{ job.priority }} · {{ job.status }} · {{ job.trigger }}</VListItemSubtitle>
            <template #append>
              <VBtn v-if="!isMobile && job.status === 'pending'" size="small" class="ss-touch" @click="cutIn(job)">插队</VBtn>
              <VBtn v-if="!isMobile" size="small" variant="text" class="ss-touch" @click="openJob(job)">工作台</VBtn>
              <VBtn icon="mdi-dots-horizontal" class="ss-touch" @click="moreJob(job)" />
            </template>
          </VListItem>
        </VList>
      </template>

      <template v-else-if="nav === 'desk'">
        <div v-if="!activeJob" class="text-medium-emphasis">从队列打开一个任务。</div>
        <template v-else>
          <div class="mb-2">{{ activeJob.title }}</div>
          <PreviewPlayer
            :graph="graph"
            :current-ms="currentMs"
            :langs="config.target_languages || ['zh-Hans']"
            :tracks="previewTracks"
            :stack="config.lang_stack || 'main_bottom'"
            :style-config="config.ass_style || {}"
            :video-url="pluginApi.previewVideoUrl(activeJob.job_id)"
            :enabled="config.preview_enabled !== false"
            @time="currentMs = $event"
          />
          <div class="d-flex ga-2 my-3">
            <VChip v-for="track in ['dialogue', 'notes', 'stacked', 'sdh']" :key="track" :color="previewTracks.includes(track) ? 'primary' : undefined" @click="previewTracks.includes(track) ? previewTracks = previewTracks.filter(item => item !== track) : previewTracks = [...previewTracks, track]">
              {{ track }}
            </VChip>
          </div>
          <div class="text-caption mb-1">按时间</div>
          <div class="ss-timeline mb-4" @click="seekTimeline">
            <button
              v-for="cue in graph.cues || []"
              :key="cue.cue_id"
              type="button"
              class="ss-cue"
              :style="{ left: cueLeft(cue), width: cueWidth(cue) }"
              @click.stop="openCue(cue); currentMs = cue.start_ms"
            >
              {{ Object.values(cue.texts || {})[0] }}
            </button>
          </div>
          <VList>
            <VListItem v-for="cue in graph.cues || []" :key="cue.cue_id" @click="openCue(cue)">
              <VListItemTitle>{{ Object.values(cue.texts || {})[0] }}</VListItemTitle>
              <VListItemSubtitle>{{ cue.start_ms }} – {{ cue.end_ms }}</VListItemSubtitle>
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
          @list-models="pluginApi.listModels($event)"
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
          <VListItem title="改成 P0" @click="changePriority(activeJob, 'P0'); jobSheet = false" />
          <VListItem title="改成 P1" @click="changePriority(activeJob, 'P1'); jobSheet = false" />
          <VListItem title="改成 P2" @click="changePriority(activeJob, 'P2'); jobSheet = false" />
          <VListItem title="取消" @click="cancelJob(activeJob); jobSheet = false" />
          <VListItem v-if="activeJob?.path && !activeJob?.job_id" title="入队" @click="enqueue(activeJob); jobSheet = false" />
        </VList>
      </VCard>
    </VBottomSheet>

    <VBottomSheet v-model="enqueueSheet" inset rounded="t-xl">
      <VCard class="pa-4">
        <VCardTitle>手动提交识别</VCardTitle>
        <div class="text-medium-emphasis mb-2">将提交 {{ enqueueForm.items?.length || 1 }} 个文件</div>
        <VSelect v-model="enqueueForm.strategy" label="本次策略" :items="[
          { title: '先搜后译', value: 'search_then_translate' },
          { title: '只搜索', value: 'search_only' },
          { title: '只识别翻译', value: 'translate_only' },
        ]" />
        <VSelect v-model="enqueueForm.priority" label="优先级" :items="['P0', 'P1', 'P2']" />
        <VSwitch v-model="enqueueForm.force" label="强制入队（忽略已有中字等门禁）" hide-details class="mb-2" />
        <VBtn color="primary" block class="ss-touch mt-2" :loading="submitting" @click="confirmEnqueue">确认提交</VBtn>
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
        <VBtn color="primary" block class="ss-touch" @click="saveCue">写回</VBtn>
      </VCard>
    </VBottomSheet>
  </div>
</template>
