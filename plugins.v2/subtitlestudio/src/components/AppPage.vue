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
const { toast, dialog, confirm } = useHostInjects()
const pluginBase = computed(() => `plugin/${props.pluginId || 'SubtitleStudio'}`)
const pluginApi = computed(() => createStudioApi(props.api, pluginBase))

const nav = ref('media')
const loading = ref(false)
const dirty = ref(false)
const mediaQuery = ref('')
const mediaType = ref('')
const mediaItems = ref([])
const mediaDetail = ref(null)
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
const enqueueForm = reactive({ strategy: 'search_then_translate', priority: 'P0' })

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
  mediaItems.value = data?.items || []
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
  enqueueForm.path = item.path
  enqueueForm.title = item.title
  enqueueForm.media_source = item.media_source
  enqueueForm.media_id = item.media_id
  enqueueForm.tmdbid = item.tmdbid
  enqueueForm.doubanid = item.doubanid
  if (isMobile.value && dialog) {
    dialog({ title: '入队覆盖项', fullscreen: true, content: '选择本次策略后入队' })
  }
  enqueueSheet.value = true
}

async function confirmEnqueue() {
  await pluginApi.value.createJob({ ...enqueueForm })
  enqueueSheet.value = false
  nav.value = 'jobs'
  await loadJobs()
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
        <div v-if="isMobile && mediaDetail" class="mb-3">
          <div class="d-flex align-center mb-3">
            <VBtn icon="mdi-arrow-left" class="ss-touch" @click="mediaDetail = null" />
            <strong class="ms-2">{{ mediaDetail.title }}</strong>
          </div>
          <div class="text-medium-emphasis mb-2">{{ mediaDetail.path }}</div>
          <VBtn color="primary" block class="ss-touch mb-2" @click="enqueue(mediaDetail)">入队</VBtn>
          <VBtn variant="tonal" block class="ss-touch" @click="moreJob(mediaDetail)">更多</VBtn>
        </div>
        <template v-else>
          <VTextField v-model="mediaQuery" label="搜索媒体" prepend-inner-icon="mdi-magnify" class="mb-3" @keyup.enter="loadMedia" />
          <div class="d-flex ga-2 mb-3 h-scroll">
            <VChip :color="!mediaType ? 'primary' : undefined" @click="mediaType = ''; loadMedia()">全部</VChip>
            <VChip :color="mediaType === 'movie' ? 'primary' : undefined" @click="mediaType = 'movie'; loadMedia()">电影</VChip>
            <VChip :color="mediaType === 'tv' ? 'primary' : undefined" @click="mediaType = 'tv'; loadMedia()">剧集</VChip>
            <VBtn size="small" class="ss-touch" :loading="loading" @click="pluginApi.refreshMedia().then(loadMedia)">刷新目录</VBtn>
          </div>
          <VList v-if="isMobile">
            <VListItem
              v-for="item in mediaItems"
              :key="item.id"
              class="ss-card mb-2"
              @click="mediaDetail = item"
            >
              <VListItemTitle>{{ item.title }}</VListItemTitle>
              <VListItemSubtitle>{{ item.type }} · {{ item.sidecars?.length || 0 }} 条外挂</VListItemSubtitle>
              <template #append><VIcon icon="mdi-chevron-right" /></template>
            </VListItem>
          </VList>
          <VTable v-else>
            <thead>
              <tr><th>标题</th><th>类型</th><th>外挂</th><th>操作</th></tr>
            </thead>
            <tbody>
              <tr v-for="item in mediaItems" :key="item.id">
                <td>{{ item.title }}</td>
                <td>{{ item.type }}</td>
                <td>{{ item.sidecars?.length || 0 }}</td>
                <td>
                  <VBtn size="small" class="ss-touch" @click="enqueue(item)">入队</VBtn>
                  <VBtn size="small" variant="text" class="ss-touch" @click="moreJob(item)">更多</VBtn>
                </td>
              </tr>
            </tbody>
          </VTable>
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
        <VCardTitle>入队</VCardTitle>
        <VSelect v-model="enqueueForm.strategy" label="本次策略" :items="[
          { title: '先搜后译', value: 'search_then_translate' },
          { title: '只搜索', value: 'search_only' },
          { title: '只识别翻译', value: 'translate_only' },
        ]" />
        <VSelect v-model="enqueueForm.priority" label="优先级" :items="['P0', 'P1', 'P2']" />
        <VBtn color="primary" block class="ss-touch mt-2" @click="confirmEnqueue">确认入队</VBtn>
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
