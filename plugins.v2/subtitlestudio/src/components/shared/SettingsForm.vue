<script setup>
import { computed, ref, watch } from 'vue'
import { LANGS, PANES, PRESETS, applyPreset, exportPreview, langLabel, sizeForRank } from '../../composables/fields'

const props = defineProps({
  modelValue: { type: Object, required: true },
  fields: { type: Array, default: () => [] },
  mobile: { type: Boolean, default: false },
})
const emit = defineEmits(['update:modelValue', 'test-endpoint', 'list-models', 'dirty'])

const pane = ref('basic')
const config = computed({
  get: () => props.modelValue,
  set: value => emit('update:modelValue', value),
})

watch(config, () => emit('dirty', true), { deep: true })

const grouped = computed(() => {
  const rows = props.fields.length ? props.fields : fallbackFields()
  return rows.filter(item => item.pane === pane.value && item.mock !== 'skip')
})

function setField(key, value) {
  config.value = { ...config.value, [key]: value }
}

function toggleList(key, value) {
  const current = [...(config.value[key] || [])]
  const index = current.indexOf(value)
  if (index >= 0) {
    if (key === 'target_languages' && current.length <= 1) return
    current.splice(index, 1)
  } else if (key === 'target_languages' && current.length >= 3) {
    return
  } else {
    current.push(value)
  }
  setField(key, current)
}

function moveLang(index, delta) {
  const langs = [...(config.value.target_languages || [])]
  const next = index + delta
  if (next < 0 || next >= langs.length) return
  ;[langs[index], langs[next]] = [langs[next], langs[index]]
  setField('target_languages', langs)
}

function usePreset(value) {
  config.value = applyPreset(config.value, value)
}

function addEndpoint() {
  const rows = [...(config.value.openai_endpoints || [])]
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
  })
  setField('openai_endpoints', rows)
}

function updateEndpoint(index, patch) {
  const rows = (config.value.openai_endpoints || []).map((item, idx) => (idx === index ? { ...item, ...patch } : item))
  if (patch.primary) {
    rows.forEach((item, idx) => {
      item.primary = idx === index
    })
  }
  setField('openai_endpoints', rows)
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

const files = computed(() => exportPreview(config.value))
</script>

<template>
  <div class="ss-settings">
    <VBtnToggle v-if="!mobile" v-model="pane" mandatory density="comfortable" class="mb-4 flex-wrap">
      <VBtn v-for="item in PANES" :key="item[0]" :value="item[0]" class="ss-touch">{{ item[1] }}</VBtn>
    </VBtnToggle>
    <VExpansionPanels v-else v-model="pane">
      <VExpansionPanel v-for="item in PANES" :key="item[0]" :value="item[0]" :title="item[1]" />
    </VExpansionPanels>

    <div class="mt-4">
      <template v-for="field in grouped" :key="field.key">
        <div class="mb-5">
          <VSwitch
            v-if="field.control === 'switch'"
            :model-value="config[field.key]"
            :label="field.label"
            color="primary"
            hide-details
            @update:model-value="setField(field.key, $event)"
          />
          <VTextarea
            v-else-if="field.control === 'textarea'"
            :model-value="config[field.key]"
            :label="field.label"
            :placeholder="field.placeholder"
            auto-grow
            :rows="field.rows || 4"
            @update:model-value="setField(field.key, $event)"
          />
          <VTextField
            v-else-if="field.control === 'password'"
            :model-value="config[field.key]"
            :label="field.label"
            type="password"
            @update:model-value="setField(field.key, $event)"
          />
          <VTextField
            v-else-if="field.control === 'number'"
            :model-value="config[field.key]"
            :label="field.label"
            type="number"
            @update:model-value="setField(field.key, Number($event))"
          />
          <VTextField
            v-else-if="field.control === 'text'"
            :model-value="config[field.key]"
            :label="field.label"
            :placeholder="field.placeholder"
            @update:model-value="setField(field.key, $event)"
          />
          <VSelect
            v-else-if="field.control === 'select'"
            :model-value="config[field.key]"
            :label="field.label"
            :items="field.options"
            @update:model-value="setField(field.key, $event)"
          />
          <div v-else-if="field.control === 'multi'">
            <div class="text-body-2 mb-2">{{ field.label }}</div>
            <VChip
              v-for="option in field.options"
              :key="option.value"
              class="ma-1 ss-touch"
              :color="(config[field.key] || []).includes(option.value) ? 'primary' : undefined"
              filter
              @click="toggleList(field.key, option.value)"
            >
              {{ option.title }}
            </VChip>
          </div>
          <div v-else-if="field.control === 'preset'">
            <div class="text-body-2 mb-2">{{ field.label }}</div>
            <VRow dense>
              <VCol v-for="item in PRESETS" :key="item.value" cols="12" md="3">
                <VCard class="ss-card" :color="config.export_preset === item.value ? 'primary' : undefined" variant="tonal" @click="usePreset(item.value)">
                  <VCardTitle class="text-subtitle-1">{{ item.title }}</VCardTitle>
                  <VCardText>{{ item.hint }}</VCardText>
                </VCard>
              </VCol>
            </VRow>
          </div>
          <div v-else-if="field.control === 'lang-order'">
            <div class="text-body-2 mb-2">{{ field.label }}</div>
            <div v-for="(lang, index) in config.target_languages || []" :key="lang" class="d-flex align-center ga-2 mb-2">
              <VChip color="primary">{{ index + 1 }} · {{ langLabel(lang) }} · {{ sizeForRank(index) }}px</VChip>
              <VBtn icon="mdi-arrow-up" size="small" class="ss-touch" @click="moveLang(index, -1)" />
              <VBtn icon="mdi-arrow-down" size="small" class="ss-touch" @click="moveLang(index, 1)" />
              <VBtn icon="mdi-close" size="small" class="ss-touch" @click="toggleList('target_languages', lang)" />
            </div>
            <VChip
              v-for="item in LANGS.filter(lang => !(config.target_languages || []).includes(lang.value))"
              :key="item.value"
              class="ma-1 ss-touch"
              @click="toggleList('target_languages', item.value)"
            >
              加 {{ item.title }}
            </VChip>
          </div>
          <div v-else-if="field.control === 'endpoints'">
            <div class="d-flex align-center mb-2">
              <div class="text-body-2">{{ field.label }}</div>
              <VSpacer />
              <VBtn size="small" class="ss-touch" @click="addEndpoint">加线路</VBtn>
            </div>
            <VCard v-for="(endpoint, index) in config.openai_endpoints || []" :key="endpoint.endpoint_id" class="ss-card mb-3 pa-3">
              <VTextField v-model="endpoint.name" label="名称" @update:model-value="updateEndpoint(index, { name: $event })" />
              <VTextField v-model="endpoint.api_url" label="API URL" @update:model-value="updateEndpoint(index, { api_url: $event })" />
              <VTextField v-model="endpoint.api_key" label="API Key" type="password" @update:model-value="updateEndpoint(index, { api_key: $event })" />
              <VTextField v-model="endpoint.model" label="模型" @update:model-value="updateEndpoint(index, { model: $event })" />
              <div class="d-flex flex-wrap ga-2">
                <VSwitch :model-value="endpoint.enabled" label="启用" hide-details @update:model-value="updateEndpoint(index, { enabled: $event })" />
                <VSwitch :model-value="endpoint.primary" label="主线路" hide-details @update:model-value="updateEndpoint(index, { primary: $event })" />
                <VSwitch :model-value="endpoint.use_proxy" label="该线路使用代理" hide-details @update:model-value="updateEndpoint(index, { use_proxy: $event })" />
                <VSwitch :model-value="endpoint.compatible" label="兼容模式" hide-details @update:model-value="updateEndpoint(index, { compatible: $event })" />
              </div>
              <div class="d-flex ga-2 mt-2">
                <VBtn size="small" class="ss-touch" @click="$emit('test-endpoint', endpoint)">测连通</VBtn>
                <VBtn size="small" class="ss-touch" @click="$emit('list-models', endpoint)">拉模型</VBtn>
              </div>
            </VCard>
          </div>
          <div class="ss-help">{{ field.purpose }} {{ field.after }}</div>
        </div>
      </template>
    </div>

    <VAlert v-if="pane === 'export'" type="info" variant="tonal" class="mt-2">
      将写出：{{ files.join(' · ') || '请至少勾一种语种、版式和格式' }}
    </VAlert>
  </div>
</template>
