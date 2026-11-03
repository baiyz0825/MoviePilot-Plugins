<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { createStudioApi } from '../api/studioApi'

const props = defineProps({
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  sourcePluginId: { type: String, default: 'SubtitleStudio' },
  config: { type: Object, default: () => ({}) },
  allowRefresh: { type: Boolean, default: true },
})

const emit = defineEmits(['action'])
const counts = ref({})
const pluginApi = createStudioApi(props.api, `plugin/${props.pluginId || props.sourcePluginId}`)
let timer = null

async function load() {
  try {
    const data = await pluginApi.status()
    counts.value = data?.counts || {}
  } catch {
    counts.value = {}
  }
}

function openQueue() {
  emit('action', { page: 'AppPage', nav: 'jobs' })
}

onMounted(() => {
  load()
  if (props.allowRefresh) {
    timer = setInterval(load, 15000)
  }
})
onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
})
defineExpose({ load })
</script>

<template>
  <VCard class="ss-card" role="button" @click="openQueue">
    <VCardTitle class="d-flex align-center justify-space-between">
      <span>字幕工坊队列</span>
      <VBtn v-if="allowRefresh" size="small" variant="text" class="ss-touch" @click.stop="load">刷新</VBtn>
    </VCardTitle>
    <VCardText>
      运行中 {{ counts.running || 0 }} · 等待 {{ counts.pending || 0 }} · 失败 {{ counts.failed || 0 }}
      <div class="text-caption text-medium-emphasis mt-1">点击打开工作台队列</div>
    </VCardText>
  </VCard>
</template>
