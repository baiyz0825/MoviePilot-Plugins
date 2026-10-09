<script setup>
import { onMounted, ref } from 'vue'
import { createStudioApi } from '../api/studioApi'

const props = defineProps({
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  sourcePluginId: { type: String, default: 'SubtitleStudio' },
  config: { type: Object, default: () => ({}) },
  allowRefresh: { type: Boolean, default: true },
})

const counts = ref({})
const pluginApi = createStudioApi(props.api, `plugin/${props.pluginId}`)

async function load() {
  try {
    const data = await pluginApi.status()
    counts.value = data?.counts || {}
  } catch {
    counts.value = {}
  }
}

onMounted(load)
defineExpose({ load })
</script>

<template>
  <VCard class="ss-card">
    <VCardTitle>字幕工坊队列</VCardTitle>
    <VCardText>
      运行中 {{ counts.running || 0 }} · 等待 {{ counts.pending || 0 }} · 失败 {{ counts.failed || 0 }}
    </VCardText>
  </VCard>
</template>
