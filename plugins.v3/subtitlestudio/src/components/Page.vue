<script setup>
import { ref } from 'vue'
import AppPage from './AppPage.vue'

defineProps({
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  sourcePluginId: { type: String, default: 'SubtitleStudio' },
  navKey: { type: String, default: 'main' },
})

const emit = defineEmits(['close', 'action', 'switch'])
const pageRef = ref(null)
</script>

<template>
  <div>
    <VToolbar density="comfortable">
      <div class="text-h6 ms-3">字幕工坊</div>
      <VSpacer />
      <VBtn icon="mdi-refresh" variant="text" @click="pageRef?.reload()" />
      <VBtn icon="mdi-close" variant="text" @click="emit('close')" />
    </VToolbar>
    <VDivider />
    <AppPage ref="pageRef" v-bind="$props" hide-title @action="emit('action')" />
  </div>
</template>
