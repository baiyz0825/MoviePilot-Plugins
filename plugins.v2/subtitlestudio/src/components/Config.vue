<script setup>
import { onMounted, ref } from 'vue'
import '../styles/theme.css'
import { createStudioApi } from '../api/studioApi'
import { useHostInjects } from '../composables/useHostInjects'
import { useMobileViewport } from '../composables/useMobileViewport'
import SettingsForm from './shared/SettingsForm.vue'

const props = defineProps({
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  initialConfig: { type: Object, default: () => ({}) },
})

const emit = defineEmits(['save', 'close', 'switch'])
const { confirm } = useHostInjects()
const isMobile = useMobileViewport()
const pluginApi = createStudioApi(props.api, `plugin/${props.pluginId || 'SubtitleStudio'}`)
const config = ref({ ...props.initialConfig })
const fields = ref([])
const dirty = ref(false)

onMounted(async () => {
  try {
    const data = await pluginApi.fields()
    fields.value = data.fields || []
  } catch {
    fields.value = []
  }
})

async function save() {
  emit('save', config.value)
}

async function close() {
  if (dirty.value && !(await confirm('有未保存的设置，确定离开？'))) return
  emit('close')
}
</script>

<template>
  <div class="plugin-root pa-3">
    <SettingsForm v-model="config" :fields="fields" :mobile="isMobile" @dirty="dirty = true" />
    <div class="ss-savebar d-flex ga-2 pa-3">
      <VBtn color="primary" class="ss-touch" @click="save">保存</VBtn>
      <VBtn variant="text" class="ss-touch" @click="close">关闭</VBtn>
      <VBtn variant="text" class="ss-touch" @click="emit('switch')">打开全页</VBtn>
    </div>
  </div>
</template>
