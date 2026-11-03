<script setup>
import { onMounted, reactive, ref } from 'vue'
import '../styles/theme.css'
import { createStudioApi } from '../api/studioApi'
import { useHostInjects } from '../composables/useHostInjects'
import { useMobileViewport } from '../composables/useMobileViewport'
import SettingsForm from './shared/SettingsForm.vue'

const props = defineProps({
  api: { type: Object, default: () => ({}) },
  pluginId: { type: String, default: 'SubtitleStudio' },
  sourcePluginId: { type: String, default: 'SubtitleStudio' },
  initialConfig: { type: Object, default: () => ({}) },
})

const emit = defineEmits(['save', 'close', 'switch'])
const { confirm, toast } = useHostInjects()
const isMobile = useMobileViewport()
const pluginApi = createStudioApi(props.api, `plugin/${props.pluginId || props.sourcePluginId || 'SubtitleStudio'}`)
const config = ref({ ...props.initialConfig })
const fields = ref([])
const dirty = ref(false)
const modelPicker = reactive({ open: false, endpointId: '', items: [] })

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

async function onListModels(endpoint) {
  try {
    const data = await pluginApi.listModels(endpoint)
    const items = data?.items || []
    if (!items.length) {
      toast.error?.('没有拉到模型')
      return
    }
    modelPicker.open = true
    modelPicker.endpointId = endpoint.endpoint_id
    modelPicker.items = items
  } catch (error) {
    toast.error?.(error?.message || '拉模型失败')
  }
}

function applyModel(model) {
  config.value = {
    ...config.value,
    openai_endpoints: (config.value.openai_endpoints || []).map(item => (
      item.endpoint_id === modelPicker.endpointId ? { ...item, model } : item
    )),
  }
  dirty.value = true
  modelPicker.open = false
  toast.success?.(`已选模型 ${model}`)
}
</script>

<template>
  <div class="plugin-root pa-3">
    <SettingsForm
      v-model="config"
      :fields="fields"
      :mobile="isMobile"
      @dirty="dirty = true"
      @test-endpoint="pluginApi.testEndpoint($event)"
      @list-models="onListModels"
    />
    <div class="ss-savebar d-flex ga-2 pa-3">
      <VBtn color="primary" class="ss-touch" @click="save">保存</VBtn>
      <VBtn variant="text" class="ss-touch" @click="close">关闭</VBtn>
      <VBtn variant="text" class="ss-touch" @click="emit('switch')">打开全页</VBtn>
    </div>
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
