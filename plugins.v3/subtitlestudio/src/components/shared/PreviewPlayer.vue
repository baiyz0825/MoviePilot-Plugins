<script setup>
import { computed } from 'vue'
import { overlayCss } from '../../composables/fields'

const props = defineProps({
  graph: { type: Object, default: () => ({ cues: [], notes: [] }) },
  currentMs: { type: Number, default: 0 },
  langs: { type: Array, default: () => ['zh-Hans'] },
  tracks: { type: Array, default: () => ['dialogue', 'notes'] },
  stack: { type: String, default: 'main_bottom' },
  videoUrl: { type: String, default: '' },
  enabled: { type: Boolean, default: true },
  styleConfig: { type: Object, default: () => ({}) },
})

const emit = defineEmits(['time'])

const active = computed(() => {
  const cues = [...(props.graph?.cues || []), ...(props.graph?.notes || [])]
  return cues.filter(item => item.start_ms <= props.currentMs && props.currentMs < item.end_ms)
})

const dialogue = computed(() => active.value.filter(item => item.kind !== 'note'))
const notes = computed(() => active.value.filter(item => item.kind === 'note'))

function textOf(cue, lang) {
  return cue?.texts?.[lang] || cue?.texts?.source || Object.values(cue?.texts || {})[0] || ''
}

function lineStyle(index, kind = 'dialogue') {
  const css = overlayCss(props.styleConfig, index, kind)
  if (kind === 'note') return css
  if (props.stack === 'main_top' && index === 0) {
    return { ...css, top: '72px', bottom: 'auto' }
  }
  return css
}

function onTime(event) {
  emit('time', Math.floor((event.target.currentTime || 0) * 1000))
}
</script>

<template>
  <div v-if="enabled" class="ss-player">
    <video v-if="videoUrl" :src="videoUrl" controls @timeupdate="onTime" />
    <div v-else class="ss-board pa-6 text-center">
      <div class="text-medium-emphasis">字幕黑板 · 当前没有可播原片</div>
    </div>
    <div
      v-if="tracks.includes('notes')"
      v-for="note in notes"
      :key="note.cue_id"
      class="ss-overlay note"
      :style="lineStyle(0, 'note')"
    >
      {{ textOf(note) }}
    </div>
    <template v-if="tracks.includes('dialogue')">
      <div
        v-for="(lang, index) in langs"
        :key="lang"
        class="ss-overlay dialogue"
        :class="[`rank${index + 1}`]"
        :style="lineStyle(index)"
      >
        {{ dialogue.map(item => textOf(item, lang)).filter(Boolean).join(' ') }}
      </div>
    </template>
  </div>
</template>
