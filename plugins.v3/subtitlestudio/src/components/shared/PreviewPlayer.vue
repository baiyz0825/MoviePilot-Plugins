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

const dialogue = computed(() => {
  const rows = active.value.filter(item => item.kind !== 'note' && item.kind !== 'sdh')
  if (props.tracks.includes('sdh')) {
    return active.value.filter(item => item.kind === 'sdh').concat(rows)
  }
  return rows
})
const notes = computed(() => active.value.filter(item => item.kind === 'note'))
const showDialogue = computed(() => props.tracks.includes('dialogue') || props.tracks.includes('stacked') || props.tracks.includes('sdh'))
const stackedMode = computed(() => props.tracks.includes('stacked'))

function textOf(cue, lang) {
  return cue?.texts?.[lang] || cue?.texts?.source || Object.values(cue?.texts || {})[0] || ''
}

function stackedText(cue) {
  return (props.langs || []).map(lang => textOf(cue, lang)).filter(Boolean).join('\n')
}

function lineStyle(index, kind = 'dialogue') {
  const css = overlayCss(props.styleConfig, index, kind)
  if (kind === 'note') return css
  if (props.stack === 'main_top') {
    return { ...css, top: `${18 + index * 30}px`, bottom: 'auto', whiteSpace: 'pre-line' }
  }
  return { ...css, whiteSpace: 'pre-line' }
}

function onTime(event) {
  emit('time', Math.floor((event.target.currentTime || 0) * 1000))
}
</script>

<template>
  <div v-if="enabled" class="ss-player">
    <video v-if="videoUrl" :src="videoUrl" controls playsinline @timeupdate="onTime" />
    <div v-else class="ss-board pa-6 text-center">
      <div class="text-medium-emphasis">字幕黑板 · 当前没有可播原片（STRM 或需重新打开任务拉取预览）</div>
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
    <template v-if="showDialogue">
      <template v-if="stackedMode">
        <div
          v-for="(cue, index) in dialogue"
          :key="cue.cue_id"
          class="ss-overlay dialogue"
          :style="lineStyle(Math.min(index, 2))"
        >
          {{ stackedText(cue) }}
        </div>
      </template>
      <template v-else>
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
    </template>
  </div>
</template>
