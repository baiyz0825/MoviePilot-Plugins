<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ASS_REFERENCE_HEIGHT, normalizeStyle, overlayCss } from '../../composables/fields'

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

const root = ref(null)
const frameHeight = ref(0)
let observer

function pictureNode() {
  return root.value?.querySelector('video, .ss-board') || root.value
}

function measure() {
  frameHeight.value = pictureNode()?.clientHeight || 0
}

function watchPicture() {
  observer?.disconnect()
  observer = null
  const node = pictureNode()
  if (!node || typeof ResizeObserver === 'undefined') return
  observer = new ResizeObserver(() => measure())
  observer.observe(node)
  measure()
}

onMounted(() => watchPicture())

watch(() => props.videoUrl, async () => {
  await nextTick()
  watchPicture()
})

onBeforeUnmount(() => observer?.disconnect())

const fontScale = computed(() => {
  const height = frameHeight.value || root.value?.clientHeight || 0
  return (height > 0 ? height : 180) / ASS_REFERENCE_HEIGHT
})

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

function sourceText(cue) {
  const texts = cue?.texts || {}
  const sourceLang = props.graph?.source_lang
  return (sourceLang && texts[sourceLang]) || texts.source || Object.values(texts)[0] || ''
}

function textOf(cue, lang) {
  if (!lang) return sourceText(cue)
  const exact = cue?.texts?.[lang] || ''
  if (exact) return exact
  // A single-language preview may show the source as an explicit convenience.
  // Multi-language rows are strict: a missing target stays missing.
  return (props.langs?.length || 0) <= 1 ? sourceText(cue) : ''
}

function dialogueText(lang) {
  return dialogue.value.map(item => textOf(item, lang)).filter(Boolean).join(' ')
}

const dialogueLines = computed(() => {
  const seen = new Set()
  return (props.langs || []).flatMap((lang) => {
    const text = dialogueText(lang)
    const identity = text.replace(/\s+/g, '').toLocaleLowerCase()
    if (!text || seen.has(identity)) return []
    seen.add(identity)
    return [{ lang, text }]
  })
})

function lineStyle(index, kind = 'dialogue') {
  const css = overlayCss(props.styleConfig, index, kind, fontScale.value)
  const spec = normalizeStyle(props.styleConfig)
  const margin = Math.round(spec.margin_v * fontScale.value)
  const gap = Math.round(spec.line_gap * fontScale.value)
  if (kind === 'note') {
    return { ...css, top: `${Math.max(8, margin)}px`, bottom: 'auto' }
  }
  const visibleCount = dialogueLines.value?.length || 1
  const rank = props.stack === 'main_top' ? Math.max(0, visibleCount - index - 1) : index
  return { ...css, bottom: `${margin + rank * gap}px`, top: 'auto', whiteSpace: 'pre-line' }
}

function onTime(event) {
  emit('time', Math.floor((event.target.currentTime || 0) * 1000))
}
</script>

<template>
  <div v-if="enabled" ref="root" class="ss-player">
    <video v-if="videoUrl" :src="videoUrl" controls playsinline @loadedmetadata="measure" @timeupdate="onTime" />
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
      <template>
        <div
          v-for="(line, index) in dialogueLines"
          :key="line.lang"
          class="ss-overlay dialogue"
          :class="[`rank${index + 1}`]"
          :style="lineStyle(index)"
        >
          {{ line.text }}
        </div>
      </template>
    </template>
  </div>
</template>
