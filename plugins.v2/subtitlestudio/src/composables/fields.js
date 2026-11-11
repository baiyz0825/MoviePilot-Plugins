import namingPresetData from '../../core/naming_presets.json'

export const LANGS = [
  { title: '简中', value: 'zh-Hans' },
  { title: '繁中', value: 'zh-Hant' },
  { title: '英文', value: 'en' },
  { title: '日文', value: 'ja' },
  { title: '韩文', value: 'ko' },
]

const PRESET_BY_ID = Object.fromEntries(
  namingPresetData.profiles.map(profile => [profile.id, profile]),
)

export const PRESETS = namingPresetData.profiles
  .filter(profile => profile.visible !== false)
  .map(profile => ({ value: profile.id, title: profile.title, hint: profile.hint, group: profile.group }))

const presetGroups = new Map()
for (const preset of PRESETS) {
  const items = presetGroups.get(preset.group) || []
  items.push(preset)
  presetGroups.set(preset.group, items)
}
export const PRESET_GROUPS = Array.from(presetGroups, ([title, items]) => ({ title, items }))

export const PANES = [
  ['basic', '基础'],
  ['ingest', '入库与监控'],
  ['search', '搜索偏好'],
  ['export', '导出包'],
  ['effects', '特效字幕'],
  ['asr', '识别 ASR'],
  ['model', '翻译与大模型'],
  ['quality', '质量与格式修复'],
  ['queue', '调轴与队列'],
]

export function langLabel(value) {
  return LANGS.find(item => item.value === value)?.title || value
}

export const STYLE_FONTS = [
  'Arial',
  'Microsoft YaHei',
  'PingFang SC',
  'Source Han Sans SC',
  'Noto Sans CJK SC',
  'SimHei',
  'Helvetica',
]

export const STYLE_LOOKS = [
  { title: '白字黑边', patch: { primary_color: '#FFFFFF', outline_color: '#000000', outline: 2, shadow: 2 } },
  { title: '黄字黑边', patch: { primary_color: '#FFE566', outline_color: '#000000', outline: 3, shadow: 1 } },
  { title: '白字软影', patch: { primary_color: '#FFFFFF', outline_color: '#000000', outline: 1, shadow: 4 } },
]

export const ASS_REFERENCE_HEIGHT = 1080

export const FRAME_LIMITS = {
  size: [8, 108],
  note: [8, 80],
  margin: [0, 216],
  gap: [8, 160],
}

export function frameShare(size) {
  const number = Number(size)
  if (!Number.isFinite(number)) return ''
  return `${(number / ASS_REFERENCE_HEIGHT * 100).toFixed(1)}%`
}

export function frameLabel(size) {
  const number = Math.round(Number(size) || 0)
  return `${number} · ${frameShare(number)}`
}

export const DEFAULT_ASS_STYLE = {
  font_name: 'Arial',
  primary_color: '#FFFFFF',
  outline_color: '#000000',
  back_color: '#000000',
  outline: 2,
  shadow: 0,
  bold: false,
  italic: false,
  sizes: [60, 47, 38],
  note_size: 32,
  note_color: '#B4E0FF',
  margin_v: 40,
  line_gap: 68,
}

export function normalizeStyle(value) {
  const raw = value && typeof value === 'object' ? value : {}
  const sizes = [0, 1, 2].map(index => {
    const next = Number((raw.sizes || DEFAULT_ASS_STYLE.sizes)[index] ?? DEFAULT_ASS_STYLE.sizes[index])
    return Math.max(FRAME_LIMITS.size[0], Math.min(FRAME_LIMITS.size[1], Number.isFinite(next) ? next : DEFAULT_ASS_STYLE.sizes[index]))
  })
  return {
    ...DEFAULT_ASS_STYLE,
    ...raw,
    font_name: String(raw.font_name || DEFAULT_ASS_STYLE.font_name).replace(/[,\\]/g, '') || 'Arial',
    sizes,
    outline: Math.max(0, Math.min(8, Number(raw.outline ?? DEFAULT_ASS_STYLE.outline))),
    shadow: Math.max(0, Math.min(8, Number(raw.shadow ?? DEFAULT_ASS_STYLE.shadow))),
    note_size: Math.max(FRAME_LIMITS.note[0], Math.min(FRAME_LIMITS.note[1], Number(raw.note_size ?? DEFAULT_ASS_STYLE.note_size))),
    margin_v: Math.max(FRAME_LIMITS.margin[0], Math.min(FRAME_LIMITS.margin[1], Number(raw.margin_v ?? DEFAULT_ASS_STYLE.margin_v))),
    line_gap: Math.max(FRAME_LIMITS.gap[0], Math.min(FRAME_LIMITS.gap[1], Number(raw.line_gap ?? DEFAULT_ASS_STYLE.line_gap))),
    bold: Boolean(raw.bold),
    italic: Boolean(raw.italic),
  }
}

export function sizeForRank(index, style) {
  const sizes = normalizeStyle(style).sizes
  return sizes[index] ?? sizes[sizes.length - 1] ?? 14
}

export function overlayCss(style, rank = 0, kind = 'dialogue', scale = 1) {
  const spec = normalizeStyle(style)
  const size = kind === 'note' ? spec.note_size : sizeForRank(rank, spec)
  const color = kind === 'note' ? spec.note_color : spec.primary_color
  const safeScale = Number.isFinite(scale) && scale > 0 ? scale : 1
  const px = Math.max(1, Math.round(size * safeScale))
  const outlinePx = Math.max(0, spec.outline * safeScale)
  const shadowPx = Math.max(0, spec.shadow * safeScale)
  const outline = `${outlinePx}px ${spec.outline_color}`
  const shadow = shadowPx > 0 ? `, 0 ${shadowPx}px ${shadowPx}px rgba(0,0,0,.45)` : ''
  return {
    fontFamily: `'${spec.font_name}', sans-serif`,
    fontSize: `${px}px`,
    color,
    background: 'transparent',
    fontWeight: spec.bold ? '700' : '400',
    fontStyle: spec.italic ? 'italic' : 'normal',
    lineHeight: '1',
    textShadow: `0 0 ${outlinePx}px ${spec.outline_color}, -1px 0 ${outline}, 1px 0 ${outline}, 0 -1px ${outline}, 0 1px ${outline}${shadow}`,
  }
}

function presetSpec(preset) {
  return PRESET_BY_ID[preset] || PRESET_BY_ID.fnos
}

function langCode(lang, preset) {
  const profile = presetSpec(preset)
  return profile.language_codes?.[lang] || profile.language_codes?.['zh-Hans'] || lang
}

function buildFilename(stem, langs, ext, { title = '', flags = [], flagFirst = false, stemSeparator = '.' } = {}) {
  const langPart = langs.filter(Boolean).join('.')
  const suffix = []
  if (flagFirst) {
    suffix.push(...flags.filter(Boolean))
    if (langPart) suffix.push(langPart)
    if (title) suffix.push(title)
  } else {
    if (langPart) suffix.push(langPart)
    if (title) suffix.push(title)
    suffix.push(...flags.filter(Boolean))
  }
  const prefix = suffix.length ? `${stem}${stemSeparator}${suffix.join('.')}` : stem
  return `${prefix}.${ext}`
}

export function supportsDefault(preset) {
  return Boolean(presetSpec(preset).supports_default)
}

function defaultFlags(profile, enabled, lang) {
  if (!enabled || !profile.supports_default) return []
  if (profile.default_language === 'zh' && !lang?.startsWith('zh')) return []
  return ['default']
}

export function applyPreset(config, preset) {
  const profile = presetSpec(preset)
  return {
    ...config,
    export_preset: profile.id,
    mark_default: Boolean(profile.supports_default && profile.mark_default),
  }
}

export function exportPreview(config, stem = 'Movie') {
  const preset = config.export_preset || 'fnos'
  const profile = presetSpec(preset)
  const langs = (config.target_languages || ['zh-Hans']).slice(0, 3)
  const layouts = config.export_layouts || ['mono']
  const formats = [...(config.export_formats || ['srt'])]
  if (layouts.includes('stacked') && langs.length > 1 && !formats.includes('ass')) formats.push('ass')
  const markDefault = Boolean(config.mark_default) && Boolean(profile.supports_default)
  const filenameOptions = {
    flagFirst: Boolean(profile.flag_first),
    stemSeparator: profile.stem_separator || '.',
  }
  const files = []
  for (const layout of layouts) {
    for (const fmt of formats) {
      if (layout === 'mono') {
        const flags = defaultFlags(profile, markDefault, langs[0])
        files.push(buildFilename(stem, [langCode(langs[0], preset)], fmt, { ...filenameOptions, flags }))
      }
      if (layout === 'stacked' && langs.length > 1) {
        // A stacked file is one track. Its filename identifies the primary language only.
        files.push(buildFilename(stem, [langCode(langs[0], preset)], fmt, filenameOptions))
      }
      if (layout === 'split') {
        langs.forEach((lang, index) => {
          const flags = defaultFlags(profile, markDefault && index === 0, lang)
          files.push(buildFilename(stem, [langCode(lang, preset)], fmt, { ...filenameOptions, flags }))
        })
      }
    }
  }
  if (config.effects_enabled) files.push(buildFilename(stem, [langCode(langs[0], preset)], 'ass', { ...filenameOptions, title: 'notes' }))
  if (config.enable_sdh) files.push(buildFilename(stem, [langCode(langs[0], preset)], 'srt', { ...filenameOptions, title: 'sdh' }))
  if (config.save_asr_track) files.push(buildFilename(stem, [langCode('en', preset)], 'srt', { ...filenameOptions, title: 'asr' }))
  return files
}
