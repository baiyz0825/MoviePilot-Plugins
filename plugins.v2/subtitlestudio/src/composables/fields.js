export const LANGS = [
  { title: '简中', value: 'zh-Hans' },
  { title: '繁中', value: 'zh-Hant' },
  { title: '英文', value: 'en' },
  { title: '日文', value: 'ja' },
  { title: '韩文', value: 'ko' },
]

export const PRESETS = [
  { value: 'library_zh', title: '媒体库 / MoviePilot', hint: 'default.chi.zh-cn · 对齐整理记录' },
  { value: 'plex', title: 'Plex', hint: 'chi / eng · 不标 default' },
  { value: 'fnos', title: '飞牛影视', hint: 'chs · 飞牛不认 zh-Hans' },
  { value: 'web', title: 'Infuse / 网页', hint: 'zh-CN · 再勾 VTT' },
  { value: 'legacy', title: '兼容旧库', hint: 'chi / chi&eng' },
]

const LANG_CODES = {
  'zh-Hans': { library_zh: 'chi.zh-cn', plex: 'chi', web: 'zh-CN', fnos: 'chs', legacy: 'chi' },
  'zh-Hant': { library_zh: 'zh-tw', plex: 'cht', web: 'zh-TW', fnos: 'cht', legacy: 'cht' },
  en: { library_zh: 'eng', plex: 'eng', web: 'en', fnos: 'eng', legacy: 'eng' },
  ja: { library_zh: 'ja', plex: 'jpn', web: 'ja', fnos: 'jpn', legacy: 'jpn' },
  ko: { library_zh: 'ko', plex: 'kor', web: 'ko', fnos: 'kor', legacy: 'kor' },
}

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

export const DEFAULT_ASS_STYLE = {
  font_name: 'Arial',
  primary_color: '#FFFFFF',
  outline_color: '#000000',
  back_color: '#000000',
  outline: 2,
  shadow: 2,
  bold: false,
  italic: false,
  sizes: [22, 17, 14],
  note_size: 16,
  note_color: '#B4E0FF',
  margin_v: 20,
  line_gap: 28,
}

export function normalizeStyle(value) {
  const raw = value && typeof value === 'object' ? value : {}
  const sizes = [0, 1, 2].map(index => {
    const next = Number((raw.sizes || DEFAULT_ASS_STYLE.sizes)[index] ?? DEFAULT_ASS_STYLE.sizes[index])
    return Math.max(8, Math.min(72, Number.isFinite(next) ? next : DEFAULT_ASS_STYLE.sizes[index]))
  })
  return {
    ...DEFAULT_ASS_STYLE,
    ...raw,
    font_name: String(raw.font_name || DEFAULT_ASS_STYLE.font_name).replace(/[,\\]/g, '') || 'Arial',
    sizes,
    outline: Math.max(0, Math.min(8, Number(raw.outline ?? 2))),
    shadow: Math.max(0, Math.min(8, Number(raw.shadow ?? 2))),
    note_size: Math.max(8, Math.min(48, Number(raw.note_size ?? 16))),
    margin_v: Math.max(0, Math.min(160, Number(raw.margin_v ?? 20))),
    line_gap: Math.max(8, Math.min(80, Number(raw.line_gap ?? 28))),
    bold: Boolean(raw.bold),
    italic: Boolean(raw.italic),
  }
}

export function sizeForRank(index, style) {
  const sizes = normalizeStyle(style).sizes
  return sizes[index] ?? sizes[sizes.length - 1] ?? 14
}

export function overlayCss(style, rank = 0, kind = 'dialogue') {
  const spec = normalizeStyle(style)
  const size = kind === 'note' ? spec.note_size : sizeForRank(rank, spec)
  const color = kind === 'note' ? spec.note_color : spec.primary_color
  const outline = `${spec.outline}px ${spec.outline_color}`
  return {
    fontFamily: `'${spec.font_name}', sans-serif`,
    fontSize: `${size}px`,
    color,
    fontWeight: spec.bold ? '700' : '400',
    fontStyle: spec.italic ? 'italic' : 'normal',
    textShadow: `0 0 ${spec.outline}px ${spec.outline_color}, 0 ${spec.shadow}px ${spec.shadow}px rgba(0,0,0,.65), -1px 0 ${outline}, 1px 0 ${outline}, 0 -1px ${outline}, 0 1px ${outline}`,
  }
}

function langCode(lang, preset) {
  return LANG_CODES[lang]?.[preset] || LANG_CODES[lang]?.library_zh || lang
}

function buildFilename(stem, langs, ext, { title = '', flags = [], flagFirst = false } = {}) {
  const parts = [stem]
  const langPart = langs.filter(Boolean).join('.')
  const flagParts = flags.filter(Boolean)
  if (flagFirst) {
    parts.push(...flagParts)
    if (langPart) parts.push(langPart)
    if (title) parts.push(title)
  } else {
    if (langPart) parts.push(langPart)
    if (title) parts.push(title)
    parts.push(...flagParts)
  }
  return `${parts.join('.')}.${ext}`
}

export function applyPreset(config, preset) {
  const next = { ...config, export_preset: preset }
  if (preset === 'plex' || preset === 'fnos') next.mark_default = false
  if (preset === 'library_zh' || preset === 'web') next.mark_default = true
  const formats = new Set(next.export_formats || ['srt'])
  formats.add('srt')
  if (preset === 'web') formats.add('vtt')
  next.export_formats = Array.from(formats)
  return next
}

export function exportPreview(config, stem = 'Movie') {
  const preset = config.export_preset || 'library_zh'
  const langs = (config.target_languages || ['zh-Hans']).slice(0, 3)
  const layouts = config.export_layouts || ['mono']
  const formats = config.export_formats || ['srt']
  const markDefault = Boolean(config.mark_default) && preset !== 'plex' && preset !== 'fnos'
  const flagFirst = preset === 'library_zh'
  const files = []
  for (const layout of layouts) {
    for (const fmt of formats) {
      if (layout === 'mono') {
        const flags = markDefault && langs[0]?.startsWith('zh') ? ['default'] : []
        files.push(buildFilename(stem, [langCode(langs[0], preset)], fmt, { flags, flagFirst }))
      }
      if (layout === 'stacked' && langs.length > 1) {
        const codes = langs.map(item => langCode(item, preset))
        if (preset === 'legacy' || preset === 'fnos') {
          files.push(buildFilename(stem, [codes.join('&')], fmt, { flagFirst }))
        } else {
          files.push(buildFilename(stem, [codes[0]], fmt, { title: 'bilingual', flagFirst }))
        }
      }
      if (layout === 'split') {
        langs.forEach((lang, index) => {
          const flags = markDefault && index === 0 && lang.startsWith('zh') ? ['default'] : []
          files.push(buildFilename(stem, [langCode(lang, preset)], fmt, { flags, flagFirst }))
        })
      }
    }
  }
  if (config.effects_enabled) files.push(buildFilename(stem, [langCode(langs[0], preset)], 'ass', { title: 'notes', flagFirst }))
  if (config.enable_sdh) files.push(buildFilename(stem, [langCode(langs[0], preset)], 'srt', { title: 'sdh', flagFirst }))
  if (config.save_asr_track) files.push(buildFilename(stem, [langCode('en', preset)], 'srt', { title: 'asr', flagFirst }))
  return files
}
