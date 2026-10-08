export const LANGS = [
  { title: '简中', value: 'zh-Hans' },
  { title: '繁中', value: 'zh-Hant' },
  { title: '英文', value: 'en' },
  { title: '日文', value: 'ja' },
  { title: '韩文', value: 'ko' },
]

export const PRESETS = [
  { value: 'library_zh', title: '媒体库中文', hint: 'zh-Hans + default · 默认勾 SRT' },
  { value: 'plex', title: 'Plex', hint: 'chi · 不标 default' },
  { value: 'web', title: '网页 / Infuse', hint: '再勾 VTT' },
  { value: 'legacy', title: '兼容旧库', hint: 'chi / chi&eng' },
]

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

export function sizeForRank(index) {
  return [22, 17, 14][index] || 14
}

export function applyPreset(config, preset) {
  const next = { ...config, export_preset: preset }
  if (preset === 'plex') next.mark_default = false
  if (preset === 'library_zh' || preset === 'web') next.mark_default = true
  const formats = new Set(next.export_formats || ['srt'])
  formats.add('srt')
  if (preset === 'web') formats.add('vtt')
  next.export_formats = Array.from(formats)
  return next
}

export function exportPreview(config, stem = 'Movie') {
  const langs = (config.target_languages || ['zh-Hans']).slice(0, 3)
  const layouts = config.export_layouts || ['mono']
  const formats = config.export_formats || ['srt']
  const files = []
  for (const layout of layouts) {
    for (const fmt of formats) {
      if (layout === 'mono') files.push(`${stem}.${langs[0]}.${fmt}`)
      if (layout === 'stacked' && langs.length > 1) files.push(`${stem}.${langs.join('.')}.${fmt}`)
      if (layout === 'split') langs.forEach(lang => files.push(`${stem}.${lang}.${fmt}`))
    }
  }
  if (config.effects_enabled) files.push(`${stem}.${langs[0]}.notes.ass`)
  if (config.enable_sdh) files.push(`${stem}.${langs[0]}.sdh.srt`)
  if (config.save_asr_track) files.push(`${stem}.en.asr.srt`)
  return files
}
