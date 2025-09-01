// Simple CSV/XLSX export (XLSX via dynamic import of SheetJS when needed)
// All comments must be in English.

import type { EvaluationItem } from '@/core/types'

export type ExportMeta = {
  runId?: string
  filters?: Record<string, unknown>
  thresholds?: Record<string, { warning: number; critical: number }>
  timestamp?: string
  branding?: { brand?: string; title?: string; footer?: string }
}

export function exportTableToCSV(filename: string, rows: Array<Record<string, unknown>>, meta?: ExportMeta) {
  const cols = Object.keys(rows[0] || {})
  const esc = (v: unknown) => {
    const s = v == null ? '' : String(v)
    if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"'
    return s
  }
  const lines: string[] = []
  // Optional branded header lines as comments to keep CSV shape intact
  if (meta?.branding?.brand || meta?.branding?.title) {
    if (meta.branding.brand) lines.push(`# brand: ${meta.branding.brand}`)
    if (meta.branding.title) lines.push(`# title: ${meta.branding.title}`)
  }
  lines.push(cols.join(','))
  for (const r of rows) lines.push(cols.map((c) => esc(r[c])).join(','))
  if (meta) {
    lines.push('')
    lines.push(`# runId: ${meta.runId || ''}`)
    lines.push(`# timestamp: ${meta.timestamp || new Date().toISOString()}`)
    if (meta.filters) lines.push(`# filters: ${JSON.stringify(meta.filters)}`)
    if (meta.thresholds) lines.push(`# thresholds: ${JSON.stringify(meta.thresholds)}`)
    if (meta.branding?.footer) lines.push(`# footer: ${meta.branding.footer}`)
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' })
  triggerDownloadBlob(filename, blob)
}

export async function exportTableToXLSX(filename: string, rows: Array<Record<string, unknown>>, meta?: ExportMeta) {
  const XLSX = await import('xlsx')
  const ws = (XLSX.utils as any).json_to_sheet(rows)
  const wb = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(wb, ws, 'data')
  if (meta) {
    const metaSheet = XLSX.utils.json_to_sheet([
      { key: 'runId', value: meta.runId || '' },
      { key: 'timestamp', value: meta.timestamp || new Date().toISOString() },
      { key: 'filters', value: JSON.stringify(meta.filters || {}) },
      { key: 'thresholds', value: JSON.stringify(meta.thresholds || {}) },
    ])
    XLSX.utils.book_append_sheet(wb, metaSheet, 'meta')
    if (meta.branding && (meta.branding.brand || meta.branding.title || meta.branding.footer)) {
      const brandingRows = [
        { key: 'brand', value: meta.branding.brand || '' },
        { key: 'title', value: meta.branding.title || '' },
        { key: 'footer', value: meta.branding.footer || '' },
      ]
      const brandSheet = XLSX.utils.json_to_sheet(brandingRows)
      XLSX.utils.book_append_sheet(wb, brandSheet, 'branding')
    }
  }
  const wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'array' })
  const blob = new Blob([wbout], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
  triggerDownloadBlob(filename, blob)
}

// PDF Option B (server or external worker) manifest builder.
// This does not perform network or rendering; it only builds a document manifest
// that a caller can send to a server-side renderer (e.g., headless Chrome service).
export interface PdfSection {
  title: string
  type: 'table' | 'image' | 'text'
  payload: any
}

export interface PdfManifest {
  meta?: ExportMeta
  sections: PdfSection[]
}

export function buildPdfManifest(sections: PdfSection[], meta?: ExportMeta): PdfManifest {
  return { meta, sections }
}

export function buildRowsFromItems(items: EvaluationItem[], visibleCols: string[]): Array<Record<string, unknown>> {
  return items.map((it) => {
    const r: Record<string, unknown> = {}
    for (const c of visibleCols) {
      if (c in it) r[c] = (it as any)[c]
      else if ((it.metrics as any)[c] != null) r[c] = (it.metrics as any)[c]
      else r[c] = ''
    }
    return r
  })
}

export interface QAItemRow {
  id: string
  question?: string
  answer?: string
  reference?: string
  bookmarked?: boolean
  // metrics will be spread dynamically
  [k: string]: any
}

export function buildRowsWithBookmarks(items: any[], metricKeys: string[], bookmarks: Set<string>): QAItemRow[] {
  return items.map((it) => {
    const row: QAItemRow = {
      id: it.id,
      question: it.user_input || it.question || '',
      answer: it.rag_answer || it.answer || '',
      reference: it.reference || it.ground_truth || '',
      bookmarked: bookmarks.has(it.id),
    }
    for (const m of metricKeys) {
      const v = it.metrics?.[m] ?? it[m]
      row[m] = typeof v === 'number' ? v : null
    }
    return row
  })
}

function triggerDownloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
