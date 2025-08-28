// Simple CSV/XLSX export (XLSX via dynamic import of SheetJS when needed)
// All comments must be in English.

import type { EvaluationItem } from '@/core/types'

export type ExportMeta = {
  runId?: string
  filters?: Record<string, unknown>
  thresholds?: Record<string, { warning: number; critical: number }>
  timestamp?: string
}

export function exportTableToCSV(filename: string, rows: Array<Record<string, unknown>>, meta?: ExportMeta) {
  const cols = Object.keys(rows[0] || {})
  const esc = (v: unknown) => {
    const s = v == null ? '' : String(v)
    if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"'
    return s
  }
  const lines = [cols.join(',')]
  for (const r of rows) lines.push(cols.map((c) => esc(r[c])).join(','))
  if (meta) {
    lines.push('')
    lines.push(`# runId: ${meta.runId || ''}`)
    lines.push(`# timestamp: ${meta.timestamp || new Date().toISOString()}`)
    if (meta.filters) lines.push(`# filters: ${JSON.stringify(meta.filters)}`)
    if (meta.thresholds) lines.push(`# thresholds: ${JSON.stringify(meta.thresholds)}`)
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
  }
  const wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'array' })
  const blob = new Blob([wbout], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
  triggerDownloadBlob(filename, blob)
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

export function buildRowsWithBookmarks(
  items: EvaluationItem[],
  visibleCols: string[],
  bookmarked: Set<string> | string[]
): Array<Record<string, unknown>> {
  const ids = new Set(Array.isArray(bookmarked) ? bookmarked.map((x) => String(x)) : Array.from(bookmarked))
  const base = buildRowsFromItems(items, visibleCols)
  return base.map((r) => ({ ...r, bookmarked: ids.has(String((r as any).id)) }))
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
