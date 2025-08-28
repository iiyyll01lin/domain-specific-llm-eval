import React from 'react'
import { usePortalStore } from '@/app/store/usePortalStore'
import { applyFilters } from '@/core/analysis/filters'
import { getMetricMeta } from '@/core/metrics/registry'
import { buildRowsFromItems, buildRowsWithBookmarks, exportTableToCSV, exportTableToXLSX } from '@/core/exporter'
import { buildFilterChips } from '@/components/filters/chips'

export const QAFailureExplorer: React.FC = () => {
  const run = usePortalStore((s) => s.run)
  const [metric, setMetric] = React.useState('Faithfulness')
  const [query, setQuery] = React.useState('')
  const [selected, setSelected] = React.useState<number | null>(null)
  const [bookmarks, setBookmarks] = React.useState<Set<string>>(new Set())

  const rows = React.useMemo(() => {
    const items = applyFilters(run?.items || [], usePortalStore.getState().filters)
    const scored = items.map((it, idx) => ({ idx, it, score: (it.metrics as any)?.[metric] as number | undefined }))
    let out = scored.filter((r) => typeof r.score === 'number')
    if (query.trim()) {
      const q = query.toLowerCase()
      out = out.filter((r) => (r.it.user_input || '').toLowerCase().includes(q))
    }
    out.sort((a, b) => (a.score! - b.score!))
  return out
  }, [run, metric, query])

  const columns = ['bookmark', 'id', 'user_input', metric]
  const onExport = async (fmt: 'csv' | 'xlsx') => {
    const visible = ['id', 'user_input', metric]
  const raw = buildRowsWithBookmarks(rows.map((r) => r.it), visible, bookmarks)
    const meta = {
      runId: 'local-run',
      filters: (usePortalStore.getState().filters as any),
      thresholds: usePortalStore.getState().thresholds as any,
      timestamp: new Date().toISOString(),
    }
    if (fmt === 'csv') exportTableToCSV('qa-failures.csv', raw, meta)
    else await exportTableToXLSX('qa-failures.xlsx', raw, meta)
  }

  const toggleBookmark = (id: string|number) => {
    setBookmarks((prev) => {
      const next = new Set(prev)
      const key = String(id)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const exportBookmarks = async () => {
    const visible = ['id', 'user_input', metric]
    const filtered = rows.filter((r) => bookmarks.has(String(r.it.id)))
  const raw = buildRowsWithBookmarks(filtered.map((r) => r.it), visible, bookmarks)
    await exportTableToXLSX('qa-bookmarks.xlsx', raw, {
      runId: 'local-run',
      filters: (usePortalStore.getState().filters as any),
      thresholds: usePortalStore.getState().thresholds as any,
      timestamp: new Date().toISOString(),
    })
  }

  return (
    <section>
      <h2>QA Failure Explorer</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <label>
          Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value)} style={{ marginLeft: 6 }}>
            {Object.keys(run?.kpis || {}).map((k) => (
              <option key={k} value={k}>{getMetricMeta(k as any).key}</option>
            ))}
          </select>
        </label>
        <input placeholder="Search question" value={query} onChange={(e) => setQuery(e.target.value)} style={{ minWidth: 240 }} />
        <span className="small-muted">Sorted by low scores</span>
  <button onClick={() => onExport('csv')}>Export CSV</button>
  <button onClick={() => onExport('xlsx')}>Export XLSX</button>
  <button onClick={exportBookmarks} disabled={bookmarks.size === 0}>Export Bookmarks</button>
      </div>
      {/* Active filter chips */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
        {buildFilterChips(usePortalStore.getState().filters).map((c) => (
          <span key={c.key} style={{ padding: '2px 8px', borderRadius: 12, border: '1px solid var(--border)', background: 'var(--bg-muted)' }}>
            {c.label} <button onClick={c.onClear} aria-label={`clear ${c.key}`} style={{ marginLeft: 6 }}>×</button>
          </span>
        ))}
      </div>
      <VirtualizedTable
        rows={rows}
        columns={columns}
        onRowClick={(r) => setSelected(r.idx)}
        isBookmarked={(id) => bookmarks.has(String(id))}
        onToggleBookmark={(id) => toggleBookmark(id)}
      />
      {selected != null && run?.items[selected] && (
        <div className="card" style={{ marginTop: 12 }}>
          <button onClick={() => setSelected(null)} style={{ float: 'right' }}>✕</button>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Details</div>
          <div><strong>ID:</strong> {run.items[selected].id}</div>
          <div><strong>Question:</strong> {run.items[selected].user_input || '—'}</div>
          <div><strong>Answer:</strong> {run.items[selected].rag_answer || '—'}</div>
          <div><strong>Reference:</strong> {run.items[selected].reference || '—'}</div>
        </div>
      )}
    </section>
  )
}

function fmt(v?: number) {
  if (v == null || Number.isNaN(v)) return 'N/A'
  return v.toFixed(3)
}

function truncate(s?: string, n = 80) {
  if (!s) return '—'
  return s.length > n ? s.slice(0, n) + '…' : s
}

type Row = { idx: number; it: any; score?: number }

type VProps = {
  rows: Row[]
  columns: string[]
  rowHeight?: number
  height?: number
  onRowClick: (r: Row) => void
  isBookmarked: (id: string|number) => boolean
  onToggleBookmark: (id: string|number) => void
}

const VirtualizedTable: React.FC<VProps> = ({ rows, columns, rowHeight = 36, height = 400, onRowClick, isBookmarked, onToggleBookmark }) => {
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const [scrollTop, setScrollTop] = React.useState(0)
  const total = rows.length
  const visibleCount = Math.ceil(height / rowHeight) + 4
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - 2)
  const end = Math.min(total, start + visibleCount)
  const visible = rows.slice(start, end)

  return (
    <div style={{ marginTop: 12, overflow: 'auto', maxHeight: height }} ref={containerRef} onScroll={(e) => setScrollTop((e.target as HTMLDivElement).scrollTop)}>
      <table style={{ width: '100%', borderCollapse: 'collapse', position: 'relative', height: total * rowHeight }}>
        <thead style={{ position: 'sticky', top: 0, background: 'var(--bg-muted)' }}>
          <tr>
            {columns.map((c) => (
              <th key={c} style={{ textAlign: 'left', borderBottom: '1px solid var(--border)', padding: 6 }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr style={{ height: start * rowHeight }}><td colSpan={columns.length} /></tr>
          {visible.map((r) => (
            <tr key={r.idx} style={{ cursor: 'pointer', height: rowHeight }} onClick={() => onRowClick(r)}>
              <td style={{ padding: 6, borderBottom: '1px solid var(--border)' }}>
                <button onClick={(e) => { e.stopPropagation(); onToggleBookmark(r.it.id) }} title="Toggle bookmark">
                  {isBookmarked(r.it.id) ? '★' : '☆'}
                </button>
              </td>
              <td style={{ padding: 6, borderBottom: '1px solid var(--border)' }}>{r.it.id}</td>
              <td style={{ padding: 6, borderBottom: '1px solid var(--border)' }}>{truncate(r.it.user_input)}</td>
              <td style={{ padding: 6, borderBottom: '1px solid var(--border)' }}>{fmt(r.score)}</td>
            </tr>
          ))}
          <tr style={{ height: Math.max(0, (total - end) * rowHeight) }}><td colSpan={columns.length} /></tr>
        </tbody>
      </table>
    </div>
  )
}
