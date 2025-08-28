import React from 'react'
import { usePortalStore } from '@/app/store/usePortalStore'
import { applyFilters } from '@/core/analysis/filters'
import { getMetricMeta } from '@/core/metrics/registry'
import { buildRowsWithBookmarks, exportTableToCSV, exportTableToXLSX } from '@/core/exporter'
import { buildFilterChips } from '@/components/filters/chips'

export default function QAFailureExplorer() {
  const { run, filters } = usePortalStore((s) => ({ run: s.run, filters: s.filters }))
  const items = React.useMemo(() => run?.items ?? [], [run?.items])
  const [query, setQuery] = React.useState('')

  // Derive metric keys and selected metric
  const metricKeys = React.useMemo(() => {
    return Array.isArray(items) && items.length > 0 ? Object.keys(items[0].metrics || {}) : []
  }, [items])
  const [selectedMetric, setSelectedMetric] = React.useState<string>('')
  React.useEffect(() => {
    if (!selectedMetric && metricKeys.length > 0) setSelectedMetric(metricKeys[0])
    else if (selectedMetric && !metricKeys.includes(selectedMetric)) setSelectedMetric(metricKeys[0] || '')
  }, [metricKeys, selectedMetric])

  // Apply base filters (from global filters) then query filter by question/user_input
  const filteredItems = React.useMemo(() => {
    const base = applyFilters(items, filters)
    if (!query.trim()) return base
    const q = query.toLowerCase()
    return base.filter((it) => (it.user_input || '').toLowerCase().includes(q))
  }, [items, filters, query])

  // Simple virtualization config
  const rowHeight = 44
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const [scrollTop, setScrollTop] = React.useState(0)
  const onScroll = () => setScrollTop(containerRef.current?.scrollTop || 0)
  const viewportHeight = 440
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - 5)
  const end = Math.min(filteredItems.length, start + Math.ceil(viewportHeight / rowHeight) + 10)

  const [bookmarks, setBookmarks] = React.useState<Set<string>>(new Set())
  const toggleBookmark = (id: string) => {
    setBookmarks((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const exportVisible = () => {
    const rows = buildRowsWithBookmarks(filteredItems, metricKeys, bookmarks)
    exportTableToCSV('qa_view.csv', rows, { timestamp: new Date().toISOString() })
  }

  const exportVisibleXlsx = async () => {
    const rows = buildRowsWithBookmarks(filteredItems, metricKeys, bookmarks)
    await exportTableToXLSX('qa_view.xlsx', rows, { timestamp: new Date().toISOString() })
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
        <label>Metric</label>
        <select value={selectedMetric} onChange={(e) => setSelectedMetric(e.target.value)} aria-label="metric-selector">
          {metricKeys.map((k) => (
            <option value={k} key={k}>
              {getMetricMeta(k as any).key}
            </option>
          ))}
        </select>
        <input
          placeholder="Search question"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ minWidth: 240 }}
          aria-label="qa-search-input"
        />
        <button onClick={exportVisible}>Export CSV</button>
        <button onClick={exportVisibleXlsx}>Export XLSX</button>
      </div>
      {/* Active filter chips */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
        {buildFilterChips(usePortalStore.getState().filters).map((c) => (
          <span key={c.key} style={{ padding: '2px 8px', borderRadius: 12, border: '1px solid var(--border)', background: 'var(--bg-muted)' }}>
            {c.label} <button onClick={c.onClear} aria-label={`clear ${c.key}`} style={{ marginLeft: 6 }}>×</button>
          </span>
        ))}
      </div>
      <div
        ref={containerRef}
        onScroll={onScroll}
        style={{ position: 'relative', height: viewportHeight, overflow: 'auto', border: '1px solid var(--border-color, #333)' }}
      >
        <div style={{ height: filteredItems.length * rowHeight, position: 'relative' }}>
          {filteredItems.slice(start, end).map((it, i) => {
            const top = (start + i) * rowHeight
            const isMarked = bookmarks.has(it.id)
            return (
              <div key={it.id} style={{ position: 'absolute', top, left: 0, right: 0, height: rowHeight, display: 'flex', alignItems: 'center', padding: '0 8px', gap: 8 }}>
                <button
                  onClick={() => toggleBookmark(it.id)}
                  aria-label="toggle-bookmark"
                  title={isMarked ? 'Unbookmark' : 'Bookmark'}
                >
                  {isMarked ? '★' : '☆'}
                </button>
                <div style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {it.user_input || it.id}
                </div>
                <div style={{ width: 120, textAlign: 'right' }}>{selectedMetric ? (it.metrics?.[selectedMetric] ?? '').toString() : ''}</div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// No extra helpers needed here; keep component lean.
