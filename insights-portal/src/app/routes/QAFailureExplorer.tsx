import React from 'react'
import { usePortalStore } from '@/app/store/usePortalStore'
import { applyFilters } from '@/core/analysis/filters'
import { getMetricMeta } from '@/core/metrics/registry'
import { buildRowsWithBookmarks, exportTableToCSV, exportTableToXLSX } from '@/core/exporter'
import { buildFilterChips } from '@/components/filters/chips'
import { loadBookmarks, saveBookmarks, loadVisibleCols, saveVisibleCols, loadVisibleMetrics, saveVisibleMetrics } from '@/core/qa/prefs'
import { TID } from '@/testing/testids'

export default function QAFailureExplorer() {
  const { run, filters, thresholds } = usePortalStore((s) => ({ run: s.run, filters: s.filters, thresholds: s.thresholds }))
  const items = React.useMemo(() => run?.items ?? [], [run?.items])
  const [query, setQuery] = React.useState('')
  const [detailsId, setDetailsId] = React.useState<string | null>(null)
  // image preview map within details
  const [imgPreview, setImgPreview] = React.useState<Record<string, string>>({})

  // Derive metric keys and selected metric
  const metricKeys = React.useMemo(() => (Array.isArray(items) && items.length > 0 ? Object.keys(items[0].metrics || {}) : []), [items])
  const [selectedMetric, setSelectedMetric] = React.useState<string>('')
  React.useEffect(() => {
    if (!selectedMetric && metricKeys.length > 0) setSelectedMetric(metricKeys[0])
    else if (selectedMetric && !metricKeys.includes(selectedMetric)) setSelectedMetric(metricKeys[0] || '')
  }, [metricKeys, selectedMetric])

  // Sort (Faithfulness first; fallback to selectedMetric)
  type SortDir = 'none' | 'asc' | 'desc'
  const [sortDir, setSortDir] = React.useState<SortDir>('none')
  const faithfulnessKey = React.useMemo(
    () => metricKeys.find(k => k.toLowerCase() === 'faithfulness') || '',
    [metricKeys]
  )

  const sortKey = React.useMemo(
    () => (faithfulnessKey || selectedMetric || ''),
    [faithfulnessKey, selectedMetric]
  )

  const sortLabel = React.useMemo(() => {
    if (faithfulnessKey) return 'Faithfulness'
    const meta = getMetricMeta(selectedMetric as any)
    return meta?.key || 'Metric'
  }, [faithfulnessKey, selectedMetric])

  const viewItems = React.useMemo(() => {
    // base filter + search
    const base = (() => {
      const base0 = applyFilters(items, filters)
      if (!query.trim()) return base0
      const q = query.toLowerCase()
      return base0.filter((it) => (it.user_input || '').toLowerCase().includes(q))
    })()
    if (sortDir === 'none' || !sortKey) return base
    // stable sort with index fallback; NaN/undefined push to bottom
    return [...base]
      .map((it, idx) => {
        const vRaw = it?.metrics?.[sortKey]
        const v = typeof vRaw === 'number' ? vRaw : Number(vRaw)
        const score = Number.isFinite(v) ? v : Number.NaN
        return { it, idx, score }
      })
      .sort((a, b) => {
        const aNaN = Number.isNaN(a.score)
        const bNaN = Number.isNaN(b.score)
        if (aNaN && bNaN) return a.idx - b.idx
        if (aNaN) return 1
        if (bNaN) return -1
        const diff = a.score - b.score
        if (diff !== 0) return sortDir === 'asc' ? diff : -diff
        return a.idx - b.idx
      })
      .map(x => x.it)
  }, [items, filters, query, sortDir, sortKey])

  // Filters and search
  // filteredItems -> viewItems (includes sort)
  const filteredItems = viewItems

  // Virtualization
  const rowHeight = 44
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const [scrollTop, setScrollTop] = React.useState(0)
  const onScroll = () => setScrollTop(containerRef.current?.scrollTop || 0)
  const viewportHeight = 440
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - 5)
  const end = Math.min(filteredItems.length, start + Math.ceil(viewportHeight / rowHeight) + 10)

  // Bookmarks
  const [bookmarks, setBookmarks] = React.useState<Set<string>>(() => loadBookmarks())
  React.useEffect(() => { saveBookmarks(bookmarks) }, [bookmarks])
  const toggleBookmark = (id: string) => setBookmarks((prev) => { const next = new Set(prev); next.has(id) ? next.delete(id) : next.add(id); return next })

  // Visible columns
  const baseCols = ['question', 'answer', 'reference'] as const
  const [visibleCols, setVisibleCols] = React.useState<Record<string, boolean>>(() => loadVisibleCols())
  const [visibleMetrics, setVisibleMetrics] = React.useState<Record<string, boolean>>(() => loadVisibleMetrics())
  React.useEffect(() => { saveVisibleCols(visibleCols) }, [visibleCols])
  React.useEffect(() => { saveVisibleMetrics(visibleMetrics) }, [visibleMetrics])
  React.useEffect(() => {
    const next: Record<string, boolean> = {}
    metricKeys.forEach((k) => { next[k] = !!visibleMetrics[k] })
    setVisibleMetrics((prev) => ({ ...next, ...prev }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metricKeys.join(',')])

  const exportVisible = () => {
    const rows = buildRowsWithBookmarks(filteredItems, metricKeys, bookmarks)
    exportTableToCSV('qa_view.csv', rows, { runId: 'local-run', filters: filters as any, thresholds: thresholds as any, timestamp: new Date().toISOString(), branding: { brand: 'Insights Portal', title: 'QA Failure Explorer', footer: 'Generated locally — offline mode' } })
  }
  const exportVisibleXlsx = async () => {
    const rows = buildRowsWithBookmarks(filteredItems, metricKeys, bookmarks)
    await exportTableToXLSX('qa_view.xlsx', rows, { runId: 'local-run', filters: filters as any, thresholds: thresholds as any, timestamp: new Date().toISOString(), branding: { brand: 'Insights Portal', title: 'QA Failure Explorer', footer: 'Generated locally — offline mode' } })
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
        <label>Metric</label>
        <select value={selectedMetric} onChange={(e) => setSelectedMetric(e.target.value)} aria-label="metric-selector">
          {metricKeys.map((k) => (
            <option value={k} key={k}>{getMetricMeta(k as any).key}</option>
          ))}
        </select>
        

        <input placeholder="Search question" value={query} onChange={(e) => setQuery(e.target.value)} style={{ minWidth: 240 }} aria-label="qa-search-input" data-testid={TID.qa.search} />
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          {baseCols.map((c) => (
            <label key={c} style={{ display: 'inline-flex', gap: 4, alignItems: 'center' }}>
              <input type="checkbox" checked={!!visibleCols[c]} onChange={(e) => setVisibleCols((v) => ({ ...v, [c]: e.target.checked }))} />{c}
            </label>
          ))}
          <details>
            <summary>Metrics</summary>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', maxWidth: 560 }}>
              {metricKeys.map((m) => (
                <label key={m} style={{ display: 'inline-flex', gap: 4, alignItems: 'center' }}>
                  <input type="checkbox" checked={!!visibleMetrics[m]} onChange={(e) => setVisibleMetrics((v) => ({ ...v, [m]: e.target.checked }))} />{m}
                </label>
              ))}
            </div>
          </details>
        </div>
        <button onClick={exportVisible} data-testid="qa-export-csv">Export CSV</button>
        <button onClick={exportVisibleXlsx} data-testid="qa-export-xlsx">Export XLSX</button>

         {/* Sort selector */}
        <div
          style={{
            marginLeft: 'auto', 
            display: 'flex',
            alignItems: 'center',
            gap: 8
          }}
        >
          <label>Sort</label>
          <select
            value={sortDir}
            onChange={(e) => setSortDir(e.target.value as SortDir)}
            aria-label="sort-selector"
            data-testid="qa-sort"
          >
            <option value="none">None</option>
            <option value="asc" disabled={!sortKey}>{`${sortLabel} ↑`}</option>
            <option value="desc" disabled={!sortKey}>{`${sortLabel} ↓`}</option>
          </select>
        </div>
      </div>

      {/* Active filter chips */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
        {buildFilterChips(usePortalStore.getState().filters).map((c) => (
          <span key={c.key} style={{ padding: '2px 8px', borderRadius: 12, border: '1px solid var(--border)', background: 'var(--bg-muted)' }}>
            {c.label} <button onClick={c.onClear} aria-label={`clear ${c.key}`} style={{ marginLeft: 6 }}>×</button>
          </span>
        ))}
      </div>

      <div ref={containerRef} onScroll={onScroll} style={{ position: 'relative', height: viewportHeight, overflow: 'auto', border: '1px solid var(--border-color, #333)' }} data-testid={TID.qa.table}>
        <div style={{ height: filteredItems.length * rowHeight, position: 'relative' }}>
          {filteredItems.slice(start, end).map((it, i) => {
            const top = (start + i) * rowHeight
            const isMarked = bookmarks.has(it.id)
            return (
              <div key={it.id} style={{ position: 'absolute', top, left: 0, right: 0, height: rowHeight, display: 'flex', alignItems: 'center', padding: '0 8px', gap: 8 }} data-testid={TID.qa.row(start + i)}>
                <button onClick={() => toggleBookmark(it.id)} aria-label="toggle-bookmark" title={isMarked ? 'Unbookmark' : 'Bookmark'}>{isMarked ? '★' : '☆'}</button>
                <button onClick={() => setDetailsId(it.id)} aria-label="open-details" data-testid={TID.qa.detailsBtn(it.id)}>Details</button>
                {visibleCols.question && (<div style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{it.user_input || it.id}</div>)}
                {visibleCols.answer && (<div style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{it.rag_answer || ''}</div>)}
                {visibleCols.reference && (<div style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{it.reference || ''}</div>)}
                <div style={{ width: 120, textAlign: 'right' }}>{selectedMetric ? (it.metrics?.[selectedMetric] ?? '').toString() : ''}</div>
                {metricKeys.filter((m) => visibleMetrics[m]).map((m) => (<div key={m} style={{ width: 120, textAlign: 'right' }}>{(it.metrics?.[m] ?? '').toString()}</div>))}
              </div>
            )
          })}
        </div>
      </div>

      {/* Row details drawer */}
      {detailsId && (
        <div role="dialog" aria-label="row-details" data-testid={TID.qa.detailsDrawer} style={{ position: 'fixed', top: 0, right: 0, width: '40%', minWidth: 360, bottom: 0, background: 'var(--bg, #111)', color: 'var(--fg, #ddd)', borderLeft: '1px solid var(--border, #333)', padding: 12, overflow: 'auto' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <strong>Row details</strong>
            <button onClick={() => { setDetailsId(null); setImgPreview({}) }} aria-label="close-details" data-testid={TID.qa.detailsClose}>×</button>
          </div>
          {(() => {
            const it = filteredItems.find((x) => x.id === detailsId)
            if (!it) return null
            const contexts = (it.rag_contexts && it.rag_contexts.length ? it.rag_contexts : it.reference_contexts) || []
            const keyText = (it.user_input || '') + ' ' + (it.reference || '')
            const highlight = (txt: string) => {
              const words = keyText.split(/[\s,.;:!?]+/).filter((w) => w.length > 3).slice(0, 10)
              let out = txt
              for (const w of words) {
                try {
                  const re = new RegExp(`(${w.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\\\$&')})`, 'ig')
                  out = out.replace(re, '<mark>$1</mark>')
                } catch { /* ignore */ }
              }
              return out
            }
            return (
              <div style={{ marginTop: 8 }}>
                <div style={{ marginBottom: 8 }}>
                  <div><strong>ID:</strong> {it.id}</div>
                  <div><strong>Question:</strong> {it.user_input || ''}</div>
                  <div><strong>Answer:</strong> {it.rag_answer || ''}</div>
                  <div><strong>Reference:</strong> {it.reference || ''}</div>
                </div>
                <div>
                  <div style={{ fontWeight: 600, marginBottom: 6 }}>Contexts</div>
                  <div style={{ display: 'grid', gap: 6 }}>
                    {contexts.map((c, idx) => (
                      <div key={idx} style={{ padding: 8, border: '1px solid var(--border, #333)', borderRadius: 6, background: 'var(--bg-muted, #1a1a1a)' }}>
                        <div dangerouslySetInnerHTML={{ __html: highlight(c) }} />
                      </div>
                    ))}
                  </div>
                </div>
                {!!it.extra && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontWeight: 600, marginBottom: 6 }}>References</div>
                    {Object.entries(it.extra).map(([k, v]) => {
                      if (typeof v !== 'string') return null
                      const isUrl = /^https?:\/\//i.test(v)
                      const isImg = /\.(png|jpg|jpeg|gif|webp)$/i.test(v)
                      if (!isUrl || isImg) return null
                      const onOpen = () => {
                        const ok = window.confirm(`即將開啟外部連結:\n${v}\n是否繼續？`)
                        if (!ok) return
                        window.open(v, '_blank', 'noopener,noreferrer')
                      }
                      return (
                        <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                          <button onClick={onOpen} aria-label={`open-link-${k}`}>Open</button>
                          <code style={{ fontSize: 12, opacity: 0.9 }}>{v}</code>
                        </div>
                      )
                    })}
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
                      {Object.entries(it.extra).map(([k, v]) => {
                        if (typeof v !== 'string') return null
                        const isImg = /\.(png|jpg|jpeg|gif|webp)$/i.test(v)
                        if (!isImg) return null
                        const loaded = imgPreview[k]
                        const onPreview = () => {
                          const ok = window.confirm(`即將載入外部圖片:\n${v}\n是否繼續？`)
                          if (!ok) return
                          setImgPreview((m) => ({ ...m, [k]: v }))
                        }
                        return (
                          <div key={k} style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 6 }}>
                            {!loaded && (
                              <>
                                <div style={{ width: 160, height: 120, display: 'grid', placeItems: 'center', border: '1px solid #333', borderRadius: 4, background: '#1b1b1b', color: '#aaa', fontSize: 12 }}>No preview</div>
                                <button onClick={onPreview} aria-label={`preview-image-${k}`}>Preview</button>
                                <code style={{ fontSize: 11, opacity: 0.8 }}>{v}</code>
                              </>
                            )}
                            {!!loaded && (
                              <img src={loaded} alt={`image-${k}`} style={{ maxWidth: 160, maxHeight: 120, objectFit: 'cover', background: '#222' }}
                                onError={(e) => {
                                  const el = e.currentTarget as HTMLImageElement
                                  el.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="160" height="120"><rect width="100%" height="100%" fill="#222"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#aaa" font-size="12">No preview</text></svg>`)
                                }} />
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )
          })()}
        </div>
      )}
    </div>
  )
}

// No extra helpers needed here; keep component lean.
