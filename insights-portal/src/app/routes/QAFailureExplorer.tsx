import React from 'react'
import { usePortalStore } from '@/app/store/usePortalStore'
import { applyFilters } from '@/core/analysis/filters'
import { getMetricMeta, metricDirection } from '@/core/metrics/registry'
import { buildRowsWithBookmarks, exportTableToCSV, exportTableToXLSX } from '@/core/exporter'
import { buildFilterChips } from '@/components/filters/chips'
import { loadBookmarks, saveBookmarks, loadVisibleMetrics, saveVisibleMetrics } from '@/core/qa/prefs'
import { TID } from '@/testing/testids'
import { extractEntities, extractAnswerTokens, buildHighlightedContext } from '@/utils/textHighlighter'

// ── Constants ────────────────────────────────────────────────────────────────

const CARD_HEIGHT = 80
const VIEWPORT_HEIGHT = 700 // conservative; overscan handles larger screens

interface SortOption {
  label: string
  key: string
  dir: 'asc' | 'desc'
}

const SORT_OPTIONS: SortOption[] = [
  { label: 'Lowest Sc  (structural_connectivity)', key: 'structural_connectivity', dir: 'asc' },
  { label: 'Lowest Se  (entity_overlap)', key: 'entity_overlap', dir: 'asc' },
  { label: 'Highest Ph (hub_noise_penalty)', key: 'hub_noise_penalty', dir: 'desc' },
  { label: 'Lowest Faithfulness', key: 'Faithfulness', dir: 'asc' },
  { label: 'Lowest AnswerRelevancy', key: 'AnswerRelevancy', dir: 'asc' },
  { label: 'Lowest ContextPrecision', key: 'ContextPrecision', dir: 'asc' },
  { label: 'Lowest ContextRecall', key: 'ContextRecall', dir: 'asc' },
  { label: 'Bookmarked first', key: '__bookmarked', dir: 'desc' },
]

// ── URL param helpers ─────────────────────────────────────────────────────────

function readUrlParam(key: string): string | null {
  try {
    return new URLSearchParams(window.location.search).get(key)
  } catch {
    return null
  }
}

function setUrlParam(key: string, value: string | null): void {
  try {
    const params = new URLSearchParams(window.location.search)
    if (value !== null) params.set(key, value)
    else params.delete(key)
    const search = params.toString()
    const newUrl = search
      ? `${window.location.pathname}?${search}`
      : window.location.pathname
    window.history.replaceState(null, '', newUrl)
  } catch {
    // ignore — history API not available (e.g., in tests)
  }
}

// ── Score badge ───────────────────────────────────────────────────────────────

function ScoreBadge({
  metricKey,
  value,
  thresholds,
}: {
  metricKey: string
  value: number | null | undefined
  thresholds: Record<string, { warning: number; critical: number }>
}) {
  if (value == null) return null
  if (isNaN(value)) return null
  const th = thresholds[metricKey]
  const dir = metricDirection(metricKey as any)
  let cls = 'badge badge--ok'
  if (th) {
    const isCrit =
      dir === 'lower' ? value > th.critical : value < th.critical
    const isWarn =
      !isCrit && (dir === 'lower' ? value > th.warning : value < th.warning)
    if (isCrit) cls = 'badge badge--error'
    else if (isWarn) cls = 'badge badge--warn'
  }
  const label = metricKey
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
  return (
    <span className={cls} title={`${metricKey}: ${value.toFixed(3)}`}>
      {label} {value.toFixed(2)}
    </span>
  )
}

// ── Context accordion item ────────────────────────────────────────────────────

const ContextItem = React.memo(function ContextItem({
  idx,
  text,
  entities,
  answerTokens,
  defaultOpen,
}: {
  idx: number
  text: string
  entities: string[]
  answerTokens: string[]
  defaultOpen?: boolean
}) {
  const { html, tier, matchCount } = React.useMemo(
    () => buildHighlightedContext(text, entities, answerTokens),
    [text, entities, answerTokens],
  )

  return (
    <details className="qa-ctx-item" open={defaultOpen}>
      <summary className="qa-ctx-summary">
        <span>Context {idx + 1}</span>
        {matchCount > 0 && (
          <span
            className={`badge ${tier === 'entity' ? 'badge--graph' : 'badge--ok'}`}
            style={{ marginLeft: 8 }}
          >
            {matchCount} {tier === 'entity' ? 'entities' : 'overlaps'}
          </span>
        )}
      </summary>
      {/* dangerouslySetInnerHTML is intentional: text/entities are from the user's own
          locally-loaded evaluation files; the injected markup is a fixed <mark class="..."> pattern. */}
      <div className="qa-ctx-body" dangerouslySetInnerHTML={{ __html: html }} />
    </details>
  )
})

// ── Main component ────────────────────────────────────────────────────────────

export default function QAFailureExplorer() {
  const { run, filters, thresholds } = usePortalStore((s) => ({
    run: s.run,
    filters: s.filters,
    thresholds: s.thresholds,
  }))
  const items = React.useMemo(() => run?.items ?? [], [run?.items])

  // ── Sort state (URL-persisted) ──────────────────────────────────────────────
  const [sortOptionIdx, setSortOptionIdx] = React.useState<number>(() => {
    const sk = readUrlParam('debugSort')
    const sd = readUrlParam('debugDir')
    if (sk) {
      const exactIdx = SORT_OPTIONS.findIndex((o) => o.key === sk && o.dir === sd)
      if (exactIdx >= 0) return exactIdx
      const keyIdx = SORT_OPTIONS.findIndex((o) => o.key === sk)
      if (keyIdx >= 0) return keyIdx
    }
    return 0
  })

  const sortOption = SORT_OPTIONS[sortOptionIdx] ?? SORT_OPTIONS[0]

  React.useEffect(() => {
    setUrlParam('debugSort', sortOption.key)
    setUrlParam('debugDir', sortOption.dir)
  }, [sortOption.key, sortOption.dir])

  // Listen for "Inspect worst ↗" deep-links dispatched by ExecutiveOverview
  React.useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<{ metric: string; dir: 'asc' | 'desc' }>).detail
      const idx = SORT_OPTIONS.findIndex(
        (o) => o.key === detail.metric && o.dir === detail.dir,
      )
      if (idx >= 0) setSortOptionIdx(idx)
    }
    window.addEventListener('portal:debugger:sort', handler)
    return () => window.removeEventListener('portal:debugger:sort', handler)
  }, [])

  // ── Selected item (URL-persisted) ──────────────────────────────────────────
  const [selectedId, setSelectedId] = React.useState<string | null>(() =>
    readUrlParam('debugId'),
  )

  const selectItem = React.useCallback((id: string | null) => {
    setSelectedId(id)
    setUrlParam('debugId', id)
  }, [])

  // ── Bookmarks ──────────────────────────────────────────────────────────────
  const [bookmarks, setBookmarks] = React.useState<Set<string>>(() => loadBookmarks())
  React.useEffect(() => { saveBookmarks(bookmarks) }, [bookmarks])

  const toggleBookmark = React.useCallback((id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setBookmarks((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }, [])

  // ── Search ─────────────────────────────────────────────────────────────────
  const [query, setQuery] = React.useState('')

  // ── Metric keys ────────────────────────────────────────────────────────────
  const metricKeys = React.useMemo(
    () => (items.length > 0 ? Object.keys(items[0].metrics ?? {}) : []),
    [items],
  )

  const GCR_PILLS = React.useMemo(
    () =>
      ['entity_overlap', 'structural_connectivity', 'hub_noise_penalty'].filter((k) =>
        metricKeys.includes(k),
      ),
    [metricKeys],
  )

  // ── Visible metrics prefs ──────────────────────────────────────────────────
  const [visibleMetrics, setVisibleMetrics] = React.useState<Record<string, boolean>>(
    () => loadVisibleMetrics(),
  )
  React.useEffect(() => { saveVisibleMetrics(visibleMetrics) }, [visibleMetrics])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  React.useEffect(() => {
    setVisibleMetrics((prev) => {
      const next: Record<string, boolean> = {}
      metricKeys.forEach((k) => { next[k] = prev[k] ?? false })
      return { ...next, ...prev }
    })
  }, [metricKeys.join(',')])  // eslint-disable-line react-hooks/exhaustive-deps

  // ── Filter + sort pipeline ─────────────────────────────────────────────────
  const sortedItems = React.useMemo(() => {
    const base = applyFilters(items, filters)
    const filtered = query.trim()
      ? base.filter((it) =>
          (it.user_input ?? '').toLowerCase().includes(query.toLowerCase()),
        )
      : base

    const { key, dir } = sortOption
    if (key === '__bookmarked') {
      return filtered.slice().sort((a, b) => {
        const aB = bookmarks.has(a.id) ? 1 : 0
        const bB = bookmarks.has(b.id) ? 1 : 0
        return bB - aB
      })
    }
    return filtered.slice().sort((a, b) => {
      const aV = (a.metrics as Record<string, number | null>)[key] ?? null
      const bV = (b.metrics as Record<string, number | null>)[key] ?? null
      // push nulls to end regardless of direction
      if (aV === null && bV === null) return 0
      if (aV === null) return 1
      if (bV === null) return -1
      return dir === 'asc' ? aV - bV : bV - aV
    })
  }, [items, filters, query, sortOption, bookmarks])

  // ── Virtualization (master list) ───────────────────────────────────────────
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const [scrollTop, setScrollTop] = React.useState(0)
  const onScroll = React.useCallback(() => {
    setScrollTop(containerRef.current?.scrollTop ?? 0)
  }, [])
  const OVERSCAN = 4
  const vStart = Math.max(0, Math.floor(scrollTop / CARD_HEIGHT) - OVERSCAN)
  const vEnd = Math.min(
    sortedItems.length,
    vStart + Math.ceil(VIEWPORT_HEIGHT / CARD_HEIGHT) + OVERSCAN * 2,
  )

  // Primary pill key = active sort metric (skip pseudo-keys)
  const primaryPill = sortOption.key !== '__bookmarked' ? sortOption.key : null

  // ── Effective selection (auto-select first item when nothing chosen) ────────
  const effectiveId = selectedId ?? sortedItems[0]?.id ?? null

  const selectedItem = React.useMemo(
    () => (effectiveId ? (sortedItems.find((x) => x.id === effectiveId) ?? null) : null),
    [sortedItems, effectiveId],
  )

  const contexts = React.useMemo(
    () =>
      (selectedItem?.rag_contexts?.length
        ? selectedItem.rag_contexts
        : selectedItem?.reference_contexts) ?? [],
    [selectedItem],
  )

  const entities = React.useMemo(
    () => (selectedItem ? extractEntities(selectedItem.extra) : []),
    [selectedItem],
  )

  const answerTokens = React.useMemo(
    () => (selectedItem ? extractAnswerTokens(selectedItem.rag_answer) : []),
    [selectedItem],
  )

  // ── Image preview (preserved for backward compat) ──────────────────────────
  const [imgPreview, setImgPreview] = React.useState<Record<string, string>>({})

  // ── Export helpers ─────────────────────────────────────────────────────────
  const exportCsv = React.useCallback(() => {
    const rows = buildRowsWithBookmarks(sortedItems, metricKeys, bookmarks)
    exportTableToCSV('qa_debugger.csv', rows, {
      runId: run?.id ?? 'local-run',
      filters: filters as any,
      thresholds: thresholds as any,
      timestamp: new Date().toISOString(),
      branding: {
        brand: 'Insights Portal',
        title: 'QA Debugger',
        footer: 'Generated locally — offline mode',
      },
    })
  }, [sortedItems, metricKeys, bookmarks, run?.id, filters, thresholds])

  const exportXlsx = React.useCallback(async () => {
    const rows = buildRowsWithBookmarks(sortedItems, metricKeys, bookmarks)
    await exportTableToXLSX('qa_debugger.xlsx', rows, {
      runId: run?.id ?? 'local-run',
      filters: filters as any,
      thresholds: thresholds as any,
      timestamp: new Date().toISOString(),
      branding: {
        brand: 'Insights Portal',
        title: 'QA Debugger',
        footer: 'Generated locally — offline mode',
      },
    })
  }, [sortedItems, metricKeys, bookmarks, run?.id, filters, thresholds])

  // ── Render ─────────────────────────────────────────────────────────────────

  if (!run) {
    return (
      <div style={{ padding: 24, color: 'var(--text-muted)', fontSize: 'var(--text-base)' }}>
        No evaluation run loaded. Return to Executive Overview and load a JSON or CSV file.
      </div>
    )
  }

  return (
    <div>
      {/* ── Toolbar ─────────────────────────────────────────────────────────── */}
      <div className="qa-debugger-toolbar">
        <span className="qa-debugger-title">QA Debugger</span>
        <select
          value={sortOptionIdx}
          onChange={(e) => setSortOptionIdx(Number(e.target.value))}
          aria-label="sort-selector"
          style={{ minWidth: 290 }}
        >
          {SORT_OPTIONS.map((opt, i) => (
            <option key={`${opt.key}-${opt.dir}`} value={i}>
              {opt.label}
            </option>
          ))}
        </select>
        <input
          placeholder="Search question…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ minWidth: 220 }}
          aria-label="qa-search-input"
          data-testid={TID.qa.search}
        />
        <span style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)', whiteSpace: 'nowrap' }}>
          {sortedItems.length} items
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          <button onClick={exportCsv} data-testid="qa-export-csv">
            Export CSV
          </button>
          <button onClick={exportXlsx} data-testid="qa-export-xlsx">
            Export XLSX
          </button>
        </div>
      </div>

      {/* Active filter chips */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 6, marginBottom: 10 }}>
        {buildFilterChips(usePortalStore.getState().filters).map((c) => (
          <span key={c.key} className="qa-filter-chip">
            {c.label}
            <button
              onClick={c.onClear}
              aria-label={`clear ${c.key}`}
              style={{ marginLeft: 6, background: 'none', border: 'none', padding: 0, cursor: 'pointer', color: 'inherit' }}
            >
              ×
            </button>
          </span>
        ))}
      </div>

      {/* ── Master-Detail Layout ─────────────────────────────────────────────── */}
      <div className="qa-debugger-layout">

        {/* ── MASTER LIST ─────────────────────────────────────────────────── */}
        <div className="qa-master-pane">
          {sortedItems.length === 0 ? (
            <div
              style={{
                display: 'grid',
                placeItems: 'center',
                height: '100%',
                color: 'var(--text-muted)',
                fontSize: 'var(--text-sm)',
              }}
            >
              No items match the current filter or search.
            </div>
          ) : (
            <div
              ref={containerRef}
              onScroll={onScroll}
              className="qa-master-scroll"
              data-testid={TID.qa.table}
            >
              <div style={{ height: sortedItems.length * CARD_HEIGHT, position: 'relative' }}>
                {sortedItems.slice(vStart, vEnd).map((it, i) => {
                  const cardIdx = vStart + i
                  const isSelected = it.id === effectiveId
                  const isBookmarked = bookmarks.has(it.id)

                  // Pills: primary sort metric + up to 2 GCR sub-metrics (deduped)
                  const pillKeys = Array.from(
                    new Set([
                      ...(primaryPill && (it.metrics as Record<string, unknown>)[primaryPill] != null
                        ? [primaryPill]
                        : []),
                      ...GCR_PILLS.filter(
                        (k) => (it.metrics as Record<string, unknown>)[k] != null,
                      ),
                    ]),
                  ).slice(0, 3)

                  return (
                    <div
                      key={it.id}
                      className={`qa-master-card${isSelected ? ' qa-master-card--selected' : ''}`}
                      style={{
                        position: 'absolute',
                        top: cardIdx * CARD_HEIGHT,
                        left: 0,
                        right: 0,
                        height: CARD_HEIGHT,
                      }}
                      onClick={() => selectItem(it.id)}
                      data-testid={TID.qa.row(cardIdx)}
                      role="button"
                      tabIndex={0}
                      aria-selected={isSelected}
                      onKeyDown={(e) => e.key === 'Enter' && selectItem(it.id)}
                    >
                      <div className="qa-master-card-header">
                        <span className="qa-master-card-index">#{cardIdx + 1}</span>
                        <span className="qa-master-card-question">
                          {it.user_input || it.id}
                        </span>
                        <button
                          onClick={(e) => toggleBookmark(it.id, e)}
                          aria-label="toggle-bookmark"
                          title={isBookmarked ? 'Remove bookmark' : 'Bookmark'}
                          className="qa-bookmark-btn"
                          data-testid={TID.qa.detailsBtn(it.id)}
                        >
                          {isBookmarked ? '★' : '☆'}
                        </button>
                      </div>
                      <div className="qa-master-card-pills">
                        {pillKeys.map((k) => (
                          <ScoreBadge
                            key={k}
                            metricKey={k}
                            value={(it.metrics as Record<string, number | null>)[k]}
                            thresholds={thresholds as any}
                          />
                        ))}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* ── DETAIL PANEL ────────────────────────────────────────────────── */}
        <div
          className="qa-detail-pane"
          role="region"
          aria-label="row-details"
          data-testid={TID.qa.detailsDrawer}
        >
          {!selectedItem ? (
            <div
              style={{
                display: 'grid',
                placeItems: 'center',
                height: '100%',
                color: 'var(--text-muted)',
                fontSize: 'var(--text-sm)',
              }}
            >
              Select a question from the list to inspect it.
            </div>
          ) : (
            <div className="qa-detail-inner">

              {/* ── Zone 1: Question ──────────────────────────────────────── */}
              <div className="qa-detail-zone qa-detail-zone--question">
                <div className="qa-detail-zone-label">
                  Question
                  <span style={{ marginLeft: 'auto', fontWeight: 400, textTransform: 'none', letterSpacing: 0, color: 'var(--text-subtle)' }}>
                    id: {selectedItem.id}
                  </span>
                </div>
                <p className="qa-detail-question-text">
                  {selectedItem.user_input || selectedItem.id}
                </p>
                {selectedItem.reference && (
                  <div className="qa-detail-reference">
                    <strong>Reference answer: </strong>
                    {selectedItem.reference}
                  </div>
                )}
              </div>

              {/* ── Zone 2: LLM Answer ───────────────────────────────────── */}
              <div className="qa-detail-zone qa-detail-zone--answer">
                <div className="qa-detail-zone-label">LLM Answer</div>
                <p className="qa-detail-answer-text">
                  {selectedItem.rag_answer || '—'}
                </p>
              </div>

              {/* ── Metrics pills row ─────────────────────────────────────── */}
              <div className="qa-detail-metrics">
                {metricKeys.map((k) => {
                  const v = (selectedItem.metrics as Record<string, number | null>)[k]
                  return (
                    <ScoreBadge
                      key={k}
                      metricKey={k}
                      value={v}
                      thresholds={thresholds as any}
                    />
                  )
                })}
              </div>

              {/* ── Zone 3: Retrieved Contexts ───────────────────────────── */}
              <div className="qa-detail-zone qa-detail-zone--contexts">
                <div className="qa-detail-zone-label">
                  Retrieved Contexts
                  {entities.length > 0 && (
                    <span className="badge badge--graph" style={{ marginLeft: 8 }}>
                      Tier 1 · {entities.length} entities
                    </span>
                  )}
                  {entities.length === 0 && answerTokens.length > 0 && (
                    <span className="badge badge--ok" style={{ marginLeft: 8 }}>
                      Tier 2 · word overlap
                    </span>
                  )}
                </div>
                {contexts.length === 0 ? (
                  <div
                    style={{
                      color: 'var(--text-muted)',
                      fontSize: 'var(--text-sm)',
                      padding: '8px 0',
                    }}
                  >
                    No contexts available for this item.
                  </div>
                ) : (
                  <div className="qa-ctx-list">
                    {contexts.map((ctx, idx) => (
                      <ContextItem
                        key={idx}
                        idx={idx}
                        text={ctx}
                        entities={entities}
                        answerTokens={answerTokens}
                        defaultOpen={idx === 0}
                      />
                    ))}
                  </div>
                )}
              </div>

              {/* ── External references / image previews (preserved) ─────── */}
              {!!selectedItem.extra && (
                <div className="qa-detail-zone">
                  <div className="qa-detail-zone-label">References</div>
                  {Object.entries(selectedItem.extra).map(([k, v]) => {
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
                      <div
                        key={k}
                        style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}
                      >
                        <button onClick={onOpen} aria-label={`open-link-${k}`}>
                          Open
                        </button>
                        <code style={{ fontSize: 12, opacity: 0.9 }}>{v}</code>
                      </div>
                    )
                  })}
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
                    {Object.entries(selectedItem.extra).map(([k, v]) => {
                      if (typeof v !== 'string') return null
                      if (!/\.(png|jpg|jpeg|gif|webp)$/i.test(v)) return null
                      const loaded = imgPreview[k]
                      const onPreview = () => {
                        const ok = window.confirm(`即將載入外部圖片:\n${v}\n是否繼續？`)
                        if (!ok) return
                        setImgPreview((m) => ({ ...m, [k]: v }))
                      }
                      return (
                        <div
                          key={k}
                          style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'flex-start',
                            gap: 6,
                          }}
                        >
                          {!loaded ? (
                            <>
                              <div
                                style={{
                                  width: 160,
                                  height: 120,
                                  display: 'grid',
                                  placeItems: 'center',
                                  border: '1px solid var(--border)',
                                  borderRadius: 4,
                                  background: 'var(--bg-muted)',
                                  color: 'var(--text-muted)',
                                  fontSize: 12,
                                }}
                              >
                                No preview
                              </div>
                              <button onClick={onPreview} aria-label={`preview-image-${k}`}>
                                Preview
                              </button>
                              <code style={{ fontSize: 11, opacity: 0.8 }}>{v}</code>
                            </>
                          ) : (
                            <img
                              src={loaded}
                              alt={`image-${k}`}
                              style={{
                                maxWidth: 160,
                                maxHeight: 120,
                                objectFit: 'cover',
                                background: 'var(--bg-muted)',
                              }}
                              onError={(e) => {
                                const el = e.currentTarget as HTMLImageElement
                                el.src =
                                  'data:image/svg+xml;utf8,' +
                                  encodeURIComponent(
                                    `<svg xmlns="http://www.w3.org/2000/svg" width="160" height="120"><rect width="100%" height="100%" fill="#222"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#aaa" font-size="12">No preview</text></svg>`,
                                  )
                              }}
                            />
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Deselect */}
              <div style={{ padding: '12px 16px', display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  onClick={() => selectItem(null)}
                  aria-label="close-details"
                  data-testid={TID.qa.detailsClose}
                >
                  Deselect
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
