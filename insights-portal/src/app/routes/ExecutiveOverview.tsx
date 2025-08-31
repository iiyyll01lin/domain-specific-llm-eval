import React, { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { RunLoader } from '@/components/RunLoader'
import { usePortalStore } from '@/app/store/usePortalStore'
import { evaluateVerdict } from '@/core/verdict'
import { ThresholdEditor } from '@/components/ThresholdEditor'
import { RunDirectoryPicker } from '@/components/RunDirectoryPicker'
import { getMetricMeta } from '@/core/metrics/registry'
import { FiltersBar } from '@/components/FiltersBar';
import type { FiltersState as UIFilters } from '@/components/filters/chips';
import { usePortalStore as useStore } from '@/app/store/usePortalStore'
import { exportTableToCSV, exportTableToXLSX } from '@/core/exporter'

export default function ExecutiveOverview() {
  const { t } = useTranslation()
  const run = usePortalStore((s) => s.run)
  const thresholds = usePortalStore((s) => s.thresholds)
  const locale = usePortalStore((s) => s.locale)
  const verdict = run ? evaluateVerdict(run.kpis, thresholds) : undefined
  const [sortByGap, setSortByGap] = React.useState(false)
  const filtersFromStore = useStore((s) => s.filters)
  const setFiltersInStore = useStore((s) => s.setFilters)

  const filters: UIFilters = useMemo(() => ({
    language: (filtersFromStore as any).language ?? undefined,
    latencyMs: (filtersFromStore as any).latencyMs,
    metrics: (filtersFromStore as any).metrics,
  }), [filtersFromStore])

  const [derived, setDerived] = React.useState<{ kpis: any; total: number; latencies?: any } | null>(null)
  // Debounce filters to reduce worker churn on rapid slider input
  const [debouncedFilters, setDebouncedFilters] = React.useState(filters)
  React.useEffect(() => {
    const h = setTimeout(() => setDebouncedFilters(filters), 200)
    return () => clearTimeout(h)
  }, [filters])

  React.useEffect(() => {
    if (!run?.items?.length) { setDerived(null); return }
    let canceled = false
    ;(async () => {
      const WM = (await import('@/workers/parser.worker.ts?worker')).default as unknown as { new(): Worker }
      const w = new WM()
      w.onmessage = (ev: MessageEvent<any>) => {
        const msg = ev.data
        if (msg.type === 'aggregated' && !canceled) {
          setDerived({ kpis: msg.kpis, total: msg.total, latencies: msg.latencies })
          w.terminate()
        }
      }
    w.postMessage({ type: 'aggregate', items: run.items, filters: debouncedFilters })
    })()
    return () => { canceled = true }
  }, [run?.items, debouncedFilters])

  const exportKpis = () => {
    const data = Object.entries((derived?.kpis ?? run?.kpis) || {}).map(([k, v]) => ({ metric: k, value: v }))
    exportTableToCSV('overview-kpis.csv', data, {
      runId: 'local-run',
      filters: filters as any,
      thresholds: thresholds as any,
      timestamp: new Date().toISOString(),
    })
  }
  const exportKpisXlsx = async () => {
    const data = Object.entries((derived?.kpis ?? run?.kpis) || {}).map(([k, v]) => ({ metric: k, value: v }))
    await exportTableToXLSX('overview-kpis.xlsx', data, {
      runId: 'local-run',
      filters: filters as any,
      thresholds: thresholds as any,
      timestamp: new Date().toISOString(),
    })
  }

  const availableMetricKeys = useMemo(() => {
    const source = (derived && derived.kpis) || (run ? run.kpis : undefined) || {}
    return Object.keys(source)
  }, [derived, run])

  // Prepare KPI entries with optional sort by threshold gap
  const kpiEntries = useMemo(() => {
    const source = (derived?.kpis ?? run?.kpis) as Record<string, number> | undefined
    if (!source) return [] as Array<[string, number]>
    const entries = Object.entries(source)
    if (!sortByGap) return entries
    const gapOf = (k: string, v: number) => {
      const th = (thresholds as any)[k]
      if (!th) return Number.POSITIVE_INFINITY
      // Negative for below thresholds to bubble up; smaller is worse
      if (v < th.critical) return -(th.critical - v)
      if (v < th.warning) return -(th.warning - v) + 1 // keep below-warning after below-critical
      return (v - th.warning) + 2 // above warning goes to the end
    }
    return entries.slice().sort((a, b) => gapOf(a[0], a[1]) - gapOf(b[0], b[1]))
  }, [derived?.kpis, run?.kpis, sortByGap, thresholds])

  // entries replaced by derived aggregation flow

  return (
    <section>
    <h2>{t('nav.executive')}</h2>
      <RunLoader />
  <RunDirectoryPicker />
      <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
        <button onClick={exportKpis}>Export KPIs (CSV)</button>
        <button onClick={exportKpisXlsx}>Export KPIs (XLSX)</button>
      </div>
      <section style={{ marginBottom: 16 }}>
        <FiltersBar
          metrics={availableMetricKeys}
          filters={filters}
          onChange={(next: UIFilters) => {
            // Map UI filters shape into store shape
            const metricRanges: Record<string, [number | null, number | null]> = {}
            for (const [k, r] of Object.entries(next.metrics || {})) {
              metricRanges[k] = [r.min ?? null, r.max ?? null]
            }
            const latencyRange: [number | null, number | null] | undefined = next.latencyMs
              ? [next.latencyMs.min ?? null, next.latencyMs.max ?? null]
              : undefined
            setFiltersInStore({
              language: (next.language as any) ?? null,
              ...(latencyRange ? { latencyRange } : {}),
              metricRanges,
            } as any)
          }}
          locale={locale}
        />
      </section>
      <div style={{ marginTop: 8 }}>
        <label>
      <input type="checkbox" checked={sortByGap} onChange={(e) => setSortByGap(e.target.checked)} /> {t('overview.sortByGap')}
        </label>
      </div>
    {!run && <p style={{ marginTop: 12 }}>{t('overview.pickHint')}</p>}
  {run && (
        <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
          <div
            style={{
              gridColumn: '1 / -1',
              padding: 12,
              border: '1px solid var(--border)',
              borderRadius: 8,
              background: 'var(--bg-muted)',
              color: 'var(--text)'
            }}
          >
      <strong>Verdict:</strong> {verdict?.verdict ?? '—'}
            {verdict?.failingMetrics?.length ? (
              <span style={{ marginLeft: 8 }}>↓ {verdict.failingMetrics.join(', ')}</span>
            ) : null}
            <span style={{ marginLeft: 16, opacity: 0.8 }}>Items: {derived?.total ?? run.counts.total}</span>
            {(derived?.latencies || run.latencies) && (
              <span style={{ marginLeft: 16, opacity: 0.8 }}>latency p50/p90: {formatNum((derived?.latencies ?? run.latencies)?.p50)} ms / {formatNum((derived?.latencies ?? run.latencies)?.p90)} ms</span>
            )}
          </div>
          {/* Use derived kpis when filtered */}
          {kpiEntries.map((pair) => {
            const k = pair[0]
            const v = pair[1]
            return (
              <div key={k} style={{ padding: 12, border: '1px solid var(--border)', borderRadius: 8, background: 'var(--bg-muted)', color: 'var(--text)' }}>
                <div style={{ fontWeight: 600 }} title={getMetricMeta(k).helpKey ? t(getMetricMeta(k).helpKey as any) : ''}>
                  {t(getMetricMeta(k).labelKey as any)} {getMetricMeta(k).key !== k ? <span className="small-muted">({k})</span> : null}
                </div>
                <div style={{ fontSize: 24 }}>{getMetricMeta(k).format?.(v, locale)}</div>
                {renderGap(k, v ?? NaN, thresholds)}
              </div>
            )
          })}
        </div>
      )}
  <ThresholdEditor />
    </section>
  )
}

// Colors are now driven by CSS variables in theme.css.

function formatNum(v?: number) {
  if (v == null || Number.isNaN(v)) return 'N/A'
  return v.toFixed(3)
}

function renderGap(key: string, v: number, thresholds: any) {
  const th = thresholds[key as keyof typeof thresholds]
  if (!th || Number.isNaN(v)) return null
  if (v < th.critical) return <div style={{ color: 'var(--status-error)' }}>低於 critical {diffPct(v, th.critical)}</div>
  if (v < th.warning) return <div style={{ color: 'var(--status-warn)' }}>低於 warning {diffPct(v, th.warning)}</div>
  return <div style={{ color: 'var(--status-ready)' }}>達標</div>
}

function diffPct(v: number, t: number) {
  const d = (t - v) / t
  return `(${(d * 100).toFixed(1)}%)`
}

// gap helper removed; sorting now implicit in derived map iteration
