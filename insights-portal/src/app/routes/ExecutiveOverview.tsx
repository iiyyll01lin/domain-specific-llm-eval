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
import DevTelemetryPanel from '@/components/DevTelemetryPanel'
import type { Thresholds } from '@/core/types'

function KpiInfoPopover(props: { metricKey: string; value: number | undefined; runId?: string; total?: number; latencies?: { p50?: number|null, p90?: number|null }; sources?: string[]; filters?: any; thresholds?: any; locale?: string }) {
  const { t } = useTranslation()
  const [open, setOpen] = React.useState(false)
  const fmt = (v: any) => (v == null || Number.isNaN(v) ? 'N/A' : typeof v === 'number' ? v.toFixed(3) : String(v))
  return (
    <span style={{ marginLeft: 6, position: 'relative', display: 'inline-block' }}>
      <button aria-label={`info-${props.metricKey}`} onClick={() => setOpen((o) => !o)} style={{ fontSize: 12 }}>i</button>
      {open && (
        <div role="dialog" aria-label="kpi-info" style={{ position: 'absolute', zIndex: 10, top: '120%', right: 0, minWidth: 260, maxWidth: 360, background: 'var(--bg, #111)', color: 'var(--text, #eee)', border: '1px solid var(--border, #333)', borderRadius: 8, padding: 8, boxShadow: '0 2px 12px rgba(0,0,0,0.4)' }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>{t(getMetricMeta(props.metricKey as any).labelKey as any)}</div>
          <div style={{ fontSize: 12, opacity: 0.9, marginBottom: 6 }}>{t(getMetricMeta(props.metricKey as any).helpKey as any)}</div>
          <div style={{ fontSize: 12 }}>
            <div>n: {props.total ?? '—'}</div>
            {props.latencies && (
              <div>latency p50/p90: {fmt(props.latencies.p50)} ms / {fmt(props.latencies.p90)} ms</div>
            )}
            {props.sources?.length ? (
              <div title={props.sources.join('\n')}>sources: {props.sources.map((s, i) => <span key={i}><code>{s}</code>{i < props.sources!.length - 1 ? ', ' : ''}</span>)}</div>
            ) : null}
            {props.filters ? <div>filters: <code style={{ fontSize: 10 }}>{JSON.stringify(props.filters)}</code></div> : null}
            {props.thresholds ? <div>thresholds: <code style={{ fontSize: 10 }}>{JSON.stringify(props.thresholds)}</code></div> : null}
          </div>
          <div style={{ textAlign: 'right', marginTop: 6 }}>
            <button onClick={() => setOpen(false)}>Close</button>
          </div>
        </div>
      )}
    </span>
  )
}

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
  const [timings, setTimings] = React.useState<Array<{ at: number; filterMs: number; sampleMs: number; aggregateMs: number; total: number }>>([])
  const [coalesceMs, setCoalesceMs] = React.useState<number>(100)
  const [bench, setBench] = React.useState<Array<{ size: number; samplePct: number | null; coalesceMs: number; filterMs: number; sampleMs: number; aggregateMs: number }>>([])
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
      // push worker runtime config
      w.postMessage({ type: 'config', coalesceMs })
  w.onmessage = (ev: MessageEvent<any>) => {
        const msg = ev.data
        if (msg.type === 'aggregated' && !canceled) {
          setDerived({ kpis: msg.kpis, total: msg.total, latencies: msg.latencies })
          if (msg.timings) {
            setTimings((arr) => arr.concat([{ at: Date.now(), filterMs: msg.timings.filterMs, sampleMs: msg.timings.sampleMs, aggregateMs: msg.timings.aggregateMs, total: msg.total }]).slice(-200))
          }
          w.terminate()
        }
      }
  // For very large datasets, send a sampling hint; keep full-count in 'total'
  const sampleHint = run.items.length > 20000 ? { pct: 0.25, method: 'random' as const } : undefined
  w.postMessage({ type: 'aggregate', items: run.items, filters: debouncedFilters, sample: sampleHint })
    })()
    return () => { canceled = true }
  }, [run?.items, debouncedFilters, coalesceMs])

  const runBenchmarks = React.useCallback(async () => {
    if (!run?.items?.length) return
    const base = run.items
    const makeSize = (n: number) => {
      if (base.length >= n) return base.slice(0, n)
      const arr: typeof base = []
      while (arr.length < n) {
        const need = Math.min(base.length, n - arr.length)
        arr.push(...base.slice(0, need))
      }
      return arr
    }
    const sizes = [5000, 20000, 100000]
    const coalesces = [0, 50, coalesceMs]
    const samples: Array<number | null> = [null, 0.25]
    const results: typeof bench = []
    for (const size of sizes) {
      const items = makeSize(size)
      for (const c of coalesces) {
        for (const pct of samples) {
          const WM = (await import('@/workers/parser.worker.ts?worker')).default as unknown as { new(): Worker }
          const w = new WM()
          w.postMessage({ type: 'config', coalesceMs: c })
          const res = await new Promise<any>((resolve) => {
            w.onmessage = (ev: MessageEvent<any>) => {
              if (ev.data?.type === 'aggregated') { resolve(ev.data); w.terminate() }
            }
            w.postMessage({ type: 'aggregate', items, filters: debouncedFilters, sample: pct == null ? undefined : { pct, method: 'random' as const } })
          })
          const t = res?.timings || { filterMs: 0, sampleMs: 0, aggregateMs: 0 }
          results.push({ size, samplePct: pct, coalesceMs: c, filterMs: t.filterMs, sampleMs: t.sampleMs, aggregateMs: t.aggregateMs })
        }
      }
    }
    setBench(results)
  }, [run?.items, debouncedFilters, coalesceMs])

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

  // Simple Session Save/Load (schemaVersion: 1)
  const saveSession = async () => {
    if (!run) return
    const state = useStore.getState()
    const session = {
      schemaVersion: 1,
      runId: run.id || 'default',
      thresholds: state.thresholds as Thresholds,
      filters: state.filters,
      locale: state.locale,
      persona: undefined,
    }
    const blob = new Blob([JSON.stringify(session, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `session_${session.runId}.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }
  const loadSession = async () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'application/json,.json'
    const file: File = await new Promise((resolve, reject) => {
      input.onchange = () => {
        const f = input.files?.[0]
        if (f) resolve(f)
        else reject(new Error('未選取檔案'))
      }
      input.click()
    })
    const text = await file.text()
    const data = JSON.parse(text)
    if (!data || (data.schemaVersion !== 1)) throw new Error('Unsupported session schemaVersion')
    const setLocale = useStore.getState().setLocale
    const setThresholds = useStore.getState().setThresholds
    const setFilters = useStore.getState().setFilters
    if (data.locale) setLocale(data.locale)
    if (data.thresholds) setThresholds(data.thresholds)
    if (data.filters) setFilters(data.filters)
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

  const runId = run?.id || 'default'
  const panelMap = useStore((s) => s.overviewPanels)
  const setPanelExpanded = useStore((s) => s.setPanelExpanded)

  return (
    <section>
    <h2>{t('nav.executive')}</h2>
      <RunLoader />
  <RunDirectoryPicker />
      {/* Dev-only timing panel; hidden in production if process.env.NODE_ENV === 'production' */}
      {import.meta.env.DEV && (
        <div style={{ marginTop: 8 }}>
          <DevTelemetryPanel
            samples={timings}
            coalesceMs={coalesceMs}
            onCoalesceChange={setCoalesceMs}
            bench={bench}
            onRunBenchmarks={runBenchmarks}
          />
        </div>
      )}
      <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
        <button onClick={exportKpis}>Export KPIs (CSV)</button>
        <button onClick={exportKpisXlsx}>Export KPIs (XLSX)</button>
  <button onClick={saveSession}>Save Session</button>
  <button onClick={loadSession}>Load Session</button>
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
          <details open={(panelMap?.[runId]?.['verdict'] ?? true)} onToggle={(e) => setPanelExpanded(runId, 'verdict', (e.target as HTMLDetailsElement).open)} style={{ gridColumn: '1 / -1' }}>
            <summary style={{ cursor: 'pointer', userSelect: 'none', padding: 8, border: '1px solid var(--border)', borderRadius: 8, background: 'var(--bg-muted)', color: 'var(--text)' }}>Verdict</summary>
            <div style={{ padding: 12, border: '1px solid var(--border)', borderRadius: 8, borderTopLeftRadius: 0, borderTopRightRadius: 0, borderTop: 'none', background: 'var(--bg-muted)', color: 'var(--text)' }}>
              <strong>Verdict:</strong> {verdict?.verdict ?? '—'}
              {verdict?.failingMetrics?.length ? (
                <span style={{ marginLeft: 8 }}>↓ {verdict.failingMetrics.join(', ')}</span>
              ) : null}
              <span style={{ marginLeft: 16, opacity: 0.8 }}>Items: {derived?.total ?? run.counts.total}</span>
              {(derived?.latencies || run.latencies) && (
                <span style={{ marginLeft: 16, opacity: 0.8 }}>latency p50/p90: {formatNum((derived?.latencies ?? run.latencies)?.p50)} ms / {formatNum((derived?.latencies ?? run.latencies)?.p90)} ms</span>
              )}
            </div>
          </details>
          {/* Use derived kpis when filtered */}
          {kpiEntries.map((pair) => {
            const k = pair[0]
            const v = pair[1]
            return (
              <div key={k} style={{ padding: 12, border: '1px solid var(--border)', borderRadius: 8, background: 'var(--bg-muted)', color: 'var(--text)' }}>
                <div style={{ fontWeight: 600 }} title={getMetricMeta(k).helpKey ? t(getMetricMeta(k).helpKey as any) : ''}>
                  {t(getMetricMeta(k).labelKey as any)} {getMetricMeta(k).key !== k ? <span className="small-muted">({k})</span> : null}
                  <KpiInfoPopover
                    metricKey={k}
                    value={v}
                    runId={run.id}
                    total={derived?.total ?? run.counts.total}
                    latencies={derived?.latencies ?? run.latencies}
                    sources={[run.artifacts?.summaryJson?.name || 'summary.json', run.artifacts?.configYaml?.name || 'config.yaml'].filter(Boolean) as string[]}
                    filters={filters}
                    thresholds={thresholds}
                    locale={locale}
                  />
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
