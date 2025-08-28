import React from 'react'
import { useTranslation } from 'react-i18next'
import { RunLoader } from '@/components/RunLoader'
import { usePortalStore } from '@/app/store/usePortalStore'
import { evaluateVerdict } from '@/core/verdict'
import { ThresholdEditor } from '@/components/ThresholdEditor'
import { RunDirectoryPicker } from '@/components/RunDirectoryPicker'
import { getMetricMeta } from '@/core/metrics/registry'
import { usePortalStore as useStore } from '@/app/store/usePortalStore'
import { exportTableToCSV } from '@/core/exporter'
import { FiltersBar } from '@/components/FiltersBar'

export const ExecutiveOverview: React.FC = () => {
  const { t } = useTranslation()
  const run = usePortalStore((s) => s.run)
  const thresholds = usePortalStore((s) => s.thresholds)
  const locale = usePortalStore((s) => s.locale)
  const verdict = run ? evaluateVerdict(run.kpis, thresholds) : undefined
  const [sortByGap, setSortByGap] = React.useState(false)
  const filters = useStore((s) => s.filters)
  const [derived, setDerived] = React.useState<{ kpis: any; total: number; latencies?: any } | null>(null)

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
      w.postMessage({ type: 'aggregate', items: run.items, filters })
    })()
    return () => { canceled = true }
  }, [run?.items, filters])

  const exportKpis = () => {
    const data = Object.entries((derived?.kpis ?? run?.kpis) || {}).map(([k, v]) => ({ metric: k, value: v }))
    exportTableToCSV('overview-kpis.csv', data, {
      runId: 'local-run',
      filters: filters as any,
      thresholds: thresholds as any,
      timestamp: new Date().toISOString(),
    })
  }

  // entries replaced by derived aggregation flow

  return (
    <section>
    <h2>{t('nav.executive')}</h2>
      <RunLoader />
  <RunDirectoryPicker />
      <div style={{ marginTop: 8 }}>
        <button onClick={exportKpis}>Export KPIs (CSV)</button>
      </div>
  <FiltersBar run={run || undefined} />
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
          {Object.entries((derived?.kpis ?? run.kpis) as Record<string, number>).map((pair) => {
            const k = Array.isArray(pair) ? pair[0] : String(pair)
            const v = Array.isArray(pair) ? pair[1] : NaN
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
