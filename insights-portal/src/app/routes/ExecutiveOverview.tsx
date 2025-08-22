import React from 'react'
import { useTranslation } from 'react-i18next'
import { RunLoader } from '@/components/RunLoader'
import { usePortalStore } from '@/app/store/usePortalStore'
import { evaluateVerdict } from '@/core/verdict'
import { ThresholdEditor } from '@/components/ThresholdEditor'
import { RunDirectoryPicker } from '@/components/RunDirectoryPicker'
import { getMetricMeta } from '@/core/metrics/registry'

export const ExecutiveOverview: React.FC = () => {
  const { t } = useTranslation()
  const run = usePortalStore((s) => s.run)
  const thresholds = usePortalStore((s) => s.thresholds)
  const locale = usePortalStore((s) => s.locale)
  const verdict = run ? evaluateVerdict(run.kpis, thresholds) : undefined
  const [sortByGap, setSortByGap] = React.useState(false)

  const entries = React.useMemo(() => {
    if (!run) return [] as Array<[string, number]>
    const arr = Object.entries(run.kpis as Record<string, number>)
    if (!sortByGap) return arr
    // Sort by largest gap vs warning threshold (or critical if both present and value < critical)
    return arr.sort((a, b) => gap(b[0], b[1], thresholds) - gap(a[0], a[1], thresholds))
  }, [run, thresholds, sortByGap])

  return (
    <section>
      <h2>Executive Overview</h2>
      <RunLoader />
  <RunDirectoryPicker />
      <div style={{ marginTop: 8 }}>
        <label>
          <input type="checkbox" checked={sortByGap} onChange={(e) => setSortByGap(e.target.checked)} /> 依「與 Profile 門檻差距」排序
        </label>
      </div>
      {!run && <p style={{ marginTop: 12 }}>請選擇 JSON 結果檔以載入 run。</p>}
      {run && (
        <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
          <div style={{ gridColumn: '1 / -1', padding: 12, border: '1px solid #ccc', borderRadius: 8, background: verdictColor(verdict?.verdict) }}>
            <strong>Verdict:</strong> {verdict?.verdict}
            {verdict?.failingMetrics?.length ? (
              <span style={{ marginLeft: 8 }}>↓ {verdict.failingMetrics.join(', ')}</span>
            ) : null}
            <span style={{ marginLeft: 16, opacity: 0.8 }}>Items: {run.counts.total}</span>
            {run.latencies && (
              <span style={{ marginLeft: 16, opacity: 0.8 }}>latency p50/p90: {formatNum(run.latencies.p50)} ms / {formatNum(run.latencies.p90)} ms</span>
            )}
          </div>
          {entries.map(([k, v]) => (
            <div key={k} style={{ padding: 12, border: '1px solid #ddd', borderRadius: 8 }}>
              <div style={{ fontWeight: 600 }}>{t(getMetricMeta(k).labelKey as any)}</div>
              <div style={{ fontSize: 24 }}>{getMetricMeta(k).format?.(v, locale)}</div>
              {renderGap(k, v ?? NaN, thresholds)}
            </div>
          ))}
        </div>
      )}
  <ThresholdEditor />
    </section>
  )
}

function verdictColor(v?: string) {
  if (v === 'Blocked') return '#ffe6e6'
  if (v === 'At Risk') return '#fff7e6'
  if (v === 'Ready') return '#eaffea'
  return '#f5f5f5'
}

function formatNum(v?: number) {
  if (v == null || Number.isNaN(v)) return 'N/A'
  return v.toFixed(3)
}

function renderGap(key: string, v: number, thresholds: any) {
  const th = thresholds[key as keyof typeof thresholds]
  if (!th || Number.isNaN(v)) return null
  if (v < th.critical) return <div style={{ color: 'crimson' }}>低於 critical {diffPct(v, th.critical)}</div>
  if (v < th.warning) return <div style={{ color: '#d48806' }}>低於 warning {diffPct(v, th.warning)}</div>
  return <div style={{ color: 'seagreen' }}>達標</div>
}

function diffPct(v: number, t: number) {
  const d = (t - v) / t
  return `(${(d * 100).toFixed(1)}%)`
}

function gap(key: string, v: number, thresholds: any) {
  const th = thresholds[key]
  if (!th || v == null || Number.isNaN(v)) return -Infinity
  // Gap is positive when below threshold, larger means worse
  const base = v < th.critical ? th.critical : th.warning
  if (!base) return -Infinity
  return Math.max(0, base - v)
}
