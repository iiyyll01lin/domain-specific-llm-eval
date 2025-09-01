import React from 'react'

export type TimingSample = {
  at: number
  filterMs: number
  sampleMs: number
  aggregateMs: number
  total: number
}

export type BenchResult = { size: number; samplePct: number | null; coalesceMs: number; filterMs: number; sampleMs: number; aggregateMs: number }

interface Props {
  samples: TimingSample[]
  coalesceMs: number
  onCoalesceChange: (ms: number) => void
  bench?: BenchResult[]
  onRunBenchmarks?: () => void
}

export const DevTelemetryPanel: React.FC<Props> = ({ samples, coalesceMs, onCoalesceChange, bench = [], onRunBenchmarks }) => {
  // Rolling averages
  const avg = (key: keyof TimingSample) => (samples.length ? samples.reduce((s, x) => s + (x[key] as number), 0) / samples.length : 0)

  // Trend sparkline
  const sparkPoints = samples.slice(-40).map((s, i) => ({ x: i, y: s.filterMs + s.sampleMs + s.aggregateMs }))
  const sparkMax = sparkPoints.reduce((m, p) => Math.max(m, p.y), 0) || 1
  const sparkPath = sparkPoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${i * 4} ${30 - Math.round((p.y / sparkMax) * 28)}`).join(' ')

  // Bench helpers
  const groupBySize = React.useMemo(() => {
    const m = new Map<number, BenchResult[]>()
    for (const r of bench) {
      if (!m.has(r.size)) m.set(r.size, [])
      m.get(r.size)!.push(r)
    }
    return Array.from(m.entries()).sort((a, b) => a[0] - b[0])
  }, [bench])
  const totals = (arr: BenchResult[]) => arr.map((r) => r.filterMs + r.sampleMs + r.aggregateMs).sort((a, b) => a - b)
  const quantile = (arr: number[], p: number) => {
    if (!arr.length) return 0
    const idx = (arr.length - 1) * p
    const lo = Math.floor(idx)
    const hi = Math.ceil(idx)
    const h = idx - lo
    return (1 - h) * arr[lo] + h * arr[hi]
  }

  const exportBenchCsv = () => {
    if (!bench.length) return
    const rows = ['size,samplePct,coalesceMs,filterMs,sampleMs,aggregateMs']
    for (const r of bench) rows.push([r.size, r.samplePct ?? '', r.coalesceMs, r.filterMs, r.sampleMs, r.aggregateMs].join(','))
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'benchmarks.csv'
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <details style={{ marginTop: 8 }}>
      <summary>Dev: Worker timings</summary>
      <div style={{ fontSize: 12, marginTop: 6 }}>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <div>avg total: {Math.round(avg('total'))} ms</div>
          <div>avg filter: {Math.round(avg('filterMs'))} ms</div>
          <div>avg aggregate: {Math.round(avg('aggregateMs'))} ms</div>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <svg width={160} height={32} aria-label="timing-sparkline"><path d={sparkPath} stroke="#888" fill="none" strokeWidth={1} /></svg>
            <span>trend (last {sparkPoints.length})</span>
          </div>
        </div>
        <div style={{ marginTop: 6 }}>
          <label>
            Coalesce window (ms):
            <input
              style={{ marginLeft: 6, width: 80 }}
              type="number"
              min={0}
              step={10}
              value={coalesceMs}
              onChange={(e) => onCoalesceChange(Number(e.target.value))}
            />
          </label>
        </div>
        <div style={{ marginTop: 6 }}>
          {onRunBenchmarks && (
            <button onClick={onRunBenchmarks} style={{ padding: '4px 8px' }}>Run 5k/20k/100k benchmarks</button>
          )}
          {bench.length > 0 && (
            <button onClick={exportBenchCsv} style={{ padding: '4px 8px', marginLeft: 8 }}>Export Benchmarks CSV</button>
          )}
        </div>
        {bench.length > 0 && (
          <div style={{ marginTop: 8, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            {groupBySize.map(([size, arr]) => {
              const t = totals(arr)
              const q1 = quantile(t, 0.25)
              const med = quantile(t, 0.5)
              const q3 = quantile(t, 0.75)
              const min = t[0] ?? 0
              const max = t[t.length - 1] ?? 0
              const width = 200
              const scale = (v: number) => {
                const lo = min, hi = Math.max(max, min + 1)
                return Math.round(((v - lo) / (hi - lo)) * width)
              }
              return (
                <div key={size} style={{ border: '1px solid #eee', padding: 6, minWidth: 260 }}>
                  <div style={{ marginBottom: 4 }}><b>{size.toLocaleString()}</b> rows (n={arr.length})</div>
                  <div style={{ position: 'relative', width, height: 14, background: '#fafafa', border: '1px solid #ddd' }}>
                    <div style={{ position: 'absolute', left: 0, top: 6, width, height: 1, background: '#ddd' }} />
                    <div style={{ position: 'absolute', left: scale(q1), top: 2, width: Math.max(2, scale(q3) - scale(q1)), height: 10, background: '#cde' }} />
                    <div style={{ position: 'absolute', left: scale(med), top: 0, width: 2, height: 14, background: '#69c' }} />
                    <div style={{ position: 'absolute', left: scale(min), top: 6, width: 2, height: 2, background: '#999' }} />
                    <div style={{ position: 'absolute', left: scale(max), top: 6, width: 2, height: 2, background: '#999' }} />
                  </div>
                  <div style={{ marginTop: 4, color: '#555' }}>min/median/max: {Math.round(min)} / {Math.round(med)} / {Math.round(max)} ms</div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </details>
  )
}

export default DevTelemetryPanel
