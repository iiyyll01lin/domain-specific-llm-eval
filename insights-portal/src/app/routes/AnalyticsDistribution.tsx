import React from 'react'
import * as echarts from 'echarts'
import { usePortalStore } from '@/app/store/usePortalStore'
import { applyFilters } from '@/core/analysis/filters'

export const AnalyticsDistribution: React.FC = () => {
  const run = usePortalStore((s) => s.run)
  const ref = React.useRef<HTMLDivElement | null>(null)
  const [metric, setMetric] = React.useState('Faithfulness')

  React.useEffect(() => {
    if (!ref.current) return
    const chart = echarts.init(ref.current)
  const filtered = applyFilters(run?.items || [], usePortalStore.getState().filters)
  const vals = filtered.map((it) => (it.metrics as any)?.[metric]).filter((v) => typeof v === 'number') as number[]
    const bins = 20
    const hist = new Array(bins).fill(0)
    for (const v of vals) {
      const idx = Math.min(bins - 1, Math.max(0, Math.floor(v * bins)))
      hist[idx]++
    }
    chart.setOption({
      grid: { left: 40, right: 20, top: 30, bottom: 30 },
      xAxis: { type: 'category', data: hist.map((_, i) => (i / bins).toFixed(2)) },
      yAxis: { type: 'value' },
      series: [{ type: 'bar', data: hist }],
      tooltip: { trigger: 'axis' },
    })
    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)
    return () => { window.removeEventListener('resize', onResize); chart.dispose() }
  }, [run, metric])

  return (
    <section>
      <h2>Analytics Distribution</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <label>
          Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value)} style={{ marginLeft: 6 }}>
            {Object.keys(run?.kpis || {}).map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
        </label>
      </div>
      <div ref={ref} style={{ height: 320, marginTop: 12 }} />
    </section>
  )
}
