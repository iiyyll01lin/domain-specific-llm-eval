import React from 'react'
import * as echarts from 'echarts'
import { usePortalStore } from '@/app/store/usePortalStore'
import { applyFilters } from '@/core/analysis/filters'
import { exportTableToCSV, exportTableToXLSX } from '@/core/exporter'

export const AnalyticsDistribution: React.FC = () => {
  const run = usePortalStore((s) => s.run)
  const setFilters = usePortalStore((s) => s.setFilters)
  const ref = React.useRef<HTMLDivElement | null>(null)
  const [metric, setMetric] = React.useState('Faithfulness')
  const [mode, setMode] = React.useState<'hist'|'box'|'scatter'>('hist')
  const [scatterY, setScatterY] = React.useState('AnswerRelevancy')
  const thresholds = usePortalStore((s) => s.thresholds)
  const [cohort, setCohort] = React.useState<'language'|'success'>('language')

  // Recompute and draw chart when inputs change
  const stableFilters = usePortalStore((s) => s.filters)
  React.useEffect(() => {
    if (!ref.current) return
    const chart = echarts.init(ref.current)
  const filtered = applyFilters(run?.items || [], stableFilters)
    const xVals = filtered.map((it) => (it.metrics as any)?.[metric]).filter((v) => typeof v === 'number') as number[]
    if (mode === 'hist') {
      const bins = 20
      const hist = new Array(bins).fill(0)
      for (const v of xVals) {
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
    } else if (mode === 'box') {
      // Group by language cohort when available; otherwise single series
      const items = filtered
      const groups: Record<string, number[]> = {}
      if (items.length) {
        for (const it of items) {
          const v = (it.metrics as any)?.[metric]
          if (typeof v !== 'number') continue
          let key = 'N/A'
          if (cohort === 'language') key = (it.language || 'N/A') as string
          else if (cohort === 'success') {
            // derive simple success flag from thresholds: success if all metrics >= warning
            const ok = Object.entries((it.metrics as any) || {}).every(([k, vv]) => {
              const th = (thresholds as any)?.[k]
              return th ? (typeof vv === 'number' ? vv >= th.warning : false) : true
            })
            key = ok ? 'success' : 'failure'
          }
          ;(groups[key] = groups[key] || []).push(v)
        }
      }
      const groupKeys = Object.keys(groups)
      const isGrouped = groupKeys.length > 0
      const vals = isGrouped ? [] : xVals.slice().sort((a, b) => a - b)
      const q = (p: number) => vals.length ? vals[Math.max(0, Math.min(vals.length - 1, Math.floor(p * (vals.length - 1))))] : null
      if (!isGrouped) {
        const min = vals[0] ?? null, q1 = q(0.25), med = q(0.5), q3 = q(0.75), max = vals[vals.length - 1] ?? null
        const iqr = (q3 != null && q1 != null) ? (q3 - q1) : null
        const lowerFence = iqr != null ? (q1 as number) - 1.5 * iqr : null
        const upperFence = iqr != null ? (q3 as number) + 1.5 * iqr : null
        const outliers = (iqr != null) ? (vals as number[]).filter((v) => v < (lowerFence as number) || v > (upperFence as number)).map((v) => [0, v]) : []
        chart.setOption({
          xAxis: { type: 'category', data: [metric] },
          yAxis: { type: 'value', min: 0, max: 1 },
          series: [
            { type: 'boxplot', data: [[min, q1, med, q3, max]] },
            { type: 'scatter', data: outliers, name: 'outliers', symbolSize: 6, itemStyle: { color: '#e57373' } }
          ],
          tooltip: { trigger: 'item' },
        })
      } else {
        const xs = groupKeys
        const boxData: Array<[number | null, number | null, number | null, number | null, number | null]> = []
        const outlierData: Array<[number, number]> = []
        xs.forEach((gk, i) => {
          const arr = (groups[gk] || []).slice().sort((a, b) => a - b)
          const pick = (p: number) => arr.length ? arr[Math.max(0, Math.min(arr.length - 1, Math.floor(p * (arr.length - 1))))] : null
          const min = arr[0] ?? null, q1 = pick(0.25), med = pick(0.5), q3 = pick(0.75), max = arr[arr.length - 1] ?? null
          boxData.push([min, q1, med, q3, max])
          const iqr = (q3 != null && q1 != null) ? (q3 - q1) : null
          const lowerFence = iqr != null ? (q1 as number) - 1.5 * iqr : null
          const upperFence = iqr != null ? (q3 as number) + 1.5 * iqr : null
          if (iqr != null) {
            for (const v of arr) {
              if (v < (lowerFence as number) || v > (upperFence as number)) outlierData.push([i, v])
            }
          }
        })
        chart.setOption({
          xAxis: { type: 'category', data: xs },
          yAxis: { type: 'value', min: 0, max: 1 },
          series: [
            { type: 'boxplot', data: boxData },
            { type: 'scatter', data: outlierData, name: 'outliers', symbolSize: 6, itemStyle: { color: '#e57373' } }
          ],
          tooltip: { trigger: 'item' },
        })
      }
    } else if (mode === 'scatter') {
      const yVals = filtered.map((it) => (it.metrics as any)?.[scatterY]).filter((v) => typeof v === 'number') as number[]
      const data = xVals.map((x, i) => [x, yVals[i] ?? null]).filter((p) => typeof p[1] === 'number') as Array<[number, number]>
      // Reflect current metric range filters into axis min/max when applicable
      const mr = (stableFilters as any)?.metricRanges || {}
      const xr = (mr as any)[metric] as [number|null, number|null] | undefined
      const yr = (mr as any)[scatterY] as [number|null, number|null] | undefined
      chart.setOption({
        xAxis: { type: 'value', min: (xr?.[0] ?? 0), max: (xr?.[1] ?? 1) },
        yAxis: { type: 'value', min: (yr?.[0] ?? 0), max: (yr?.[1] ?? 1) },
        series: [{ type: 'scatter', data }],
        tooltip: { trigger: 'item' },
        brush: { toolbox: ['rect', 'polygon', 'clear'], xAxisIndex: 'all', yAxisIndex: 'all' },
        toolbox: { feature: { brush: {} } },
      })
      chart.off('brushselected')
      chart.on('brushselected', (params: any) => {
        const batch = params.batch?.[0]
        if (!batch) return
        const areas = Array.isArray(batch.areas) ? batch.areas : []
        // Merge multiple brushed areas into a single min/max per axis
        let xMin: number | null = null, xMax: number | null = null
        let yMin: number | null = null, yMax: number | null = null
        for (const a of areas) {
          const rx = a?.range?.[0]
          const ry = a?.range?.[1]
          if (rx) {
            xMin = xMin == null ? rx[0] : Math.min(xMin, rx[0])
            xMax = xMax == null ? rx[1] : Math.max(xMax, rx[1])
          }
          if (ry) {
            yMin = yMin == null ? ry[0] : Math.min(yMin, ry[0])
            yMax = yMax == null ? ry[1] : Math.max(yMax, ry[1])
          }
        }
        if (xMin == null || xMax == null || yMin == null || yMax == null) return
        // Write back to global filters (metric ranges)
        setFilters({ metricRanges: { [metric]: [xMin, xMax], [scatterY]: [yMin, yMax] } })
      })
    }
    const onResize = () => chart.resize()
    window.addEventListener('resize', onResize)
    return () => { window.removeEventListener('resize', onResize); chart.dispose() }
  }, [run, metric, scatterY, mode, stableFilters, setFilters, cohort, thresholds])

  const onExportCsv = () => {
    const filtered = applyFilters(run?.items || [], stableFilters)
    const rows = filtered.map((it) => ({ id: it.id, metric, value: (it.metrics as any)?.[metric] ?? null }))
    exportTableToCSV(`analytics_hist_${metric}.csv`, rows, { timestamp: new Date().toISOString() })
  }
  const onExportXlsx = async () => {
    const filtered = applyFilters(run?.items || [], stableFilters)
    const rows = filtered.map((it) => ({ id: it.id, metric, value: (it.metrics as any)?.[metric] ?? null }))
    await exportTableToXLSX(`analytics_${metric}.xlsx`, rows, { timestamp: new Date().toISOString() })
  }
  const onExportPng = () => {
    const inst = echarts.getInstanceByDom(ref.current!)
    if (!inst) return
    const url = inst.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#fff' })
    const a = document.createElement('a')
    a.href = url
    a.download = `analytics_${mode}_${metric}.png`
    a.click()
  }

  return (
    <section>
      <h2>Analytics Distribution</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <label>
          Mode
          <select value={mode} onChange={(e) => setMode(e.target.value as any)} style={{ marginLeft: 6 }}>
            <option value="hist">Histogram</option>
            <option value="box">Box</option>
            <option value="scatter">Scatter</option>
          </select>
        </label>
        {mode === 'box' && (
          <label>
            Cohort
            <select value={cohort} onChange={(e) => setCohort(e.target.value as any)} style={{ marginLeft: 6 }}>
              <option value="language">Language</option>
              <option value="success">Success/Failure</option>
            </select>
          </label>
        )}
        <label>
          Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value)} style={{ marginLeft: 6 }}>
            {Object.keys(run?.kpis || {}).map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
        </label>
        {mode === 'scatter' && (
          <label>
            Y
            <select value={scatterY} onChange={(e) => setScatterY(e.target.value)} style={{ marginLeft: 6 }}>
              {Object.keys(run?.kpis || {}).map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </label>
        )}
        <button onClick={onExportCsv} aria-label="export-analytics-csv">Export CSV</button>
        <button onClick={onExportXlsx} aria-label="export-analytics-xlsx">Export XLSX</button>
        <button onClick={onExportPng} aria-label="export-analytics-png">Export PNG</button>
      </div>
      <div ref={ref} style={{ height: 320, marginTop: 12 }} aria-label="histogram" role="img" />
    </section>
  )
}
