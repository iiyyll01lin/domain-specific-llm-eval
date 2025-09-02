// @ts-nocheck
import React, { useMemo, useState } from 'react'
import { usePortalStore } from '@/app/store/usePortalStore'
import { applyFilters } from '@/core/analysis/filters'
import { exportTableToCSV, exportTableToXLSX } from '@/core/exporter'

type StatRow = {
  metric: string
  mean: number | null
  median: number | null
  p50: number | null
  p90: number | null
}

function statsFor(values: number[]): StatRow {
  if (!values.length) return { metric: '', mean: null, median: null, p50: null, p90: null }
  const sorted = values.slice().sort((a, b) => a - b)
  const mean = sorted.reduce((s, v) => s + v, 0) / sorted.length
  const pick = (p: number) => sorted[Math.max(0, Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1))))]
  const p50 = pick(0.5)
  const p90 = pick(0.9)
  const median = p50
  return { metric: '', mean, median, p50, p90 }
}

function fmt(v: number | null) { return v == null ? 'N/A' : v.toFixed(3) }

export default function CompareView() {
  const runs = usePortalStore((s: any) => s.runs || {})
  const selectedRuns = usePortalStore((s: any) => s.selectedRuns || [])
  const thresholds = usePortalStore((s: any) => s.thresholds)
  const filters = usePortalStore((s: any) => s.filters)
  const [baseline, setBaseline] = useState<string | null>(null)

  const effectiveBaseline = baseline && selectedRuns.includes(baseline) ? baseline : (selectedRuns[0] || null)
  const metricKeys = useMemo(() => {
    const keys = new Set<string>()
    for (const id of selectedRuns) {
      const r = runs[id]
      for (const it of (r?.items || [])) {
        Object.keys((it.metrics as any) || {}).forEach((k) => keys.add(k))
      }
    }
    return Array.from(keys)
  }, [runs, selectedRuns])

  type CompareRow = { metric: string; cells: Record<string, StatRow & { deltaAbs?: number | null; deltaPct?: number | null }> }
  const table = useMemo(() => {
    if (!effectiveBaseline) return { rows: [] as CompareRow[], base: '' }
    const out: CompareRow[] = []
    for (const m of metricKeys) {
      const row: CompareRow = { metric: m, cells: {} }
      // compute baseline
      const baseVals = applyFilters(runs[effectiveBaseline]?.items || [], filters)
        .map((it) => (it.metrics as any)?.[m]).filter((v) => typeof v === 'number') as number[]
      const baseStat = statsFor(baseVals); baseStat.metric = m
      row.cells[effectiveBaseline] = baseStat
      const baseMean = baseStat.mean
      for (const id of selectedRuns) {
        const vals = applyFilters(runs[id]?.items || [], filters)
          .map((it) => (it.metrics as any)?.[m]).filter((v) => typeof v === 'number') as number[]
        const st = statsFor(vals); st.metric = m
        if (baseMean != null && st.mean != null) {
          const d = st.mean - baseMean
          const pct = baseMean === 0 ? null : (d / baseMean) * 100
          ;(st as any).deltaAbs = d
          ;(st as any).deltaPct = pct
        } else {
          ;(st as any).deltaAbs = null
          ;(st as any).deltaPct = null
        }
        row.cells[id] = st
      }
      out.push(row)
    }
    return { rows: out, base: effectiveBaseline }
  }, [runs, selectedRuns, effectiveBaseline, metricKeys, filters])

  const colorByDelta = (metric: string, d: number | null) => {
    if (d == null) return undefined
    // If increasing is good (most metrics are higher-is-better) then positive delta is green.
    // Thresholds can indicate desired direction implicitly; we assume higher better when threshold exists.
    const goodUp = !!(thresholds as any)?.[metric]
    const improve = goodUp ? d >= 0 : d <= 0
    return { color: improve ? '#2e7d32' : '#c62828' }
  }

  const onExportCsv = () => {
    const rows: Array<Record<string, unknown>> = []
    for (const r of table.rows) {
      const rec: Record<string, unknown> = { metric: r.metric }
      for (const id of selectedRuns) {
        const label = id.split('/').slice(-1)[0]
        const st = (r.cells as any)[id] as StatRow & { deltaAbs?: number|null; deltaPct?: number|null }
        rec[`${label}.mean`] = st?.mean ?? null
        rec[`${label}.median`] = st?.median ?? null
        rec[`${label}.p50`] = st?.p50 ?? null
        rec[`${label}.p90`] = st?.p90 ?? null
        rec[`${label}.deltaAbs`] = st?.deltaAbs ?? null
        rec[`${label}.deltaPct`] = st?.deltaPct ?? null
      }
      rows.push(rec)
    }
    exportTableToCSV('compare.csv', rows, { timestamp: new Date().toISOString(), filters, thresholds })
  }

  const onExportXlsx = async () => {
    const rows: Array<Record<string, unknown>> = []
    for (const r of table.rows) {
      const rec: Record<string, unknown> = { metric: r.metric }
      for (const id of selectedRuns) {
        const label = id.split('/').slice(-1)[0]
        const st = (r.cells as any)[id] as StatRow & { deltaAbs?: number|null; deltaPct?: number|null }
        rec[`${label}.mean`] = st?.mean ?? null
        rec[`${label}.median`] = st?.median ?? null
        rec[`${label}.p50`] = st?.p50 ?? null
        rec[`${label}.p90`] = st?.p90 ?? null
        rec[`${label}.deltaAbs`] = st?.deltaAbs ?? null
        rec[`${label}.deltaPct`] = st?.deltaPct ?? null
      }
      rows.push(rec)
    }
    await exportTableToXLSX('compare.xlsx', rows, { timestamp: new Date().toISOString(), filters, thresholds })
  }

  if (selectedRuns.length < 2) {
    return <div>Please add 2 or more runs to compare via Directory Picker.</div>
  }

  return (
    <section>
      <h2>Compare Runs</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <label>
          Baseline
          <select value={effectiveBaseline || ''} onChange={(e) => setBaseline((e.target as HTMLSelectElement).value)} style={{ marginLeft: 6 }} data-testid="compare-baseline">
            {selectedRuns.map((rid: string) => (
              <option key={rid} value={rid}>{rid.split('/').slice(-1)[0]}</option>
            ))}
          </select>
        </label>
        <button onClick={onExportCsv} aria-label="export-compare-csv">Export CSV</button>
        <button onClick={onExportXlsx} aria-label="export-compare-xlsx">Export XLSX</button>
      </div>
      <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: `repeat(${2 + selectedRuns.length * 2}, minmax(120px, 1fr))`, gap: 6 }}>
        <div style={{ fontWeight: 600 }}>Metric</div>
        {selectedRuns.map((rid: string) => (
          <React.Fragment key={`h-${rid}`}>
            <div style={{ fontWeight: 600 }}>{rid.split('/').slice(-1)[0]} (mean/median)</div>
            <div style={{ fontWeight: 600 }}>Δ (abs/%)</div>
          </React.Fragment>
        ))}
        {table.rows.map((r: CompareRow) => (
          <React.Fragment key={`row-${r.metric}`}>
            <div>{r.metric}</div>
      {selectedRuns.map((rid: string) => {
              const st = (r.cells as any)[rid] as StatRow & { deltaAbs?: number|null; deltaPct?: number|null }
              const d = st?.deltaAbs ?? null
              const pct = st?.deltaPct ?? null
              return (
                <React.Fragment key={`cell-${r.metric}-${rid}`}>
                  <div>{fmt(st?.mean ?? null)} / {fmt(st?.median ?? null)} (p50={fmt(st?.p50 ?? null)}, p90={fmt(st?.p90 ?? null)})</div>
                  <div style={colorByDelta(r.metric, d)}>{d == null ? 'N/A' : `${d >= 0 ? '+' : ''}${(d).toFixed(3)} / ${pct == null ? 'N/A' : ((pct >= 0 ? '+' : '') + pct.toFixed(1) + '%')}`}</div>
                </React.Fragment>
              )
            })}
          </React.Fragment>
        ))}
      </div>
    </section>
  )
}
