import React, { useMemo, useState } from 'react'
import { usePortalStore } from '@/app/store/usePortalStore'
import { applyFilters } from '@/core/analysis/filters'
import { exportTableToCSV, exportTableToXLSX, exportMultipleSheetsXLSX } from '@/core/exporter'
import { getMetricMeta } from '@/core/metrics/registry'

// Shared coloring helper using metric direction from registry
function colorByDeltaStyle(metric: string, d: number | null): React.CSSProperties | undefined {
  if (d == null) return undefined
  const meta = getMetricMeta(metric as any)
  const improve = meta.direction === 'lower' ? d <= 0 : d >= 0
  return { color: improve ? '#2e7d32' : '#c62828' }
}

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

  type CompareRow = { metric: string; cells: Record<string, StatRow & { deltaAbs?: number | null; deltaPct?: number | null; n?: number; naPct?: number }> }
  const table = useMemo(() => {
    if (!effectiveBaseline) return { rows: [] as CompareRow[], base: '' }
    const out: CompareRow[] = []
    for (const m of metricKeys) {
      const row: CompareRow = { metric: m, cells: {} }
      // compute baseline
      const baseItems = applyFilters(runs[effectiveBaseline]?.items || [], filters)
      const baseVals = baseItems
        .map((it) => (it.metrics as any)?.[m]).filter((v) => typeof v === 'number') as number[]
      const baseStat = statsFor(baseVals); baseStat.metric = m
      ;(baseStat as any).n = baseVals.length
      ;(baseStat as any).naPct = baseItems.length ? Math.max(0, 100 * (1 - baseVals.length / baseItems.length)) : 0
      row.cells[effectiveBaseline] = baseStat
      const baseMean = baseStat.mean
      for (const id of selectedRuns) {
        const items = applyFilters(runs[id]?.items || [], filters)
        const vals = items
          .map((it) => (it.metrics as any)?.[m]).filter((v) => typeof v === 'number') as number[]
        const st = statsFor(vals); st.metric = m
        ;(st as any).n = vals.length
        ;(st as any).naPct = items.length ? Math.max(0, 100 * (1 - vals.length / items.length)) : 0
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

  // colorByDeltaStyle is defined at module scope for reuse

  const onMetricClick = (metric: string) => {
    window.dispatchEvent(new CustomEvent('portal:set-analytics-metric', { detail: { metric } }))
    window.dispatchEvent(new CustomEvent('portal:navigate', { detail: { route: 'analytics' } }))
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
  rec[`${label}.samples`] = (st as any)?.n ?? null
  rec[`${label}.naPct`] = (st as any)?.naPct ?? null
      }
      rows.push(rec)
    }
  exportTableToCSV('compare.csv', rows, { timestamp: new Date().toISOString(), filters, thresholds, branding: { brand: 'Insights Portal', title: 'Compare Report', footer: 'Generated locally — offline mode' } })
  }

  const onExportXlsx = async () => {
    const detailRows: Array<Record<string, unknown>> = []
    const overviewRows: Array<Record<string, unknown>> = []
    for (const r of table.rows) {
      const rec: Record<string, unknown> = { metric: r.metric }
      const ov: Record<string, unknown> = { metric: r.metric }
      for (const id of selectedRuns) {
        const label = id.split('/').slice(-1)[0]
        const st = (r.cells as any)[id] as StatRow & { deltaAbs?: number|null; deltaPct?: number|null }
        rec[`${label}.mean`] = st?.mean ?? null
        rec[`${label}.median`] = st?.median ?? null
        rec[`${label}.p50`] = st?.p50 ?? null
        rec[`${label}.p90`] = st?.p90 ?? null
        rec[`${label}.deltaAbs`] = st?.deltaAbs ?? null
        rec[`${label}.deltaPct`] = st?.deltaPct ?? null
        rec[`${label}.samples`] = (st as any)?.n ?? null
        rec[`${label}.naPct`] = (st as any)?.naPct ?? null
        ov[`${label}.n`] = (st as any)?.n ?? null
        ov[`${label}.naPct`] = (st as any)?.naPct ?? null
      }
      detailRows.push(rec)
      overviewRows.push(ov)
    }
    await exportMultipleSheetsXLSX(
      'compare.xlsx',
      [
        { name: 'data', rows: detailRows },
        { name: 'overview', rows: overviewRows },
      ],
      { timestamp: new Date().toISOString(), filters, thresholds, branding: { brand: 'Insights Portal', title: 'Compare Report', footer: 'Generated locally — offline mode' } }
    )
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
  {/* Cohort-based multi-run compare (focused on one metric) */}
  <CohortCompare selectedRuns={selectedRuns} runs={runs} thresholds={thresholds} filters={filters} />
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
            <div>
              {(() => {
                const meta = getMetricMeta(r.metric as any)
                const th = (thresholds as any)?.[r.metric]
                const dir = meta.direction || 'higher'
                const tip = th
                  ? `Metric: ${r.metric}\nDirection: ${dir} is better\nTargets: warning ${dir === 'lower' ? '≤' : '≥'} ${typeof th.warning === 'number' ? th.warning.toFixed(2) : th.warning}, critical ${dir === 'lower' ? '≤' : '≥'} ${typeof th.critical === 'number' ? th.critical.toFixed(2) : th.critical}`
                  : `Metric: ${r.metric}\nDirection: ${dir} is better`
                return (
                  <>
                    <button onClick={() => onMetricClick(r.metric)} style={{ all: 'unset', cursor: 'pointer', color: '#1976d2' }} title={tip}>{r.metric}</button>
                    {th && (
                      <span style={{ marginLeft: 6, color: '#888' }} title={tip}>
                        target {dir === 'lower' ? '≤' : '≥'} {typeof th.warning === 'number' ? th.warning.toFixed(2) : th.warning}
                      </span>
                    )}
                  </>
                )
              })()}
            </div>
      {selectedRuns.map((rid: string) => {
              const st = (r.cells as any)[rid] as StatRow & { deltaAbs?: number|null; deltaPct?: number|null }
              const d = st?.deltaAbs ?? null
              const pct = st?.deltaPct ?? null
              return (
                <React.Fragment key={`cell-${r.metric}-${rid}`}>
                  <div>
                    {fmt(st?.mean ?? null)} / {fmt(st?.median ?? null)} (p50={fmt(st?.p50 ?? null)}, p90={fmt(st?.p90 ?? null)})
                    <span style={{ marginLeft: 6, color: '#888' }}>n={(st as any)?.n ?? 0}, N/A={(st as any)?.naPct != null ? ((st as any).naPct as number).toFixed(1) + '%' : 'N/A'}</span>
                  </div>
                  <div style={colorByDeltaStyle(r.metric, d)}>{d == null ? 'N/A' : `${d >= 0 ? '+' : ''}${(d).toFixed(3)} / ${pct == null ? 'N/A' : ((pct >= 0 ? '+' : '') + pct.toFixed(1) + '%')}`}</div>
                </React.Fragment>
              )
            })}
          </React.Fragment>
        ))}
      </div>
    </section>
  )
}

// Lightweight cohort compare: compute per-run group means and deltas vs baseline for a chosen metric
function CohortCompare(props: { selectedRuns: string[]; runs: any; thresholds: any; filters: any }) {
  const { selectedRuns, runs, thresholds, filters } = props
  const [metric, setMetric] = useState<string>('Faithfulness')
  const [cohort, setCohort] = useState<'language'|'success'|'failingMetric'>('language')
  const [expanded, setExpanded] = useState<boolean>(false)
  const baseline = selectedRuns[0]
  const data = useMemo(() => {
    if (!baseline || selectedRuns.length < 2) return { groups: [] as string[], rows: [] as any[] }
    // Gather all groups across runs
    const groupsSet = new Set<string>()
    const perRun: Record<string, Record<string, number[]>> = {}
    for (const rid of selectedRuns) {
      const items = applyFilters(runs[rid]?.items || [], filters)
      const map: Record<string, number[]> = {}
      for (const it of items) {
        const v = (it.metrics as any)?.[metric]
        if (typeof v !== 'number') continue
        let key = 'N/A'
        if (cohort === 'language') key = (it.language || 'N/A') as string
        else if (cohort === 'success') {
          const ok = Object.entries((it.metrics as any) || {}).every(([k, vv]) => {
            const th = (thresholds as any)?.[k]
            return th ? (typeof vv === 'number' ? vv >= th.warning : true) : true
          })
          key = ok ? 'success' : 'failure'
        } else if (cohort === 'failingMetric') {
          let placed = false
          for (const [mk, vv] of Object.entries((it.metrics as any) || {})) {
            const th = (thresholds as any)?.[mk]
            if (th && typeof vv === 'number' && vv < th.warning) {
              key = mk
              placed = true
              break
            }
          }
          if (!placed) key = 'none'
        }
        ;(map[key] = map[key] || []).push(v)
        groupsSet.add(key)
      }
      perRun[rid] = map
    }
    const groups = Array.from(groupsSet)
    const rows: any[] = groups.map((g) => {
      const rec: any = { group: g }
      const baseVals = perRun[baseline]?.[g] || []
      const baseMean = baseVals.length ? baseVals.reduce((s, v) => s + v, 0) / baseVals.length : null
      for (const rid of selectedRuns) {
        const vals = perRun[rid]?.[g] || []
        const m = vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null
        rec[rid] = { mean: m, n: vals.length, delta: baseMean != null && m != null ? m - baseMean : null }
      }
      return rec
    })
    return { groups, rows }
  }, [selectedRuns, runs, thresholds, filters, cohort, metric, baseline])

  const exportCsv = () => {
    const rows: Record<string, unknown>[] = []
    for (const r of data.rows) {
      const rec: any = { group: r.group }
      for (const rid of selectedRuns) {
        const label = rid.split('/').slice(-1)[0]
        const cell = (r as any)[rid] || {}
        rec[`${label}.mean`] = cell.mean ?? null
        rec[`${label}.n`] = cell.n ?? 0
        rec[`${label}.delta`] = cell.delta ?? null
      }
      rows.push(rec)
    }
    exportTableToCSV(`cohort_${metric}.csv`, rows, { timestamp: new Date().toISOString(), filters, thresholds, branding: { brand: 'Insights Portal', title: 'Compare Report', footer: 'Generated locally — offline mode' } })
  }
  const exportXlsx = async () => {
    const rows: Record<string, unknown>[] = []
    for (const r of data.rows) {
      const rec: any = { group: r.group }
      for (const rid of selectedRuns) {
        const label = rid.split('/').slice(-1)[0]
        const cell = (r as any)[rid] || {}
        rec[`${label}.mean`] = cell.mean ?? null
        rec[`${label}.n`] = cell.n ?? 0
        rec[`${label}.delta`] = cell.delta ?? null
      }
      rows.push(rec)
    }
    await exportTableToXLSX(`cohort_${metric}.xlsx`, rows, { timestamp: new Date().toISOString(), filters, thresholds, branding: { brand: 'Insights Portal', title: 'Compare Report', footer: 'Generated locally — offline mode' } })
  }

  if (selectedRuns.length < 2) return null
  return (
    <div style={{ marginTop: 16 }}>
      <button onClick={() => setExpanded((v) => !v)} aria-expanded={expanded} aria-controls="cohort-panel">
        Cohort Compare
      </button>
      {expanded && (
        <div id="cohort-panel" style={{ marginTop: 8 }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <label>
              Metric
              <select value={metric} onChange={(e) => setMetric((e.target as HTMLSelectElement).value)} style={{ marginLeft: 6 }}>
                {Object.keys((runs[selectedRuns[0]]?.items?.[0]?.metrics || {})).map((k) => (
                  <option key={k} value={k}>{k}</option>
                ))}
              </select>
            </label>
            <label>
              Cohort
              <select value={cohort} onChange={(e) => setCohort((e.target as HTMLSelectElement).value as any)} style={{ marginLeft: 6 }}>
                <option value="language">Language</option>
                <option value="success">Success/Failure</option>
                <option value="failingMetric">Failing Metric</option>
              </select>
            </label>
            <button onClick={exportCsv} aria-label="export-cohort-csv">Export CSV</button>
            <button onClick={exportXlsx} aria-label="export-cohort-xlsx">Export XLSX</button>
          </div>
          <div style={{ display: 'grid', gap: 6, gridTemplateColumns: `repeat(${1 + selectedRuns.length * 2}, minmax(120px, 1fr))`, marginTop: 8 }}>
            <div style={{ fontWeight: 600 }}>Group</div>
            {selectedRuns.map((rid) => (
              <React.Fragment key={`ch-${rid}`}>
                <div style={{ fontWeight: 600 }}>{rid.split('/').slice(-1)[0]} (mean)</div>
                <div style={{ fontWeight: 600 }}>Δ vs base</div>
              </React.Fragment>
            ))}
            {data.rows.map((row) => (
              <React.Fragment key={`cg-${row.group}`}>
                <div>{row.group}</div>
                {selectedRuns.map((rid) => {
                  const cell = (row as any)[rid] || {}
                  const d = cell.delta as number | null | undefined
                  return (
                    <React.Fragment key={`cg-${row.group}-${rid}`}>
                      <div>{cell.mean == null ? 'N/A' : (cell.mean as number).toFixed(3)} (n={cell.n ?? 0})</div>
                      <div style={colorByDeltaStyle(metric, d ?? null)}>{d == null ? 'N/A' : `${d >= 0 ? '+' : ''}${(d as number).toFixed(3)}`}</div>
                    </React.Fragment>
                  )
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
