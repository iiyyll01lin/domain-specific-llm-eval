import React from 'react'
import { usePortalStore } from '@/app/store/usePortalStore'
import type { RunParsed } from '@/core/types'
import { buildFilterChips } from '@/components/filters/chips'

type Props = {
  run?: RunParsed
}

export const FiltersBar: React.FC<Props> = ({ run }) => {
  const filters = usePortalStore((s) => s.filters)
  const setFilters = usePortalStore((s) => s.setFilters)
  const clearFilters = usePortalStore((s) => s.clearFilters)

  const metricKeys = React.useMemo(() => Object.keys(run?.kpis || {}), [run])

  return (
    <div className="card" style={{ padding: 8, marginTop: 8 }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <label>
          language
          <input
            style={{ marginLeft: 6, minWidth: 120 }}
            placeholder="e.g., zh, en"
            value={filters.language ?? ''}
            onChange={(e) => setFilters({ language: e.target.value || null })}
          />
        </label>
        <label>
          latency ms ≥
          <input
            type="number"
            style={{ width: 100, marginLeft: 6 }}
            value={filters.latencyRange?.[0] ?? ''}
            onChange={(e) => setFilters({ latencyRange: [e.target.value ? Number(e.target.value) : null, filters.latencyRange?.[1] ?? null] })}
          />
        </label>
        <label>
          ≤
          <input
            type="number"
            style={{ width: 100, marginLeft: 6 }}
            value={filters.latencyRange?.[1] ?? ''}
            onChange={(e) => setFilters({ latencyRange: [filters.latencyRange?.[0] ?? null, e.target.value ? Number(e.target.value) : null] })}
          />
        </label>
        <button onClick={clearFilters}>Clear All</button>
      </div>

      {metricKeys.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 8, marginTop: 8 }}>
          {metricKeys.map((k) => {
            const [lo, hi] = (filters.metricRanges?.[k] ?? [null, null]) as [number|null, number|null]
            return (
              <div key={k} style={{ border: '1px solid var(--border)', borderRadius: 6, padding: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>{k}</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className="small-muted">min</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={lo ?? 0}
                    onChange={(e) => {
                      const v = Number(e.target.value)
                      const cur = filters.metricRanges?.[k] ?? [null, null]
                      const next: [number|null, number|null] = [v, (cur[1] as number|null)]
                      setFilters({ metricRanges: { ...(filters.metricRanges || {}), [k]: next } })
                    }}
                  />
                  <span className="small-muted">max</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={hi ?? 1}
                    onChange={(e) => {
                      const v = Number(e.target.value)
                      const cur = filters.metricRanges?.[k] ?? [null, null]
                      const next: [number|null, number|null] = [(cur[0] as number|null), v]
                      setFilters({ metricRanges: { ...(filters.metricRanges || {}), [k]: next } })
                    }}
                  />
                </div>
                <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={lo ?? ''}
                    placeholder="min"
                    onChange={(e) => {
                      const v = e.target.value === '' ? null : Number(e.target.value)
                      const cur = filters.metricRanges?.[k] ?? [null, null]
                      const next: [number|null, number|null] = [v, (cur[1] as number|null)]
                      setFilters({ metricRanges: { ...(filters.metricRanges || {}), [k]: next } })
                    }}
                    style={{ width: 80 }}
                  />
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={hi ?? ''}
                    placeholder="max"
                    onChange={(e) => {
                      const v = e.target.value === '' ? null : Number(e.target.value)
                      const cur = filters.metricRanges?.[k] ?? [null, null]
                      const next: [number|null, number|null] = [(cur[0] as number|null), v]
                      setFilters({ metricRanges: { ...(filters.metricRanges || {}), [k]: next } })
                    }}
                    style={{ width: 80 }}
                  />
                  <button onClick={() => {
                    const mr = { ...(filters.metricRanges || {}) }
                    delete mr[k]
                    setFilters({ metricRanges: mr })
                  }}>Reset</button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Active chips */}
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
        {buildFilterChips(filters).map((c) => (
          <span key={c.key} style={{ padding: '2px 8px', borderRadius: 12, border: '1px solid var(--border)', background: 'var(--bg-muted)' }}>
            {c.label} <button onClick={c.onClear} aria-label={`clear ${c.key}`} style={{ marginLeft: 6 }}>×</button>
          </span>
        ))}
      </div>
    </div>
  )
}
