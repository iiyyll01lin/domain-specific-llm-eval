import type { EvaluationItem } from '@/core/types'

export type Filters = {
  language?: string | null
  latencyRange?: [number | null, number | null]
  metricRanges?: Record<string, [number | null, number | null]>
}

export function applyFilters(items: EvaluationItem[], f: Filters): EvaluationItem[] {
  if (!items.length) return items
  return items.filter((it) => {
    if (f.language && (it.language || '').toLowerCase() !== f.language.toLowerCase()) return false
    if (f.latencyRange) {
      const [lo, hi] = f.latencyRange
      const v = it.latencyMs
      if (lo != null && (v == null || v < lo)) return false
      if (hi != null && (v == null || v > hi)) return false
    }
    if (f.metricRanges) {
      for (const [k, [lo, hi]] of Object.entries(f.metricRanges)) {
        const v = (it.metrics as any)[k] as number | undefined
        if (lo != null && (v == null || v < lo)) return false
        if (hi != null && (v == null || v > hi)) return false
      }
    }
    return true
  })
}

export function aggregateKpisFiltered(items: EvaluationItem[]): Record<string, number> {
  const acc: Record<string, { sum: number; n: number }> = {}
  for (const it of items) {
    for (const [k, v] of Object.entries(it.metrics)) {
      if (v == null || Number.isNaN(v as number)) continue
      const a = (acc[k] = acc[k] || { sum: 0, n: 0 })
      a.sum += v as number
      a.n += 1
    }
  }
  const out: Record<string, number> = {}
  for (const [k, a] of Object.entries(acc)) out[k] = a.n ? a.sum / a.n : NaN
  return out
}
