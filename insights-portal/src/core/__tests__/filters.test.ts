import { describe, it, expect } from 'vitest'
import { applyFilters, aggregateKpisFiltered } from '@/core/analysis/filters'

const items = [
  { id: '1', language: 'zh', latencyMs: 100, metrics: { Faithfulness: 0.3, AnswerRelevancy: 0.8 } },
  { id: '2', language: 'en', latencyMs: 200, metrics: { Faithfulness: 0.6, AnswerRelevancy: 0.7 } },
  { id: '3', language: 'zh', latencyMs: 300, metrics: { Faithfulness: 0.9, AnswerRelevancy: 0.9 } },
] as any

describe('filters and aggregation', () => {
  it('filters by language and latency range', () => {
    const out = applyFilters(items, { language: 'zh', latencyRange: [150, 350], metricRanges: {} })
    expect(out.map((x) => x.id)).toEqual(['3'])
  })
  it('filters by metric range and aggregates', () => {
    const out = applyFilters(items, { metricRanges: { Faithfulness: [0.5, 1] } } as any)
    const kpis = aggregateKpisFiltered(out)
    expect(out.map((x) => x.id)).toEqual(['2', '3'])
    // Average of 0.6 and 0.9
    expect(Number(kpis.Faithfulness?.toFixed(3))).toBe(0.75)
  })
})
