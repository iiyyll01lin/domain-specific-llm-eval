import { describe, it, expect } from 'vitest'
import { applyFilters } from '@/core/analysis/filters'

describe('Filters metric ranges', () => {
  it('filters items by metric range', () => {
    const items = [
      { id: 'a', metrics: { Faithfulness: 0.2 } },
      { id: 'b', metrics: { Faithfulness: 0.7 } },
      { id: 'c', metrics: { Faithfulness: 0.95 } },
    ] as any
    const filtered = applyFilters(items, { metricRanges: { Faithfulness: [0.5, 0.9] } })
    expect(filtered.map((x:any) => x.id)).toEqual(['b'])
  })
})
