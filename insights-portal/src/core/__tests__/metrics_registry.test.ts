import { describe, it, expect } from 'vitest'
import { getMetricMeta } from '@/core/metrics/registry'
import { aggregateKpis } from '@/core/schemas'

describe('Metrics registry and aggregation (T-013)', () => {
  it('falls back to generic meta for unknown keys', () => {
    const m = getMetricMeta('MyCustomMetric' as any)
    expect(m.key).toBe('MyCustomMetric')
    expect(m.labelKey).toBe('metrics.MyCustomMetric.label')
    expect(typeof m.format).toBe('function')
  })

  it('aggregateKpis includes unknown metrics present in items', () => {
    const items = [
      { id: '1', metrics: { MyCustomMetric: 0.2 } },
      { id: '2', metrics: { MyCustomMetric: 0.4 } },
    ] as any
    const kpis = aggregateKpis(items)
    expect(kpis.MyCustomMetric).toBeCloseTo(0.3, 5)
  })
})
