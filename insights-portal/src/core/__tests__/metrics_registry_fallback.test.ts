import { describe, it, expect } from 'vitest'
import { getMetricMeta, metricDirection } from '@/core/metrics/registry'

describe('metrics registry fallback', () => {
  it('returns known meta for known metric', () => {
    const m = getMetricMeta('Faithfulness')
    expect(m.labelKey).toContain('Faithfulness')
    expect(m.direction).toBe('higher')
  })
  it('creates fallback for unknown metric', () => {
    const m = getMetricMeta('MyNewMetric' as any)
    expect(m.key).toBe('MyNewMetric')
    expect(m.labelKey).toBe('metrics.MyNewMetric.label')
    expect(metricDirection('MyNewMetric' as any)).toBe('higher')
  })
})
