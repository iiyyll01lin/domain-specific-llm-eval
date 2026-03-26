import { describe, it, expect } from 'vitest'
import { computeLatencyStats } from '@/core/schemas'

describe('computeLatencyStats', () => {
  it('computes avg and percentiles', () => {
    const items = [10, 20, 30, 40, 50].map((v, i) => ({ id: String(i), metrics: {}, latencyMs: v })) as any
    const s = computeLatencyStats(items)
    expect(s.avg).toBe(30)
    expect(s.p50).toBe(30)
    expect(s.p90).toBe(50)
    expect(s.p99).toBe(50)
  })
})
