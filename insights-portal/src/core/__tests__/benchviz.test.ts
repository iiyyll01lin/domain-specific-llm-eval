import { describe, it, expect } from 'vitest'
import { computeBox } from '@/utils/benchviz'

describe('benchviz computeBox', () => {
  it('computes quartiles for a sorted set', () => {
    const stats = computeBox([1, 2, 3, 4, 5])
    expect(stats.n).toBe(5)
    expect(stats.min).toBe(1)
    expect(stats.max).toBe(5)
    expect(stats.median).toBe(3)
  expect(stats.q1).toBe(2)
  expect(stats.q3).toBe(4)
  })

  it('handles empty array', () => {
    const stats = computeBox([])
    expect(stats.n).toBe(0)
  })
})
