import { describe, it, expect } from 'vitest'
import { evaluateVerdict } from '../verdict'

describe('evaluateVerdict', () => {
  const thresholds = {
    A: { warning: 0.9, critical: 0.8 },
    B: { warning: 0.5, critical: 0.4 },
  } as any

  it('returns Ready when all >= warning', () => {
    const r = evaluateVerdict({ A: 0.95, B: 0.5 }, thresholds)
    expect(r.verdict).toBe('Ready')
  })

  it('returns At Risk on any warning failure', () => {
    const r = evaluateVerdict({ A: 0.85, B: 0.6 }, thresholds)
    expect(r.verdict).toBe('At Risk')
  })

  it('returns Blocked on any critical failure', () => {
    const r = evaluateVerdict({ A: 0.75, B: 0.6 }, thresholds)
    expect(r.verdict).toBe('Blocked')
  })
})
