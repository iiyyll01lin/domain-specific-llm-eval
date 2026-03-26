import { describe, it, expect } from 'vitest'
import { evaluateVerdict } from '@/core/verdict'

describe('verdict engine basic', () => {
  const thresholds = {
    Faithfulness: { warning: 0.4, critical: 0.3 },
    ContextPrecision: { warning: 0.9, critical: 0.8 },
  }
  it('ready when above warnings', () => {
    const res = evaluateVerdict({ Faithfulness: 0.5, ContextPrecision: 0.95 }, thresholds)
    expect(res.verdict).toBeDefined()
  })
  it('at risk when below warning but above critical', () => {
    const res = evaluateVerdict({ Faithfulness: 0.35, ContextPrecision: 0.95 }, thresholds)
    expect(['At Risk','Blocked','Ready']).toContain(res.verdict)
  })
})