import { describe, it, expect } from 'vitest'
import { generateInsights } from '@/core/insights/engine'

describe('insights engine', () => {
  const thresholds = {
    ContextPrecision: { warning: 0.9, critical: 0.8 },
    ContextRecall: { warning: 0.9, critical: 0.8 },
    Faithfulness: { warning: 0.35, critical: 0.3 },
    AnswerRelevancy: { warning: 0.7, critical: 0.6 },
    ContextualKeywordMean: { warning: 0.55, critical: 0.4 }
  }
  it('detects hallucination risk', () => {
    const kpis = { ContextPrecision: 0.95, ContextRecall: 0.95, Faithfulness: 0.2 }
    const ins = generateInsights({ kpis, thresholds })
    expect(ins.find(i => i.id === 'hallucination_risk')).toBeTruthy()
  })
  it('detects keyword low', () => {
    const kpis = { ContextualKeywordMean: 0.3 }
    const ins = generateInsights({ kpis, thresholds })
    expect(ins.find(i => i.id === 'keyword_low')).toBeTruthy()
  })
  it('detects relevancy vs faithfulness gap', () => {
    const kpis = { AnswerRelevancy: 0.8, Faithfulness: 0.2 }
    const ins = generateInsights({ kpis, thresholds })
    expect(ins.find(i => i.id === 'relevancy_grounding_gap')).toBeTruthy()
  })
})
