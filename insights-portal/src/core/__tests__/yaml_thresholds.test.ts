import { describe, it, expect } from 'vitest'
import { parseSimpleYAML, extractThresholdsFromConfig } from '@/core/yaml'

describe('yaml thresholds extraction', () => {
  it('parses simple thresholds mapping', () => {
    const text = `thresholds:\n  Faithfulness:\n    warning: 0.4\n    critical: 0.3\n  AnswerRelevancy: 0.65\n`
    const cfg = parseSimpleYAML(text)
    const th = extractThresholdsFromConfig(cfg)!
    expect(th.Faithfulness.warning).toBe(0.4)
    expect(th.Faithfulness.critical).toBe(0.3)
    expect(th.AnswerRelevancy.warning).toBe(0.65)
    expect(th.AnswerRelevancy.critical).toBeLessThanOrEqual(0.65)
  })
})
