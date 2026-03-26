import { describe, it, expect } from 'vitest'
import { normalizeItem, RawItemSchema } from '@/core/schemas'

describe('normalizeItem', () => {
  it('keeps known and unknown metric keys within [0,1]', () => {
    const raw = RawItemSchema.parse({ id: '1', ContextPrecision: 0.9, custom_metric: 0.42, other: 2 })
    const it = normalizeItem(raw)
    expect(it.metrics.ContextPrecision).toBe(0.9)
    expect((it.metrics as any).custom_metric).toBe(0.42)
    expect((it.metrics as any).other).toBeUndefined()
  })
})
