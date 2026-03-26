import { describe, it, expect } from 'vitest'
import { buildRowsFromItems } from '@/core/exporter'

describe('export row builder', () => {
  it('maps visible columns from item fields and metrics', () => {
    const items = [{ id: '1', user_input: 'q', metrics: { Faithfulness: 0.4 } } as any]
    const rows = buildRowsFromItems(items as any, ['id', 'user_input', 'Faithfulness'])
    expect(rows[0].id).toBe('1')
    expect(rows[0].user_input).toBe('q')
    expect(rows[0].Faithfulness).toBe(0.4)
  })
})
