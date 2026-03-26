import { describe, it, expect } from 'vitest'
import { buildRowsWithBookmarks } from '@/core/exporter'

describe('buildRowsWithBookmarks', () => {
  it('marks bookmarked rows with boolean flag', () => {
    const items = [
      { id: '1', user_input: 'q1', metrics: { Faithfulness: 0.1 } },
      { id: '2', user_input: 'q2', metrics: { Faithfulness: 0.9 } },
    ] as any
    const out = buildRowsWithBookmarks(items, ['id', 'user_input', 'Faithfulness'], new Set(['2']))
    expect(out).toHaveLength(2)
    expect(out[0].bookmarked).toBe(false)
    expect(out[1].bookmarked).toBe(true)
  })
})
