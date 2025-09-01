import { describe, it, expect, beforeEach } from 'vitest'
import { loadBookmarks, saveBookmarks, loadVisibleCols, saveVisibleCols, loadVisibleMetrics, saveVisibleMetrics } from '@/core/qa/prefs'

describe('QA prefs helpers', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('persists and loads bookmarks', () => {
    const s = new Set<string>(['a', 'b'])
    saveBookmarks(s)
    const again = loadBookmarks()
    expect(again.has('a')).toBe(true)
    expect(again.has('b')).toBe(true)
  })

  it('persists and loads visible columns', () => {
    const cols = { question: true, answer: true, reference: false }
    saveVisibleCols(cols)
    const loaded = loadVisibleCols()
    expect(loaded).toEqual(cols)
  })

  it('persists and loads visible metrics', () => {
    const metrics = { Faithfulness: true, AnswerRelevancy: false }
    saveVisibleMetrics(metrics)
    const loaded = loadVisibleMetrics()
    expect(loaded).toEqual(metrics)
  })
})
