import { describe, it, expect } from 'vitest'

// Note: In Vitest + jsdom, Web Worker instantiation with ?worker may not function.
// This test uses a conditional skip when Worker is not available.

describe('worker aggregate integration (smoke)', () => {
  it('aggregates simple items with filters', async () => {
  if (typeof globalThis !== 'object' || typeof (globalThis /** as any */).Worker === 'undefined') return
  const WM = (await import('@/workers/parser.worker.ts?worker')).default as unknown as { new(): Worker }
  const w = new WM()
    const items = [
      { id: '1', language: 'en', latencyMs: 100, metrics: { Faithfulness: 0.8, AnswerRelevancy: 0.7 } },
      { id: '2', language: 'zh', latencyMs: 200, metrics: { Faithfulness: 0.4, AnswerRelevancy: 0.6 } },
    ] as any[]
    const res = await new Promise<any>((resolve) => {
      w.onmessage = (ev: MessageEvent<any>) => { if (ev.data?.type === 'aggregated') { resolve(ev.data); w.terminate() } }
      w.postMessage({ type: 'aggregate', items, filters: { language: null, latencyRange: [null, null], metricRanges: {} } })
    })
    expect(res).toBeTruthy()
    expect(res.total).toBe(2)
    expect(res.kpis).toBeTruthy()
  })
})
