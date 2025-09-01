import { describe, it, expect } from 'vitest'
import { measureRowDetailsLatency, immediateRowDetails } from '@/core/qa/rowDetails'
import { buildPdfManifest } from '@/core/exporter'

describe('QA Row Details SLA', () => {
  it('measures row details loader latency and should be under 200ms for immediate loader', async () => {
    const { durationMs, data } = await measureRowDetailsLatency(() => immediateRowDetails({ foo: 1 }), 'id-1')
    expect(data.foo).toBe(1)
    expect(durationMs).toBeLessThan(200)
  })
})

describe('Export PDF manifest builder', () => {
  it('builds a manifest with meta and sections', () => {
    const manifest = buildPdfManifest([
      { title: 'KPIs', type: 'table', payload: [{ k: 'Faithfulness', v: 0.7 }] },
      { title: 'Overview Chart', type: 'image', payload: { href: 'data:image/png;base64,AAA' } },
    ], { branding: { brand: 'ACME', title: 'Eval Report', footer: '© 2025' }, runId: 'local' })
    expect(manifest.sections.length).toBe(2)
    expect(manifest.meta?.branding?.brand).toBe('ACME')
  })
})
