import { describe, it, expect } from 'vitest'
import { startServer } from '../pdf-service.js'

// Env-gated puppeteer test: set PDF_TEST_PUPPETEER=1 and install puppeteer to enable.

describe('pdf-service puppeteer golden (env-gated)', () => {
  it('renders with header/footer and non-trivial size in puppeteer mode', async () => {
    if (process.env.PDF_TEST_PUPPETEER !== '1') return
    process.env.PDF_RENDERER = 'puppeteer'
    const srv = startServer(0)
    const addr = srv.address() as any
    const port = typeof addr === 'string' ? 8787 : addr.port
    const res = await fetch(`http://127.0.0.1:${port}/render/pdf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: 'Golden',
        manifest: {
      meta: { branding: { title: 'Executive Overview', footer: 'Confidential' }, cover: true },
      sections: [
            { title: 'Intro', type: 'text', payload: 'hello' },
            { title: 'KPIs', type: 'table', payload: [{ k: 'Faithfulness', v: 0.7 }] },
          ],
        },
      }),
    })
    const buf = new Uint8Array(await res.arrayBuffer())
    expect(res.status).toBe(200)
    expect(res.headers.get('content-type')).toContain('application/pdf')
    // Read meta header to assert content
    const metaRaw = res.headers.get('x-pdf-info')
    if (!metaRaw) throw new Error('Missing X-PDF-Info header')
    const meta = JSON.parse(metaRaw)
    expect(meta.header).toContain('Executive Overview')
    expect(meta.footer).toContain('Confidential')
    expect(meta.tableRows).toBeGreaterThan(0)
  expect(meta.headerFooter).toBe(true)
  expect(meta.cover).toBe(true)
  expect(meta.sections).toBeGreaterThan(0)
    // size should be non-trivial
    expect(buf.length).toBeGreaterThan(2000)
    srv.close()
  }, 30000)
})
