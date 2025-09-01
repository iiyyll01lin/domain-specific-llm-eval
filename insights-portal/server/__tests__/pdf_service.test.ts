import { describe, it, expect } from 'vitest'
import { startServer } from '../pdf-service.js'

// Lightweight golden test for PDF service scaffold

describe('pdf-service scaffold', () => {
  it('returns a PDF-like buffer with %PDF header', async () => {
  const srv = startServer(0)
  const addr = srv.address() as any
  // Node on some platforms returns string | AddressInfo
  const port = typeof addr === 'string' ? 8787 : addr.port
    const res = await fetch(`http://127.0.0.1:${port}/render/pdf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'test', manifest: { sections: [] } })
    })
    const buf = new Uint8Array(await res.arrayBuffer())
    expect(res.status).toBe(200)
    expect(res.headers.get('content-type')).toContain('application/pdf')
    const txt = new TextDecoder('ascii').decode(buf.slice(0, 4))
    expect(txt).toBe('%PDF')
    srv.close()
  })
})
