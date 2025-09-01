// Minimal Option B PDF service using Node's http (ESM).
// Note: This is a JS scaffold to avoid TS Node types friction in lint. Comments in English only.

import http from 'node:http'

/**
 * @typedef {{ title?: string, manifest: { meta?: any, sections: Array<{ title: string, type: 'table'|'image'|'text', payload: any }> } }} PdfRequest
 */

export function startServer(port = 8787) {
  const srv = http.createServer(async (req, res) => {
    if (req.method === 'POST' && req.url === '/render/pdf') {
      let body = ''
      req.on('data', (c) => { body += c })
      req.on('end', async () => {
        try {
          const parsed /** @type {PdfRequest} */ = JSON.parse(body)
          const suggestedName = (parsed.title?.trim() || 'report').replace(/[^a-zA-Z0-9._-]+/g, '_')
          const pdfBytes = Buffer.from('%PDF-1.4\n%\xFF\xFF\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF')
          res.writeHead(200, { 'Content-Type': 'application/pdf', 'Content-Disposition': `attachment; filename="${suggestedName}.pdf"` })
          res.end(pdfBytes)
        } catch (e) {
          res.statusCode = 400
          res.end(String(e?.message || e))
        }
      })
      return
    }
    res.statusCode = 404
    res.end('Not Found')
  })
  srv.listen(port)
  return srv
}

// Auto-start when executed directly (not when imported by tests)
try {
  if (typeof process !== 'undefined' && Array.isArray(process.argv) && process.argv[1] && process.argv[1].endsWith('pdf-service.js')) {
    const port = process.env.PORT ? Number(process.env.PORT) : 8787
    startServer(port)
    // eslint-disable-next-line no-console
    console.log(`[pdf-service] listening on :${port}`)
  }
} catch {
  // ignore
}
