// Minimal Option B PDF service using Puppeteer-like interface.
// Note: No network or actual installation here; this file documents a simple API and local harness.
// All comments in English only.

import { createServer } from 'http'
import { Buffer } from 'buffer'

export interface PdfRequest {
  title?: string
  manifest: {
    meta?: any
    sections: Array<{ title: string; type: 'table'|'image'|'text'; payload: any }>
  }
}

export function startServer(port = 8787) {
  const srv = createServer(async (req, res) => {
    if (req.method === 'POST' && req.url === '/render/pdf') {
      let body = ''
      req.on('data', (c) => { body += c })
      req.on('end', async () => {
        try {
          const parsed = JSON.parse(body) as PdfRequest
          // For this scaffold, just echo back a fake PDF buffer (empty PDF bytes) and headers.
          const suggestedName = (parsed.title?.trim() || 'report').replace(/[^a-zA-Z0-9._-]+/g, '_')
          const pdfBytes = Buffer.from('%PDF-1.4\n%\xFF\xFF\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF')
          res.writeHead(200, { 'Content-Type': 'application/pdf', 'Content-Disposition': `attachment; filename="${suggestedName}.pdf"` })
          res.end(pdfBytes)
        } catch (e: any) {
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

