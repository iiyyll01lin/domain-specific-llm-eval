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
          const mode = String(process.env.PDF_RENDERER || '').toLowerCase()
          // Try puppeteer if explicitly requested; otherwise fallback to dummy.
      if (mode === 'puppeteer') {
            try {
        // Dynamic import via eval to avoid bundlers trying to resolve at build time.
        const puppeteer = await (0, eval)("import('puppeteer')")
              const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] })
              const page = await browser.newPage()
              // Build a trivial HTML from manifest for golden checks.
              const title = parsed.title || 'Report'
              const sections = Array.isArray(parsed.manifest?.sections) ? parsed.manifest.sections : []
              const bodyHtml = sections.map((s) => {
                if (!s || typeof s !== 'object') return ''
                const h = `<h2>${String(s.title || '')}</h2>`
                if (s.type === 'text') return `${h}<p>${String(s.payload || '')}</p>`
                if (s.type === 'table') {
                  const rows = Array.isArray(s.payload) ? s.payload : []
                  const keys = rows.length ? Object.keys(rows[0]) : []
                  const thead = `<tr>${keys.map((k) => `<th>${k}</th>`).join('')}</tr>`
                  const tbody = rows.map((r) => `<tr>${keys.map((k) => `<td>${String(r[k])}</td>`).join('')}</tr>`).join('')
                  return `${h}<table border=1 cellspacing=0 cellpadding=4>${thead}${tbody}</table>`
                }
                if (s.type === 'image' && s.payload?.href) return `${h}<img src="${s.payload.href}" alt="image" />`
                return h
              }).join('\n')
              const html = `<!doctype html><html><head><meta charset="utf-8"><title>${title}</title></head><body><header style="font-size:12px;opacity:0.7">${title}</header>${bodyHtml}<footer style="font-size:10px;opacity:0.6">${new Date().toISOString()}</footer></body></html>`
              await page.setContent(html, { waitUntil: 'load' })
              const pdfBytes = await page.pdf({ printBackground: true, format: 'A4', margin: { top: '28mm', bottom: '20mm', left: '10mm', right: '10mm' } })
              await browser.close()
              res.writeHead(200, { 'Content-Type': 'application/pdf', 'Content-Disposition': `attachment; filename="${suggestedName}.pdf"` })
              res.end(pdfBytes)
              return
            } catch (e) {
              // Fall through to dummy on any puppeteer error
            }
          }
          // Dummy bytes fallback
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
