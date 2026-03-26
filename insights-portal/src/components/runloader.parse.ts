// Helper functions extracted from RunLoader to satisfy react-refresh/only-export-components
// Vite worker import; cast to avoid needing dom worker types here
import WorkerModule from '../workers/parser.worker.ts?worker'

export function defaultParseSummary(
  file: File,
  setRunData: (data: any) => void,
  onError?: (err: any) => void,
  onProgress?: (p: string) => void,
) {
  const worker: Worker = new (WorkerModule as unknown as { new (): Worker })()

  // UI：read
  onProgress?.('read: 0/1')
  // Broadcast event before starting parsing to let ExecutiveOverview clear old errors
  window.dispatchEvent(new CustomEvent('run-parse-start'))

  worker.onmessage = (ev: MessageEvent<any>) => {
    const msg = ev.data
    if (msg.type === 'progress') {
      onProgress?.(`${msg.phase}: ${msg.current}/${msg.total}`)
    } else if (msg.type === 'parsed') {
      setRunData({ id: file.name, items: msg.items, kpis: msg.kpis, counts: { total: msg.total }, latencies: msg.latencies, artifacts: { summaryJson: file } })
      onProgress?.('完成')
      worker.terminate()
    } else if (msg.type === 'error') {
       let detail: any
      if (msg.error && typeof msg.error === 'object') {
        detail = msg.error                            
      } else if (typeof msg.message === 'string') {
        detail = msg.message                          
      } else {
        detail = 'Unknown parse error'               
      }

      const text =
        typeof detail === 'string'
          ? detail
          : (detail.message || detail.raw || (detail.code ? `Error code: ${detail.code}` : JSON.stringify(detail)))

      onError?.(detail)
      onProgress?.('失敗')

      // Broadcast an event when parsing fails, allowing the UI to display an error banner
      window.dispatchEvent(new CustomEvent('run-parse-error', { detail }))
      
      worker.terminate()
    }
  }

  // Any worker-level errors (e.g. initialization failure)
  worker.onerror = (e) => {
    onError?.(e.message || String(e))
    onProgress?.('失敗')

    /** Worker initialization errors are also broadcast */
    window.dispatchEvent(new CustomEvent('run-parse-error', {
      detail: { code: 'WORKER_RUNTIME', raw: e }
    }))

    worker.terminate()
  }

  worker.postMessage({ type: 'parse-summary-json', file })
}
