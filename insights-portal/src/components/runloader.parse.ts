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
  worker.onmessage = (ev: MessageEvent<any>) => {
    const msg = ev.data
    if (msg.type === 'progress') {
      onProgress?.(`${msg.phase}: ${msg.current}/${msg.total}`)
    } else if (msg.type === 'parsed') {
      setRunData({ id: file.name, items: msg.items, kpis: msg.kpis, counts: { total: msg.total }, latencies: msg.latencies, artifacts: { summaryJson: file } })
      onProgress?.('完成')
      worker.terminate()
    } else if (msg.type === 'error') {
      onError?.(msg.message)
      worker.terminate()
    }
  }
  worker.postMessage({ type: 'parse-summary-json', file })
}
