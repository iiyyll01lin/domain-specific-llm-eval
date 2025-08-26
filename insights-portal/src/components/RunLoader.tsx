import React from 'react'
// @ts-ignore - vite worker import
import ParserWorker from '../workers/parser.worker.ts?worker'
import { usePortalStore } from '@/app/store/usePortalStore'

export const RunLoader: React.FC = () => {
  const setRunData = usePortalStore((s) => s.setRunData)
  const [progress, setProgress] = React.useState<string>('')
  const [error, setError] = React.useState<string>('')

  const onPickFile = async () => {
    try {
      setError('')
      // Prefer File System Access API when available; otherwise fallback to <input type="file">
      let file: File | undefined

      const pickWithFS = async (): Promise<File | undefined> => {
        try {
          const api: any = (window as any).showOpenFilePicker
          if (!api) return undefined
          const res: any = await api({ types: [{ description: 'JSON', accept: { 'application/json': ['.json'] } }] })
          const handle = Array.isArray(res) ? res[0] : res
          if (!handle || typeof handle.getFile !== 'function') return undefined
          return await handle.getFile()
        } catch (e: any) {
          if (e?.name === 'AbortError') throw e
          // Fall back to input picker on any other error (e.g., insecure context)
          return undefined
        }
      }

      const pickWithInput = async (): Promise<File> =>
        await new Promise<File>((resolve, reject) => {
          const input = document.createElement('input')
          input.type = 'file'
          input.accept = 'application/json,.json'
          input.onchange = () => {
            const f = input.files?.[0]
            if (f) resolve(f)
            else reject(new Error('未選取檔案'))
          }
          input.click()
        })

      file = await pickWithFS()
      if (!file) file = await pickWithInput()

      await parseSummaryJson(file)
    } catch (e: any) {
      if (e?.name === 'AbortError') return
      setError(e?.message ?? String(e))
    }
  }

  const parseSummaryJson = async (file: File) => {
    setProgress('準備解析...')
    defaultParseSummary(file, setRunData, (m) => setError(String(m)), (p) => setProgress(p))
  }

  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
      <button onClick={onPickFile}>選擇 JSON 檔載入 run</button>
      {progress && <span aria-live="polite">{progress}</span>}
      {error && <span style={{ color: 'crimson' }}>{error}</span>}
    </div>
  )
}

export function defaultParseSummary(
  file: File,
  setRunData: (data: any) => void,
  onError?: (err: any) => void,
  onProgress?: (p: string) => void,
) {
  const worker: Worker = new ParserWorker()
  worker.onmessage = (ev: MessageEvent<any>) => {
    const msg = ev.data
    if (msg.type === 'progress') {
      onProgress?.(`${msg.phase}: ${msg.current}/${msg.total}`)
    } else if (msg.type === 'parsed') {
  setRunData({ items: msg.items, kpis: msg.kpis, counts: { total: msg.total }, latencies: msg.latencies })
      onProgress?.('完成')
      worker.terminate()
    } else if (msg.type === 'error') {
      onError?.(msg.message)
      worker.terminate()
    }
  }
  worker.postMessage({ type: 'parse-summary-json', file })
}

;(RunLoader as any).defaultParseSummary = defaultParseSummary

export default RunLoader
