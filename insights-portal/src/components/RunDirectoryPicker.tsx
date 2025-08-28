import React from 'react'
import { iterateDir } from '@/core/fs'
import { usePortalStore } from '@/app/store/usePortalStore'
import { parseSimpleYAML, extractThresholdsFromConfig } from '@/core/yaml'

interface DetectedRun {
  runPath: string
  summaryJson?: File
  configYaml?: File
  totalItems?: number
  metricsCoverage?: Record<string, number>
  artifactCounts?: { resultJson: number; csv: number; otherJson: number }
}

export const RunDirectoryPicker: React.FC = () => {
  const setRunData = usePortalStore((s) => s.setRunData)
  const [runs, setRuns] = React.useState<DetectedRun[]>([])
  const [error, setError] = React.useState('')
  const [busy, setBusy] = React.useState(false)
  const [scanning, setScanning] = React.useState(false)

  const onPickDir = async () => {
    setError('')
    setBusy(true)
    try {
      const dir = await (window as any).showDirectoryPicker?.()
      if (!dir) return
      const found: Record<string, DetectedRun> = {}
      for await (const { path, handle } of iterateDir(dir, 0, 3)) {
        if (handle.kind === 'file') {
          const lower = path.toLowerCase()
          const runPath = path.split('/').slice(0, -1).join('/')
          // Initialize record
          found[runPath] = found[runPath] || { runPath, artifactCounts: { resultJson: 0, csv: 0, otherJson: 0 } }
          if (lower.endsWith('.json') && lower.includes('ragas_enhanced_evaluation_results_')) {
            const file = await (handle as any).getFile()
            found[runPath].summaryJson = file
            found[runPath].artifactCounts!.resultJson += 1
          } else if (lower.endsWith('config.yaml')) {
            const file = await (handle as any).getFile()
            found[runPath].configYaml = file
          } else if (lower.endsWith('.csv')) {
            found[runPath].artifactCounts!.csv += 1
          } else if (lower.endsWith('.json')) {
            found[runPath].artifactCounts!.otherJson += 1
          }
        }
      }
      setRuns(Object.values(found))
  // Fast scan each detected summary JSON to estimate counts and coverage
  setScanning(true)
  // Dynamic import of the web worker for fast scan; cast to any to accommodate ?worker suffix without extra types
  const WorkerCtor = (await import('../workers/parser.worker.ts?worker')).default as unknown as {
    new (): Worker
  }
      await Promise.all(
        Object.values(found).map(async (r) => {
          if (!r.summaryJson) return
          try {
            const worker: Worker = new WorkerCtor()
            await new Promise<void>((resolve) => {
              worker.onmessage = (ev: MessageEvent<any>) => {
                const msg = ev.data
                if (msg.type === 'scan') {
                  setRuns((prev) => prev.map((x) => (x.runPath === r.runPath ? { ...x, totalItems: msg.total, metricsCoverage: msg.metricsCoverage } : x)))
                  worker.terminate()
                  resolve()
                }
              }
              worker.postMessage({ type: 'fast-scan', file: r.summaryJson, sample: 50 })
            })
          } catch {
            // ignore scan errors
          }
        })
      )
      setScanning(false)
    } catch (e: any) {
      if (e?.name === 'AbortError') return
      setError(e?.message ?? String(e))
    } finally {
      setBusy(false)
    }
  }

  const loadRun = async (r: DetectedRun) => {
    if (!r.summaryJson) return
    // lazy import worker-based parser already in RunLoader; reuse via direct Worker postMessage
    const mod = await import('./RunLoader')
    // re-use hidden method through a small shim to avoid duplication
    ;(mod as any).defaultParseSummary?.(r.summaryJson, setRunData, console.error)

    // Load thresholds overrides from config.yaml if present
    try {
      if (r.configYaml) {
        const text = await r.configYaml.text()
        const cfg = parseSimpleYAML(text)
        const th = extractThresholdsFromConfig(cfg)
        if (th) {
          // merge with existing thresholds
          const setThresholds = usePortalStore.getState().setThresholds
          const prev = usePortalStore.getState().thresholds
          setThresholds({ ...prev, ...th })
        }
      }
    } catch (e) {
      console.warn('Failed to parse config.yaml thresholds:', e)
    }
  }

  return (
    <div style={{ marginTop: 12 }}>
      <button onClick={onPickDir} disabled={busy}>{busy ? '掃描中…' : '選擇資料夾並掃描 runs'}</button>
      {error && <span style={{ color: 'crimson', marginLeft: 8 }}>{error}</span>}
      {!!runs.length && (
        <div style={{ marginTop: 8 }}>
      <div>偵測到 {runs.length} 個 run：{scanning ? '（快速掃描中…）' : ''}</div>
          <ul>
            {runs.map((r) => (
              <li key={r.runPath} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <code style={{ opacity: 0.8 }}>{r.runPath}</code>
                <span>· JSON: {r.summaryJson ? '✓' : '—'} · config.yaml: {r.configYaml ? '✓' : '—'} · items: {r.totalItems ?? '—'}</span>
                {r.artifactCounts && (
                  <span style={{ opacity: 0.8 }}>
                    · artifacts: results {r.artifactCounts.resultJson}, csv {r.artifactCounts.csv}, other json {r.artifactCounts.otherJson}
                  </span>
                )}
                {r.metricsCoverage && (
                  <span style={{ opacity: 0.8 }}>
                    · coverage: {Object.entries(r.metricsCoverage)
                      .slice(0, 3)
                      .map(([k, v]) => `${k}:${Math.round(v * 100)}%`)
                      .join(', ')}{Object.keys(r.metricsCoverage).length > 3 ? '…' : ''}
                  </span>
                )}
                <button onClick={() => loadRun(r)} disabled={!r.summaryJson}>載入</button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
