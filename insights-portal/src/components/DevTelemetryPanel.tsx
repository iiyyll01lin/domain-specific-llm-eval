import React from 'react'

export type TimingSample = {
  at: number
  filterMs: number
  sampleMs: number
  aggregateMs: number
  total: number
}

interface Props {
  samples: TimingSample[]
  coalesceMs: number
  onCoalesceChange: (ms: number) => void
}

export const DevTelemetryPanel: React.FC<Props> = ({ samples, coalesceMs, onCoalesceChange }) => {
  const avg = (key: keyof TimingSample) => {
    if (!samples.length) return 0
    return samples.reduce((s, x) => s + (x[key] as number), 0) / samples.length
  }
  return (
    <details style={{ marginTop: 8 }}>
      <summary>Dev: Worker timings</summary>
      <div style={{ fontSize: 12, marginTop: 6 }}>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <span>samples: {samples.length}</span>
          <span>avg filter: {avg('filterMs').toFixed(1)}ms</span>
          <span>avg sample: {avg('sampleMs').toFixed(1)}ms</span>
          <span>avg aggregate: {avg('aggregateMs').toFixed(1)}ms</span>
        </div>
        <div style={{ marginTop: 6 }}>
          <label>
            Coalesce window (ms):
            <input
              type="number"
              value={coalesceMs}
              onChange={(e) => onCoalesceChange(Math.max(0, Number(e.target.value) || 0))}
              style={{ width: 90, marginLeft: 6 }}
            />
          </label>
        </div>
        <div style={{ maxHeight: 160, overflow: 'auto', marginTop: 8, border: '1px solid var(--border)', padding: 6 }}>
          <table style={{ width: '100%', fontSize: 12 }}>
            <thead>
              <tr>
                <th align="left">time</th>
                <th align="right">filter</th>
                <th align="right">sample</th>
                <th align="right">aggregate</th>
                <th align="right">total filtered</th>
              </tr>
            </thead>
            <tbody>
              {samples.slice(-50).map((s, i) => (
                <tr key={i}>
                  <td>{new Date(s.at).toLocaleTimeString()}</td>
                  <td align="right">{s.filterMs}ms</td>
                  <td align="right">{s.sampleMs}ms</td>
                  <td align="right">{s.aggregateMs}ms</td>
                  <td align="right">{s.total}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </details>
  )
}

export default DevTelemetryPanel
