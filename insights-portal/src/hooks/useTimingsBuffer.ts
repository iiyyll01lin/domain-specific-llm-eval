import React from 'react'

export type TimingEntry = { at: number; filterMs: number; sampleMs: number; aggregateMs: number; total: number }

export function useTimingsBuffer(limit = 200) {
  const [buf, setBuf] = React.useState<TimingEntry[]>([])
  const push = React.useCallback((e: TimingEntry) => {
    setBuf((b) => b.concat([e]).slice(-limit))
  }, [limit])
  const clear = React.useCallback(() => setBuf([]), [])
  return { buf, push, clear }
}
