import React from 'react'
import { getLifecycleConfig } from './config'

export interface PollingState<T> {
  data: T
  isLoading: boolean
  error: Error | null
  lastUpdated?: number
}

export function usePollingResource<T>(
  initial: T,
  fetcher: (signal: AbortSignal) => Promise<T>,
  options?: { enabled?: boolean }
): PollingState<T> & { refetch: () => void } {
  const { pollIntervalMs } = getLifecycleConfig()
  const [state, setState] = React.useState<PollingState<T>>({ data: initial, isLoading: true, error: null })
  const enabled = options?.enabled ?? true
  const [nonce, setNonce] = React.useState(0)

  React.useEffect(() => {
    if (!enabled) return undefined
    let disposed = false
    let timeoutId: ReturnType<typeof setTimeout> | undefined

    const execute = async () => {
      const abortController = new AbortController()
      try {
        if (!disposed) {
          setState((prev: PollingState<T>) => ({ ...prev, isLoading: true, error: null }))
        }
        const result = await fetcher(abortController.signal)
        if (!disposed) {
          setState({ data: result, isLoading: false, error: null, lastUpdated: Date.now() })
        }
      } catch (err) {
        if (!disposed) {
          setState((prev: PollingState<T>) => ({ ...prev, isLoading: false, error: err instanceof Error ? err : new Error(String(err)) }))
        }
      } finally {
        abortController.abort()
        if (!disposed) {
          timeoutId = setTimeout(execute, pollIntervalMs)
        }
      }
    }

    execute()

    return () => {
      disposed = true
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [enabled, pollIntervalMs, fetcher, nonce])

  const refetch = React.useCallback(() => setNonce((n: number) => n + 1), [])

  return { ...state, refetch }
}
