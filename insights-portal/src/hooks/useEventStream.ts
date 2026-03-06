/**
 * TASK-072: useEventStream
 * React hook that manages a WebSocket connection to the WS gateway's
 * /ui/events endpoint, with:
 *   - automatic reconnect with exponential backoff
 *   - heartbeat monitoring (TASK-070/071)
 *   - gap detection triggering REST resync (TASK-071)
 *   - progressive downgrade to polling after N failures (TASK-073)
 *
 * TASK-073: Progressive Downgrade Logic
 * After MAX_CONSECUTIVE_FAILURES consecutive WS failures the hook
 * stops attempting new connections and fires onDowngrade(). It
 * automatically attempts to upgrade back after DOWNGRADE_DURATION_MS.
 */
import React from 'react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type WsEnvelopeType = 'welcome' | 'heartbeat' | 'data' | 'error'

export interface WsEnvelope {
  type: WsEnvelopeType
  seq: number
  ts: number
  topic?: string
  payload?: Record<string, unknown>
}

export type EventStreamStatus =
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'downgraded'
  | 'closed'

export interface UseEventStreamOptions {
  /** WS gateway base URL, e.g. "ws://localhost:8008" */
  url: string
  /** Comma-separated topics to subscribe to — passed as ?topics= query param */
  topics?: string
  /** Called when a data envelope arrives for a subscribed topic */
  onMessage?: (envelope: WsEnvelope) => void
  /** Called when the hook has detected a seq gap and needs REST resync */
  onResyncNeeded?: (missingSeqs: number[]) => void
  /** Called when the hook enters downgraded (polling) mode */
  onDowngrade?: () => void
  /** Called when the hook exits downgraded mode and reconnects */
  onUpgrade?: () => void
  /** Disable the hook entirely (e.g. feature flag off) */
  enabled?: boolean
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const INITIAL_BACKOFF_MS = 200
const MAX_BACKOFF_MS = 30_000
const BACKOFF_MULTIPLIER = 2

/** Number of consecutive connection failures before downgrading */
const MAX_CONSECUTIVE_FAILURES = 5

/** How long to stay in downgraded mode before attempting upgrade (2 min) */
const DOWNGRADE_DURATION_MS = 2 * 60 * 1_000

/** Seconds before a heartbeat is considered missed (slightly over the 15s server interval) */
const HEARTBEAT_TIMEOUT_MS = 20_000

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useEventStream(options: UseEventStreamOptions) {
  const {
    url,
    topics = '',
    onMessage,
    onResyncNeeded,
    onDowngrade,
    onUpgrade,
    enabled = true,
  } = options

  const [status, setStatus] = React.useState<EventStreamStatus>('closed')

  // Refs — mutated within effect callbacks without triggering re-renders
  const wsRef = React.useRef<WebSocket | null>(null)
  const backoffRef = React.useRef(INITIAL_BACKOFF_MS)
  const consecutiveFailuresRef = React.useRef(0)
  const receivedSeqsRef = React.useRef<number[]>([])
  const heartbeatTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const downgradeTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const isDowngradedRef = React.useRef(false)

  // Stable callbacks
  const onMessageRef = React.useRef(onMessage)
  const onResyncRef = React.useRef(onResyncNeeded)
  const onDowngradeRef = React.useRef(onDowngrade)
  const onUpgradeRef = React.useRef(onUpgrade)
  React.useEffect(() => { onMessageRef.current = onMessage }, [onMessage])
  React.useEffect(() => { onResyncRef.current = onResyncNeeded }, [onResyncNeeded])
  React.useEffect(() => { onDowngradeRef.current = onDowngrade }, [onDowngrade])
  React.useEffect(() => { onUpgradeRef.current = onUpgrade }, [onUpgrade])

  const clearHeartbeatTimer = React.useCallback(() => {
    if (heartbeatTimerRef.current !== null) {
      clearTimeout(heartbeatTimerRef.current)
      heartbeatTimerRef.current = null
    }
  }, [])

  const resetHeartbeatTimer = React.useCallback(() => {
    clearHeartbeatTimer()
    heartbeatTimerRef.current = setTimeout(() => {
      // Heartbeat timeout — treat as connection failure
      wsRef.current?.close()
    }, HEARTBEAT_TIMEOUT_MS)
  }, [clearHeartbeatTimer])

  /** Check received sequence numbers for gaps and fire resync if needed */
  const checkForGaps = React.useCallback((newSeq: number) => {
    const seqs = receivedSeqsRef.current
    if (seqs.length > 0) {
      const last = seqs[seqs.length - 1]
      if (newSeq - last > 1) {
        const missing: number[] = []
        for (let s = last + 1; s < newSeq; s++) missing.push(s)
        onResyncRef.current?.(missing)
      }
    }
    receivedSeqsRef.current = [...seqs.slice(-999), newSeq] // cap buffer
  }, [])

  // -------------------------------------------------------------------------
  // TASK-073: enter/exit downgrade
  // -------------------------------------------------------------------------

  const enterDowngrade = React.useCallback(() => {
    isDowngradedRef.current = true
    setStatus('downgraded')
    onDowngradeRef.current?.()

    // Schedule upgrade attempt
    downgradeTimerRef.current = setTimeout(() => {
      isDowngradedRef.current = false
      consecutiveFailuresRef.current = 0
      backoffRef.current = INITIAL_BACKOFF_MS
      onUpgradeRef.current?.()
      // The main effect will re-run because isDowngraded flipped
      connect() // eslint-disable-line @typescript-eslint/no-use-before-define
    }, DOWNGRADE_DURATION_MS)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // connect / disconnect
  // -------------------------------------------------------------------------

  const connect = React.useCallback(() => {
    if (!enabled || isDowngradedRef.current) return

    setStatus('connecting')

    const wsUrl = topics
      ? `${url}/ui/events?topics=${encodeURIComponent(topics)}`
      : `${url}/ui/events`

    let ws: WebSocket
    try {
      ws = new WebSocket(wsUrl)
    } catch {
      // URL parsing failure — treat as permanent failure
      consecutiveFailuresRef.current++
      if (consecutiveFailuresRef.current >= MAX_CONSECUTIVE_FAILURES) {
        enterDowngrade()
      }
      return
    }
    wsRef.current = ws

    ws.onopen = () => {
      consecutiveFailuresRef.current = 0
      backoffRef.current = INITIAL_BACKOFF_MS
      setStatus('connected')
      resetHeartbeatTimer()
    }

    ws.onmessage = (event: MessageEvent) => {
      let envelope: WsEnvelope
      try {
        envelope = JSON.parse(event.data as string) as WsEnvelope
      } catch {
        return
      }

      // Reset heartbeat timer on ANY message
      resetHeartbeatTimer()

      if (envelope.type === 'heartbeat') return

      if (envelope.type === 'data' || envelope.type === 'welcome') {
        checkForGaps(envelope.seq)
        if (envelope.type === 'data') {
          onMessageRef.current?.(envelope)
        }
      }

      if (envelope.type === 'error') {
        ws.close(4000)
      }
    }

    ws.onerror = () => {
      // onerror is always followed by onclose; let onclose handle reconnect
    }

    ws.onclose = () => {
      clearHeartbeatTimer()
      wsRef.current = null

      if (!enabled || isDowngradedRef.current) {
        setStatus('closed')
        return
      }

      consecutiveFailuresRef.current++
      if (consecutiveFailuresRef.current >= MAX_CONSECUTIVE_FAILURES) {
        enterDowngrade()
        return
      }

      // Exponential backoff reconnect
      const delay = Math.min(backoffRef.current, MAX_BACKOFF_MS)
      backoffRef.current = Math.min(backoffRef.current * BACKOFF_MULTIPLIER, MAX_BACKOFF_MS)
      setStatus('reconnecting')

      reconnectTimerRef.current = setTimeout(() => {
        connect()
      }, delay)
    }
  }, [url, topics, enabled, resetHeartbeatTimer, clearHeartbeatTimer, checkForGaps, enterDowngrade])

  // -------------------------------------------------------------------------
  // Effect — start/stop on mount/unmount or option changes
  // -------------------------------------------------------------------------

  React.useEffect(() => {
    if (!enabled) {
      wsRef.current?.close()
      setStatus('closed')
      return
    }

    connect()

    return () => {
      // Cleanup on unmount or re-run
      if (reconnectTimerRef.current !== null) clearTimeout(reconnectTimerRef.current)
      if (downgradeTimerRef.current !== null) clearTimeout(downgradeTimerRef.current)
      clearHeartbeatTimer()
      wsRef.current?.close()
      wsRef.current = null
      setStatus('closed')
    }
  }, [enabled, url, topics]) // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // Manual close
  // -------------------------------------------------------------------------

  const close = React.useCallback(() => {
    if (reconnectTimerRef.current !== null) clearTimeout(reconnectTimerRef.current)
    if (downgradeTimerRef.current !== null) clearTimeout(downgradeTimerRef.current)
    clearHeartbeatTimer()
    wsRef.current?.close()
    wsRef.current = null
    setStatus('closed')
  }, [clearHeartbeatTimer])

  return { status, close }
}
