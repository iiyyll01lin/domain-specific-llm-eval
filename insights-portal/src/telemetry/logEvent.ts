/**
 * telemetry/logEvent.ts
 *
 * Client-side telemetry helper for the Insights Portal (TASK-081).
 * Batches high-frequency events and flushes them to the telemetry endpoint
 * to avoid overwhelming the backend with individual requests.
 *
 * Event taxonomy follows ADR-005 (telemetry-taxonomy.md).
 * Event schema versioning follows ADR-006 (event-schema-versioning.md).
 *
 * Usage:
 *   import { logEvent } from '@/telemetry/logEvent'
 *   logEvent({ type: 'ui.kg.render', payload: { nodeCount: 42, durationMs: 120 } })
 */

/** Schema version — increment on breaking payload changes (per ADR-006). */
export const EVENT_SCHEMA_VERSION = 1

/** Supported event types from ADR-005 taxonomy. */
export type EventType =
  | 'ui.kg.render'
  | 'ui.ws.connect'
  | 'ui.ws.disconnect'
  | 'ui.kg.subgraph.fetch'
  | 'ui.kg.subgraph.error'
  | 'ui.eval.run'
  | 'ui.page.load'

export interface TelemetryEvent {
  /** ADR-005 event type identifier. */
  type: EventType
  /** Unix timestamp in milliseconds (auto-set if omitted). */
  ts?: number
  /** ADR-006 schema version (auto-set). */
  schema_version?: number
  /** Event-specific payload. */
  payload?: Record<string, unknown>
}

export interface LogEventOptions {
  /** Flush immediately instead of batching. Default: false. */
  immediate?: boolean
}

// ---------------------------------------------------------------------------
// Internal batching state
// ---------------------------------------------------------------------------

const DEFAULT_BATCH_SIZE = 20
const DEFAULT_FLUSH_INTERVAL_MS = 5_000

let _batch: TelemetryEvent[] = []
let _flushTimer: ReturnType<typeof setTimeout> | null = null
let _endpoint = '/api/telemetry/events'
let _batchSize = DEFAULT_BATCH_SIZE
let _flushIntervalMs = DEFAULT_FLUSH_INTERVAL_MS

/** Configure the telemetry logger (call once during app bootstrap). */
export function configureTelemetry(opts: {
  endpoint?: string
  batchSize?: number
  flushIntervalMs?: number
}): void {
  if (opts.endpoint !== undefined) _endpoint = opts.endpoint
  if (opts.batchSize !== undefined) _batchSize = opts.batchSize
  if (opts.flushIntervalMs !== undefined) _flushIntervalMs = opts.flushIntervalMs
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Record a telemetry event. The event is added to the internal batch and
 * will be flushed either when the batch is full, the flush timer fires, or
 * `flushEvents()` is called explicitly.
 */
export function logEvent(event: TelemetryEvent, opts: LogEventOptions = {}): void {
  const enriched: TelemetryEvent = {
    ...event,
    ts: event.ts ?? Date.now(),
    schema_version: EVENT_SCHEMA_VERSION,
  }

  _batch.push(enriched)

  if (opts.immediate || _batch.length >= _batchSize) {
    flushEvents()
    return
  }

  if (_flushTimer === null) {
    _flushTimer = setTimeout(flushEvents, _flushIntervalMs)
  }
}

/**
 * Flush all buffered events to the telemetry endpoint immediately.
 * Clears the batch and resets the flush timer.
 */
export function flushEvents(): void {
  if (_batch.length === 0) return

  const toSend = _batch.slice()
  _batch = []

  if (_flushTimer !== null) {
    clearTimeout(_flushTimer)
    _flushTimer = null
  }

  // Fire-and-forget — telemetry MUST NOT block user interactions
  _sendBatch(toSend).catch(() => {
    // Silently discard; telemetry failures should never surface to users
  })
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

async function _sendBatch(events: TelemetryEvent[]): Promise<void> {
  if (typeof fetch === 'undefined') return // SSR / test environments

  await fetch(_endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ events }),
    // Keep telemetry alive even if the page is navigating away
    keepalive: true,
  })
}

// ---------------------------------------------------------------------------
// Page-unload flush — ensures buffered events are not lost on tab close
// ---------------------------------------------------------------------------

if (typeof window !== 'undefined') {
  window.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      flushEvents()
    }
  })
}
