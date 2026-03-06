/**
 * telemetry/logEvent.test.ts  (TASK-081)
 *
 * Tests for the client-side telemetry batch logger.
 * Run with: vitest (or jest with ts-jest)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  logEvent,
  flushEvents,
  configureTelemetry,
  EVENT_SCHEMA_VERSION,
  type TelemetryEvent,
} from './logEvent'

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

const mockFetch = vi.fn().mockResolvedValue({ ok: true } as Response)
;(globalThis as any).fetch = mockFetch

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function capturedEvents(): TelemetryEvent[] {
  const calls = mockFetch.mock.calls
  if (calls.length === 0) return []
  const lastBody = JSON.parse(calls[calls.length - 1][1].body as string)
  return lastBody.events as TelemetryEvent[]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('logEvent – telemetry batch logger', () => {
  beforeEach(() => {
    mockFetch.mockClear()
    // Reset internal batch by flushing anything leftover
    flushEvents()
    mockFetch.mockClear()
    configureTelemetry({ endpoint: '/api/telemetry/events', batchSize: 20, flushIntervalMs: 5000 })
  })

  afterEach(() => {
    vi.clearAllTimers()
  })

  it('auto-enriches ts and schema_version', () => {
    logEvent({ type: 'ui.page.load' }, { immediate: true })
    const events = capturedEvents()
    expect(events).toHaveLength(1)
    expect(typeof events[0].ts).toBe('number')
    expect(events[0].schema_version).toBe(EVENT_SCHEMA_VERSION)
  })

  it('batches multiple events and flushes together', () => {
    logEvent({ type: 'ui.kg.render', payload: { nodeCount: 10 } })
    logEvent({ type: 'ui.ws.connect' })
    expect(mockFetch).not.toHaveBeenCalled()

    flushEvents()
    const events = capturedEvents()
    expect(events).toHaveLength(2)
    expect(events[0].type).toBe('ui.kg.render')
    expect(events[1].type).toBe('ui.ws.connect')
  })

  it('flushes immediately when immediate: true', () => {
    logEvent({ type: 'ui.eval.run' }, { immediate: true })
    expect(mockFetch).toHaveBeenCalledTimes(1)
    const events = capturedEvents()
    expect(events[0].type).toBe('ui.eval.run')
  })

  it('auto-flushes when batch reaches batchSize', () => {
    configureTelemetry({ batchSize: 3 })
    logEvent({ type: 'ui.kg.render' })
    logEvent({ type: 'ui.ws.connect' })
    expect(mockFetch).not.toHaveBeenCalled()

    logEvent({ type: 'ui.ws.disconnect' }) // 3rd event triggers flush
    expect(mockFetch).toHaveBeenCalledTimes(1)
    expect(capturedEvents()).toHaveLength(3)
  })

  it('clears the batch after flush', () => {
    logEvent({ type: 'ui.kg.render' }, { immediate: true })
    mockFetch.mockClear()

    flushEvents() // second flush — nothing to send
    expect(mockFetch).not.toHaveBeenCalled()
  })

  it('does not call fetch when batch is empty', () => {
    flushEvents()
    expect(mockFetch).not.toHaveBeenCalled()
  })

  it('supports all documented event types from ADR-005 taxonomy', () => {
    const eventTypes = [
      'ui.kg.render',
      'ui.ws.connect',
      'ui.ws.disconnect',
      'ui.kg.subgraph.fetch',
      'ui.kg.subgraph.error',
      'ui.eval.run',
      'ui.page.load',
    ] as const

    for (const type of eventTypes) {
      logEvent({ type }, { immediate: true })
    }

    // Each immediate call triggers fetch; verify total calls
    expect(mockFetch).toHaveBeenCalledTimes(eventTypes.length)
  })

  it('preserves explicit ts when provided', () => {
    const explicitTs = 1700000000000
    logEvent({ type: 'ui.page.load', ts: explicitTs }, { immediate: true })
    expect(capturedEvents()[0].ts).toBe(explicitTs)
  })

  it('carries payload through to the batch', () => {
    const payload = { nodeCount: 42, durationMs: 120 }
    logEvent({ type: 'ui.kg.render', payload }, { immediate: true })
    expect(capturedEvents()[0].payload).toEqual(payload)
  })
})
