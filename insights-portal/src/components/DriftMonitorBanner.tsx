/**
 * DriftMonitorBanner
 *
 * Polls GET /api/v1/drift-status on the webhook daemon (port 8008 by default)
 * every 5 minutes and renders a status banner above the Executive Overview.
 *
 * Styling is intentionally conditional:
 *   HEALTHY           → subtle green, collapsed by default
 *   INSUFFICIENT_DATA → grey, collapsed
 *   WARNING           → amber, expanded with metric detail
 *   DRIFTING          → red,   expanded with metric detail + action CTA
 *   UNAVAILABLE       → grey, collapsed (service unreachable)
 */
import React, { useEffect, useState } from 'react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type DriftStatus =
  | 'HEALTHY'
  | 'WARNING'
  | 'DRIFTING'
  | 'INSUFFICIENT_DATA'
  | 'PENDING'
  | 'UNAVAILABLE'

export interface MetricDriftDetail {
  metric: string
  baseline_mean: number
  recent_mean: number
  baseline_std: number
  z_score: number
  delta_pct: number
  flagged: boolean
}

export interface DriftStatusPayload {
  status: DriftStatus
  checked_at?: string
  message?: string
  baseline_window_size?: number
  recent_window_size?: number
  metrics?: Record<string, MetricDriftDetail>
}

// ---------------------------------------------------------------------------
// Visual config per status
// ---------------------------------------------------------------------------

interface StatusVisual {
  icon: string
  label: string
  bg: string
  border: string
  color: string
  /** When true the detail section is hidden by default (toggle with ▼) */
  collapsedByDefault: boolean
}

const STATUS_VISUAL: Record<DriftStatus, StatusVisual> = {
  HEALTHY: {
    icon: '✅',
    label: 'Healthy',
    bg: 'rgba(34, 197, 94, 0.07)',
    border: '1px solid rgba(34, 197, 94, 0.30)',
    color: 'var(--status-ok, #16a34a)',
    collapsedByDefault: true,
  },
  WARNING: {
    icon: '⚠️',
    label: 'Warning',
    bg: 'rgba(234, 179, 8, 0.10)',
    border: '1px solid rgba(234, 179, 8, 0.55)',
    color: 'var(--status-warn, #ca8a04)',
    collapsedByDefault: false,
  },
  DRIFTING: {
    icon: '🚨',
    label: 'Drifting',
    bg: 'rgba(220, 38, 38, 0.09)',
    border: '1px solid rgba(220, 38, 38, 0.55)',
    color: 'var(--status-error, #dc2626)',
    collapsedByDefault: false,
  },
  INSUFFICIENT_DATA: {
    icon: '📊',
    label: 'Insufficient Data',
    bg: 'rgba(107, 114, 128, 0.07)',
    border: '1px solid rgba(107, 114, 128, 0.28)',
    color: 'var(--text-muted, #6b7280)',
    collapsedByDefault: true,
  },
  PENDING: {
    icon: '⏳',
    label: 'Pending',
    bg: 'rgba(107, 114, 128, 0.07)',
    border: '1px solid rgba(107, 114, 128, 0.28)',
    color: 'var(--text-muted, #6b7280)',
    collapsedByDefault: true,
  },
  UNAVAILABLE: {
    icon: '—',
    label: 'Unavailable',
    bg: 'rgba(107, 114, 128, 0.05)',
    border: '1px solid rgba(107, 114, 128, 0.20)',
    color: 'var(--text-muted, #6b7280)',
    collapsedByDefault: true,
  },
}

const METRIC_LABELS: Record<string, string> = {
  entity_overlap: 'Entity Overlap (Sₑ)',
  structural_connectivity: 'Structural Connectivity (Sᶜ)',
  hub_noise_penalty: 'Hub Noise (Pₕ)',
}

// Resolve the webhook base URL from Vite env or fall back to localhost:8008.
const WEBHOOK_BASE: string = (() => {
  const env =
    typeof import.meta !== 'undefined'
      ? (import.meta as any)?.env?.VITE_WEBHOOK_BASE
      : undefined
  return ((env as string | undefined) || 'http://localhost:8008').replace(/\/$/, '')
})()

const POLL_INTERVAL_MS = 5 * 60 * 1000 // 5 minutes

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function DriftMonitorBanner() {
  const [drift, setDrift] = useState<DriftStatusPayload | null>(null)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    let canceled = false

    const doFetch = async () => {
      try {
        const res = await fetch(`${WEBHOOK_BASE}/api/v1/drift-status`, {
          headers: { Accept: 'application/json' },
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data: DriftStatusPayload = await res.json()
        if (!canceled) {
          setDrift(data)
          // Auto-expand visually important states when first loaded
          const visual = STATUS_VISUAL[data.status]
          if (visual && !visual.collapsedByDefault) setExpanded(true)
        }
      } catch {
        if (!canceled) setDrift({ status: 'UNAVAILABLE' })
      }
    }

    doFetch()
    const interval = setInterval(doFetch, POLL_INTERVAL_MS)
    return () => {
      canceled = true
      clearInterval(interval)
    }
  }, [])

  if (!drift) return null

  const visual = STATUS_VISUAL[drift.status] ?? STATUS_VISUAL.UNAVAILABLE
  const isCollapsible = visual.collapsedByDefault
  const showDetails = !isCollapsible || expanded

  const flaggedMetrics = drift.metrics
    ? Object.values(drift.metrics).filter((m) => m.flagged)
    : []

  const isDrift = drift.status === 'DRIFTING' || drift.status === 'WARNING'

  return (
    <div
      role="status"
      aria-label="Drift Monitor"
      data-drift-status={drift.status}
      style={{
        margin: '8px 0 14px',
        padding: '8px 12px',
        borderRadius: 8,
        border: visual.border,
        background: visual.bg,
        color: visual.color,
        fontSize: 13,
        transition: 'background 0.25s, border-color 0.25s',
      }}
    >
      {/* ── Header row ─────────────────────────────────────────────────── */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          flexWrap: 'wrap',
        }}
      >
        <span style={{ fontWeight: 600, flexShrink: 0 }}>
          {visual.icon}{' '}
          <span style={{ opacity: 0.85 }}>Drift Monitor</span>
          {' — '}
          <span aria-label={`drift-status-${drift.status}`}>{visual.label}</span>
        </span>

        {drift.checked_at && (
          <span style={{ fontSize: 11, opacity: 0.65, marginLeft: 'auto' }}>
            checked {new Date(drift.checked_at).toLocaleString()}
          </span>
        )}

        {isCollapsible && (
          <button
            onClick={() => setExpanded((e) => !e)}
            aria-expanded={expanded}
            aria-label="toggle drift details"
            style={{
              fontSize: 11,
              padding: '0 4px',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'inherit',
              opacity: 0.6,
              marginLeft: drift.checked_at ? 0 : 'auto',
            }}
          >
            {expanded ? '▲' : '▼'}
          </button>
        )}
      </div>

      {/* ── Detail section ─────────────────────────────────────────────── */}
      {showDetails && (
        <div style={{ marginTop: 8 }}>
          {drift.message && (
            <div style={{ marginBottom: flaggedMetrics.length > 0 ? 6 : 0, opacity: 0.9 }}>
              {drift.message}
            </div>
          )}

          {flaggedMetrics.length > 0 && (
            <ul
              style={{
                margin: '4px 0 0 16px',
                padding: 0,
                listStyle: 'disc',
                lineHeight: 1.7,
              }}
            >
              {flaggedMetrics.map((m) => {
                const sign = m.delta_pct >= 0 ? '+' : ''
                return (
                  <li key={m.metric}>
                    <strong>{METRIC_LABELS[m.metric] ?? m.metric}</strong>:{' '}
                    recent&nbsp;
                    <code>{m.recent_mean.toFixed(3)}</code> vs baseline&nbsp;
                    <code>{m.baseline_mean.toFixed(3)}</code>{' '}
                    <span style={{ opacity: 0.75 }}>
                      ({sign}{m.delta_pct.toFixed(1)}%,&nbsp;z={m.z_score.toFixed(2)})
                    </span>
                  </li>
                )
              })}
            </ul>
          )}

          {/* Action Required CTA for WARNING / DRIFTING */}
          {isDrift && (
            <div
              role="alert"
              style={{
                marginTop: 10,
                padding: '7px 10px',
                borderRadius: 6,
                background: 'rgba(0,0,0,0.07)',
                fontWeight: 500,
                lineHeight: 1.5,
              }}
            >
              ⚡ <strong>Action Required:</strong> Analyze failing queries
              and inject new domain documents into the Knowledge Graph.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default DriftMonitorBanner
