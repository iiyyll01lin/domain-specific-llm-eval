import React from 'react'
import { useTranslation } from 'react-i18next'
import { usePortalStore } from '@/app/store/usePortalStore'
import { fetchKgJobs, fetchSubgraph } from './api'
import { usePollingResource } from './usePollingResource'
import type { KgJobItem, SubgraphResult } from './types'

const statusColor: Record<string, string> = {
  queued: '#0277bd',
  running: '#fbc02d',
  completed: '#2e7d32',
  error: '#c62828',
}

/** Feature flag: set window.ENABLE_KG_PANEL = true to show this panel. */
function isKgPanelEnabled(): boolean {
  if (typeof window === 'undefined') return false
  return Boolean((window as any).ENABLE_KG_PANEL)
}

const formatDateTime = (value?: string) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, {
    hour12: false,
    dateStyle: 'short',
    timeStyle: 'medium',
  }).format(date)
}

const formatNumber = (value?: number) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return value.toLocaleString()
}

function StatusBadge({ status, label }: { status: string; label: string }) {
  const color = statusColor[status] || '#37474f'
  return (
    <span style={{ background: color, color: '#fff', padding: '2px 8px', borderRadius: 8, fontSize: 12, textTransform: 'capitalize' }}>
      {label}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Subgraph sampling pill
// ---------------------------------------------------------------------------

const SamplingPill: React.FC<{ subgraph: SubgraphResult }> = ({ subgraph }) => {
  const { t } = useTranslation()
  const isSampled = subgraph.sampled === true
  const nodeCount = subgraph.nodes.length
  const totalCount = subgraph.total_nodes
  const hasTruncated = subgraph.nodes.some((n) => n.truncated)

  const label = isSampled
    ? t('lifecycle.kgSubgraph.sampledPill', {
        count: nodeCount,
        total: totalCount ?? '?',
      })
    : t('lifecycle.kgSubgraph.fullPill', { count: nodeCount })

  const pillStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    padding: '2px 8px',
    borderRadius: 12,
    fontSize: 11,
    fontWeight: 600,
    background: isSampled ? '#fff3e0' : '#e8f5e9',
    color: isSampled ? '#e65100' : '#2e7d32',
    border: `1px solid ${isSampled ? '#ffcc80' : '#a5d6a7'}`,
  }

  return (
    <span style={pillStyle} title={hasTruncated ? t('lifecycle.kgSubgraph.truncatedHint') : undefined}>
      {label}
      {hasTruncated && ' ✂'}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Subgraph overlay
// ---------------------------------------------------------------------------

interface SubgraphOverlayProps {
  subgraph: SubgraphResult
  onClose: () => void
}

const SubgraphOverlay: React.FC<SubgraphOverlayProps> = ({ subgraph, onClose }) => {
  const { t } = useTranslation()
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t('lifecycle.kgSubgraph.overlayTitle')}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.55)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        style={{
          background: '#1e2a32',
          borderRadius: 12,
          padding: 24,
          width: '90vw',
          maxWidth: 720,
          maxHeight: '80vh',
          overflowY: 'auto',
          color: '#e0e0e0',
        }}
      >
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <strong style={{ fontSize: 15 }}>
              {t('lifecycle.kgSubgraph.overlayTitle')}: <code style={{ fontSize: 12 }}>{subgraph.seed_node}</code>
            </strong>
            <SamplingPill subgraph={subgraph} />
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label={t('lifecycle.kgSubgraph.close')}
            style={{ background: 'none', border: 'none', color: '#90a4ae', cursor: 'pointer', fontSize: 18 }}
          >
            ✕
          </button>
        </header>

        <p style={{ fontSize: 12, color: '#78909c', margin: '0 0 12px' }}>
          {t('lifecycle.kgSubgraph.stats', {
            nodes: subgraph.nodes.length,
            edges: subgraph.edges.length,
            depth: subgraph.depth,
          })}
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Nodes list */}
          <div>
            <h4 style={{ margin: '0 0 8px', fontSize: 13 }}>{t('lifecycle.kgSubgraph.nodesHeading')}</h4>
            <ul style={{ margin: 0, padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 4 }}>
              {subgraph.nodes.map((node) => (
                <li
                  key={node.id}
                  style={{
                    padding: '4px 8px',
                    background: '#263238',
                    borderRadius: 6,
                    fontSize: 12,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <span style={{ fontWeight: 500 }}>{node.label}</span>
                  {node.type && (
                    <span style={{ fontSize: 10, color: '#78909c', marginLeft: 8 }}>{node.type}</span>
                  )}
                  {node.truncated && <span style={{ fontSize: 10, color: '#e65100' }}>✂ truncated</span>}
                </li>
              ))}
            </ul>
          </div>

          {/* Edges list */}
          <div>
            <h4 style={{ margin: '0 0 8px', fontSize: 13 }}>{t('lifecycle.kgSubgraph.edgesHeading')}</h4>
            <ul style={{ margin: 0, padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 4 }}>
              {subgraph.edges.map((edge, i) => (
                <li
                  key={`${edge.source}-${edge.target}-${i}`}
                  style={{
                    padding: '4px 8px',
                    background: '#263238',
                    borderRadius: 6,
                    fontSize: 11,
                    color: '#b0bec5',
                  }}
                >
                  <span>{edge.source}</span>
                  <span style={{ margin: '0 4px', color: '#78909c' }}>→</span>
                  <span style={{ color: '#90caf9', fontSize: 10 }}>{edge.relation ?? '—'}</span>
                  <span style={{ margin: '0 4px', color: '#78909c' }}>→</span>
                  <span>{edge.target}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Entity focus form (TASK-067)
// ---------------------------------------------------------------------------

interface SubgraphFocusFormProps {
  job: KgJobItem
}

const SubgraphFocusForm: React.FC<SubgraphFocusFormProps> = ({ job }) => {
  const { t } = useTranslation()
  const [seedNode, setSeedNode] = React.useState('')
  const [depth, setDepth] = React.useState(2)
  const [isLoading, setIsLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [subgraph, setSubgraph] = React.useState<SubgraphResult | null>(null)
  const abortRef = React.useRef<AbortController | null>(null)

  const handleSubmit = React.useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()
      const trimmed = seedNode.trim()
      if (!trimmed) return

      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      setIsLoading(true)
      setError(null)
      setSubgraph(null)

      try {
        const result = await fetchSubgraph({
          signal: controller.signal,
          kgId: job.kg_id,
          seedNode: trimmed,
          depth,
        })
        setSubgraph(result)
      } catch (err: unknown) {
        if (err instanceof Error && err.name === 'AbortError') return
        setError(err instanceof Error ? err.message : t('lifecycle.kgSubgraph.unknownError'))
      } finally {
        setIsLoading(false)
      }
    },
    [seedNode, depth, job.kg_id, t],
  )

  React.useEffect(() => {
    return () => { abortRef.current?.abort() }
  }, [])

  if (job.status !== 'completed') return null

  return (
    <div style={{ marginTop: 8, padding: '8px 12px', background: '#1e2a32', borderRadius: 8, fontSize: 13 }}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <label htmlFor={`seed-${job.kg_id}`} style={{ color: '#90a4ae', fontSize: 12 }}>
          {t('lifecycle.kgSubgraph.seedLabel')}
        </label>
        <input
          id={`seed-${job.kg_id}`}
          type="text"
          value={seedNode}
          onChange={(e) => setSeedNode(e.target.value)}
          placeholder={t('lifecycle.kgSubgraph.seedPlaceholder')}
          style={{
            flex: '1 1 160px',
            padding: '3px 8px',
            borderRadius: 4,
            border: '1px solid #455a64',
            background: '#263238',
            color: '#e0e0e0',
            fontSize: 12,
          }}
          aria-label={t('lifecycle.kgSubgraph.seedLabel')}
        />
        <label htmlFor={`depth-${job.kg_id}`} style={{ color: '#90a4ae', fontSize: 12 }}>
          {t('lifecycle.kgSubgraph.depthLabel')}
        </label>
        <input
          id={`depth-${job.kg_id}`}
          type="number"
          min={1}
          max={5}
          value={depth}
          onChange={(e) => setDepth(Number(e.target.value))}
          style={{ width: 48, padding: '3px 6px', borderRadius: 4, border: '1px solid #455a64', background: '#263238', color: '#e0e0e0', fontSize: 12 }}
          aria-label={t('lifecycle.kgSubgraph.depthLabel')}
        />
        <button
          type="submit"
          disabled={isLoading || !seedNode.trim()}
          style={{ padding: '4px 12px', borderRadius: 4, fontSize: 12, cursor: 'pointer' }}
        >
          {isLoading ? t('lifecycle.kgSubgraph.loading') : t('lifecycle.kgSubgraph.fetch')}
        </button>
      </form>

      {error && (
        <p style={{ margin: '4px 0 0', color: '#ef9a9a', fontSize: 12 }} role="alert">
          {error}
        </p>
      )}

      {subgraph && (
        <SubgraphOverlay subgraph={subgraph} onClose={() => setSubgraph(null)} />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// KgPanel main component
// ---------------------------------------------------------------------------

export const KgPanel: React.FC = () => {
  const { t } = useTranslation()
  const kgJobs = usePortalStore((s) => s.kgJobs)
  const setKgJobs = usePortalStore((s) => s.setKgJobs)

  const { data, isLoading, error, lastUpdated, refetch } = usePollingResource<KgJobItem[]>(
    kgJobs,
    React.useCallback((signal) => fetchKgJobs({ signal }), []),
  )

  React.useEffect(() => {
    setKgJobs(data)
  }, [data, setKgJobs])

  const lastUpdatedLabel = lastUpdated
    ? t('lifecycle.shared.lastUpdated', {
        time: formatDateTime(new Date(lastUpdated).toISOString()),
      })
    : t('lifecycle.shared.awaiting')
  const translateStatus = React.useCallback(
    (status: string) => t(`lifecycle.kgJobs.status.${status}`, { defaultValue: status.replace(/_/g, ' ') }),
    [t],
  )

  if (!isKgPanelEnabled()) {
    return (
      <section
        style={{ border: '1px solid #263238', borderRadius: 12, padding: 16, color: '#78909c' }}
      >
        <h2 style={{ margin: 0 }}>{t('lifecycle.kgJobs.title')}</h2>
        <p style={{ marginTop: 8 }}>{t('lifecycle.kgJobs.subtitle')}</p>
        <p style={{ fontStyle: 'italic', fontSize: 12 }}>
          KG panel is feature-flagged. Set <code>window.ENABLE_KG_PANEL = true</code> to enable.
        </p>
      </section>
    )
  }

  return (
    <section style={{ border: '1px solid #263238', borderRadius: 12, padding: 16 }}>
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 12,
        }}
      >
        <div>
          <h2 style={{ margin: 0 }}>{t('lifecycle.kgJobs.title')}</h2>
          <p style={{ margin: '4px 0 0', color: '#78909c', fontSize: 12 }}>
            {t('lifecycle.kgJobs.subtitle')}
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button type="button" onClick={refetch} disabled={isLoading}>
            {t('lifecycle.shared.refresh')}
          </button>
          <span style={{ fontSize: 12, color: '#90a4ae' }}>{lastUpdatedLabel}</span>
        </div>
      </header>

      {error && (
        <div
          style={{
            background: '#ffebee',
            color: '#c62828',
            padding: 12,
            borderRadius: 8,
            marginBottom: 12,
          }}
        >
          {t('lifecycle.kgJobs.error')} {error.message}
        </div>
      )}

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid #455a64' }}>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.kgId')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.status')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.docCount')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.nodeCount')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.edgeCount')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.created')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.updated')}</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && !kgJobs.length && (
              <tr>
                <td
                  colSpan={7}
                  style={{ padding: 12, textAlign: 'center', color: '#78909c' }}
                >
                  {t('lifecycle.kgJobs.loading')}
                </td>
              </tr>
            )}
            {!isLoading && !kgJobs.length && (
              <tr>
                <td
                  colSpan={7}
                  style={{ padding: 12, textAlign: 'center', color: '#78909c' }}
                >
                  {t('lifecycle.kgJobs.empty')}
                </td>
              </tr>
            )}
            {kgJobs.map((job, index) => (
              <React.Fragment key={`${job.kg_id}-${index}`}>
                <tr style={{ borderBottom: '1px solid rgba(69,90,100,0.4)' }}>
                  <td
                    style={{
                      padding: '8px 4px',
                      fontFamily: 'monospace',
                      fontSize: 12,
                      maxWidth: 180,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                    title={job.kg_id}
                  >
                    {job.kg_id}
                  </td>
                  <td style={{ padding: '8px 4px' }}>
                      <StatusBadge status={job.status} label={translateStatus(job.status)} />
                  </td>
                  <td style={{ padding: '8px 4px' }}>{formatNumber(job.doc_count)}</td>
                  <td style={{ padding: '8px 4px' }}>{formatNumber(job.node_count)}</td>
                  <td style={{ padding: '8px 4px' }}>{formatNumber(job.edge_count)}</td>
                  <td style={{ padding: '8px 4px', fontSize: 12 }}>
                    {formatDateTime(job.created_at)}
                  </td>
                  <td style={{ padding: '8px 4px', fontSize: 12 }}>
                    {formatDateTime(job.updated_at)}
                  </td>
                </tr>
                {/* TASK-067: Entity focus form shown below each completed KG job */}
                <tr>
                  <td colSpan={7} style={{ padding: 0 }}>
                    <SubgraphFocusForm job={job} />
                  </td>
                </tr>
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
