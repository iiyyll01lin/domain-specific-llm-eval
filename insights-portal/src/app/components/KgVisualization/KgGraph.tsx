import React from 'react'
import { useFeatureFlags } from '@/core/featureFlags'

export interface GraphNode { id: string; label: string; degree?: number; entityType?: string }
export interface GraphEdge { id: string; source: string; target: string; kind?: string; weight?: number }
export interface KnowledgeGraphSummary { kg_id: string; node_count: number; relationship_count: number; degree_histogram?: number[]; top_entities?: { entity: string; degree: number }[] }

export interface KgGraphProps {
  summary: KnowledgeGraphSummary
  fetchFullGraph?: () => Promise<{ nodes: GraphNode[]; edges: GraphEdge[] }>
  height?: number
  theme?: 'light' | 'dark'
}

// Lazy cytoscape wrapper (separate chunk) loaded only when user enables full graph view.
const CytoscapeGraph = React.lazy(() => import('./internal/CytoscapeGraph'))

export const KgGraph: React.FC<KgGraphProps> = ({ summary, fetchFullGraph, height = 420 }) => {
  const { kgVisualization } = useFeatureFlags()
  const [mode, setMode] = React.useState<'summary'|'loading'|'graph'|'error'>('summary')
  const [graphData, setGraphData] = React.useState<{nodes: GraphNode[]; edges: GraphEdge[]}|null>(null)
  const [error, setError] = React.useState<string|undefined>()

  const enableGraph = React.useCallback(async () => {
    if (!kgVisualization) return
    if (!fetchFullGraph) { setMode('graph'); return }
    setMode('loading')
    try {
      const { nodes, edges } = await fetchFullGraph()
      // Soft cap to avoid huge initial render
      const cappedNodes = nodes.slice(0, 500)
      const idSet = new Set(cappedNodes.map(n => n.id))
      const cappedEdges = edges.filter(e => idSet.has(e.source) && idSet.has(e.target))
      setGraphData({ nodes: cappedNodes, edges: cappedEdges })
      setMode('graph')
    } catch (e:any) {
      setError(e?.message || 'Failed to load graph')
      setMode('error')
    }
  }, [kgVisualization, fetchFullGraph])

  const histogram = summary.degree_histogram || []
  const topEntities = summary.top_entities || []

  if (!kgVisualization) {
    return (
      <div style={{ padding: 12 }}>
        <strong>KG Visualization disabled.</strong>
        <div>Nodes: {summary.node_count} · Relationships: {summary.relationship_count}</div>
      </div>
    )
  }

  return (
    <div style={{ border: '1px solid var(--border-color,#ccc)', borderRadius: 6, padding: 12 }} data-testid="kg-container">
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0 }}>Knowledge Graph (Summary)</h3>
        {mode === 'summary' && (
          <button onClick={enableGraph} disabled={mode==='loading'} data-testid="kg-enable-btn">Enable Graph</button>
        )}
      </header>
      <div style={{ fontSize: 12, opacity: 0.8, marginBottom: 8 }}>Nodes: {summary.node_count} · Relationships: {summary.relationship_count}</div>
      {mode === 'summary' && (
        <div style={{ display: 'grid', gap: 12, gridTemplateColumns: '1fr 1fr' }}>
          <div>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Degree Histogram</div>
            <div data-testid="kg-summary-hist" style={{ display: 'flex', gap: 2, alignItems: 'flex-end', height: 80 }}>
              {histogram.slice(0, 50).map((v, i) => (
                <div key={i} style={{ background: '#4a90e2', width: 4, height: Math.max(2, Math.min(70, v)), transition: 'height .3s' }} title={`bin ${i}: ${v}`}></div>
              ))}
              {histogram.length === 0 && <div style={{ fontStyle: 'italic' }}>No histogram data</div>}
            </div>
          </div>
          <div>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Top Entities</div>
            <ol data-testid="kg-top-entities" style={{ margin: 0, paddingLeft: 16, maxHeight: 80, overflow: 'auto' }}>
              {topEntities.slice(0, 10).map(te => (
                <li key={te.entity}>{te.entity} ({te.degree})</li>
              ))}
              {topEntities.length === 0 && <li style={{ fontStyle: 'italic' }}>No entities</li>}
            </ol>
          </div>
        </div>
      )}
      {mode === 'loading' && <div style={{ padding: 24 }}>Loading graph…</div>}
      {mode === 'error' && <div style={{ color: 'var(--danger,#b00)' }}>Error: {error} <button onClick={enableGraph}>Retry</button></div>}
      {mode === 'graph' && (
        <React.Suspense fallback={<div style={{ padding: 24 }}>Rendering graph…</div>}>
          <CytoscapeGraph data={graphData} height={height} summary={summary} />
        </React.Suspense>
      )}
    </div>
  )
}

export default KgGraph
