import React from 'react'
import type { GraphNode, GraphEdge, KnowledgeGraphSummary } from '../KgGraph'

// Placeholder until cytoscape dependency is added. Keeps type surface stable.
// Later: dynamically import 'cytoscape' & optionally layout extensions.

interface Props {
  data: { nodes: GraphNode[]; edges: GraphEdge[] } | null
  summary: KnowledgeGraphSummary
  height: number
}

const CytoscapeGraph: React.FC<Props> = ({ data, summary, height }) => {
  return (
    <div data-testid="kg-graph-canvas" style={{ height, border: '1px solid #ddd', borderRadius: 4, position: 'relative', fontSize: 12 }}>
      <div style={{ padding: 8 }}>Graph placeholder (nodes: {data?.nodes.length || 0} / {summary.node_count})</div>
      <div style={{ position: 'absolute', bottom: 4, right: 8, opacity: 0.6 }}>KG-ID: {summary.kg_id}</div>
    </div>
  )
}

export default CytoscapeGraph
