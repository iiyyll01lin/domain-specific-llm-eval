import React from 'react'
import * as echarts from 'echarts'
import type { GraphNode, GraphEdge, KnowledgeGraphSummary } from '../KgGraph'

interface Props {
  data: { nodes: GraphNode[]; edges: GraphEdge[] } | null
  summary: KnowledgeGraphSummary
  height: number
}

// Distinct colors per entity type — based on Okabe-Ito palette (color-blind safe)
const ENTITY_COLORS: Record<string, string> = {
  default:  '#58a6ff',
  document: '#2dd4bf',
  concept:  '#a78bfa',
  entity:   '#f59e0b',
  relation: '#34d399',
  hub:      '#f87171',
}

function getColor(et?: string): string {
  return ENTITY_COLORS[et || 'default'] ?? ENTITY_COLORS.default
}

const CytoscapeGraph: React.FC<Props> = ({ data, summary, height }) => {
  const containerRef = React.useRef<HTMLDivElement | null>(null)
  const chartRef     = React.useRef<echarts.ECharts | null>(null)

  // ── Mount / unmount lifecycle ─────────────────────────────────────────
  React.useEffect(() => {
    if (!containerRef.current) return
    const chart = echarts.init(containerRef.current, undefined, { renderer: 'canvas' })
    chartRef.current = chart
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(containerRef.current)
    return () => {
      ro.disconnect()
      chart.dispose()
      chartRef.current = null
    }
  }, [])

  // ── Data update ───────────────────────────────────────────────────────
  React.useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    if (!data || !data.nodes.length) {
      chart.setOption({
        graphic: [{
          type: 'text', left: 'center', top: 'middle',
          style: { text: 'No graph data', fill: 'var(--text-muted, #8b949e)', font: '14px Inter, system-ui' },
        }],
        series: [],
      }, { notMerge: true })
      return
    }

    // Map degree to symbol size [12, 44]
    const maxDeg = Math.max(1, ...data.nodes.map((n) => n.degree ?? 0))
    const sizeOf = (deg: number) => 12 + Math.round(((deg / maxDeg) ** 0.5) * 32)

    const gfxNodes = data.nodes.map((n) => ({
      id:         n.id,
      name:       n.label,
      symbolSize: sizeOf(n.degree ?? 0),
      itemStyle:  { color: getColor(n.entityType), borderWidth: 1.5, borderColor: 'rgba(255,255,255,0.18)' },
      label: {
        show:     (n.degree ?? 0) >= Math.max(1, Math.ceil(maxDeg * 0.2)),
        fontSize: 10,
        distance: 4,
      },
      // Tooltip data
      _degree:   n.degree,
      _type:     n.entityType,
    }))

    const gfxEdges = data.edges.map((e) => ({
      source:    e.source,
      target:    e.target,
      lineStyle: {
        width:   Math.max(0.5, Math.min(4, (e.weight ?? 0.5) * 3)),
        opacity: 0.45,
        color:   e.kind === 'similar' ? '#2dd4bf' : '#58a6ff',
        curveness: 0.05,
      },
    }))

    chart.setOption({
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        formatter: (p: any) => {
          if (p.dataType === 'node') {
            const type   = p.data._type   ? `<div style="opacity:.7">Type: ${p.data._type}</div>`   : ''
            const degree = p.data._degree != null ? `<div style="opacity:.7">Degree: ${p.data._degree}</div>` : ''
            return `<div style="font-size:12px;line-height:1.6"><b>${p.data.name}</b>${type}${degree}</div>`
          }
          return `<div style="font-size:12px"><b>${p.data.source}</b> → <b>${p.data.target}</b></div>`
        },
      },
      series: [{
        type:       'graph',
        layout:     'force',
        data:       gfxNodes,
        edges:      gfxEdges,
        roam:       true,
        draggable:  true,
        force: {
          repulsion:  180,
          gravity:    0.06,
          edgeLength: [50, 130],
          friction:   0.65,
        },
        emphasis: {
          focus:     'adjacency',
          lineStyle: { width: 3, opacity: 0.9 },
          label:     { show: true },
        },
        edgeSymbol:     ['none', 'arrow'],
        edgeSymbolSize: [0, 7],
        label:          { position: 'right', fontSize: 10, color: 'var(--text, #e6edf3)' },
        lineStyle:      { color: 'source', curveness: 0.05 },
        // Smooth animation settings
        animation:           true,
        animationDuration:   600,
        animationEasing:     'cubicOut',
        animationDurationUpdate: 300,
      }],
    }, { notMerge: true })
  }, [data])

  return (
    <div data-testid="kg-graph-canvas" style={{ height, position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {/* Minimal legend */}
      <div style={{
        position: 'absolute', bottom: 8, left: 10,
        display: 'flex', gap: 10, flexWrap: 'wrap',
        fontSize: 10, opacity: 0.7, pointerEvents: 'none',
      }}>
        {Object.entries(ENTITY_COLORS).map(([type, color]) => (
          <span key={type} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: color }} />
            {type}
          </span>
        ))}
      </div>

      {/* KG metadata footer */}
      <div style={{ position: 'absolute', bottom: 6, right: 10, opacity: 0.5, fontSize: 10, pointerEvents: 'none' }}>
        KG&nbsp;{summary.kg_id}&nbsp;·&nbsp;{data?.nodes.length ?? 0}/{summary.node_count}&nbsp;nodes
      </div>
    </div>
  )
}

export default CytoscapeGraph
