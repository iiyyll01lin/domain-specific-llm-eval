import React from 'react'

interface KpiSkeletonProps {
  /** Number of placeholder cards to render.  */
  count?: number
}

/**
 * Animated skeleton loader that mimics the KPI card grid while a run is being parsed.
 * Uses the `.skeleton-pulse` CSS animation from theme.css.
 */
export const KpiSkeleton: React.FC<KpiSkeletonProps> = ({ count = 8 }) => {
  return (
    <div
      data-testid="kpi-skeleton"
      aria-busy="true"
      aria-label="Loading metrics…"
      style={{
        marginTop: 16,
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: 12,
      }}
    >
      {/* Full-width verdict placeholder */}
      <div
        className="skeleton-pulse"
        style={{
          gridColumn: '1 / -1',
          height: 74,
          borderRadius: 'var(--radius-lg)',
        }}
      />
      {/* Full-width insights placeholder */}
      <div
        className="skeleton-pulse"
        style={{
          gridColumn: '1 / -1',
          height: 56,
          borderRadius: 'var(--radius-lg)',
        }}
      />
      {/* Individual KPI card placeholders */}
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="kpi-card"
          style={{ minHeight: 92, display: 'flex', flexDirection: 'column', gap: 8 }}
        >
          {/* Label line */}
          <div
            className="skeleton-pulse"
            style={{ height: 12, width: '55%', borderRadius: 'var(--radius-sm)' }}
          />
          {/* Value line */}
          <div
            className="skeleton-pulse"
            style={{ height: 28, width: '75%', borderRadius: 'var(--radius-sm)', marginTop: 4 }}
          />
          {/* Status line */}
          <div
            className="skeleton-pulse"
            style={{ height: 10, width: '40%', borderRadius: 'var(--radius-sm)' }}
          />
        </div>
      ))}
    </div>
  )
}

export default KpiSkeleton
