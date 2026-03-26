import React from 'react'
import { useTranslation } from 'react-i18next'

interface EmptyStateProps {
  /** Called when the user clicks "Load JSON" */
  onLoadFile?: () => void
  /** Called when the user clicks "Load CSV" */
  onLoadCsv?: () => void
}

/**
 * Full-page empty state shown when no run has been loaded yet.
 * Displays an abstract knowledge-graph SVG motif + bilingual CTA buttons.
 */
export const EmptyState: React.FC<EmptyStateProps> = ({ onLoadFile, onLoadCsv }) => {
  const { t } = useTranslation()

  return (
    <div
      data-testid="empty-state"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 'clamp(32px, 8vh, 80px) 24px',
        textAlign: 'center',
        gap: 28,
        minHeight: 360,
      }}
    >
      {/* Abstract knowledge-graph illustration */}
      <svg
        width="128"
        height="128"
        viewBox="0 0 128 128"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
        style={{ flexShrink: 0 }}
      >
        {/* Outer ring */}
        <circle cx="64" cy="64" r="62" stroke="var(--border)" strokeWidth="1.5" fill="var(--surface-2)" />

        {/* Edge lines (dashed) */}
        <line x1="64" y1="40" x2="34" y2="76" stroke="var(--border)" strokeWidth="1.5" strokeDasharray="5 4" strokeLinecap="round" />
        <line x1="64" y1="40" x2="94" y2="76" stroke="var(--border)" strokeWidth="1.5" strokeDasharray="5 4" strokeLinecap="round" />
        <line x1="38"  y1="84" x2="56" y2="90" stroke="var(--border)" strokeWidth="1.5" strokeDasharray="5 4" strokeLinecap="round" />
        <line x1="90"  y1="84" x2="72" y2="90" stroke="var(--border)" strokeWidth="1.5" strokeDasharray="5 4" strokeLinecap="round" />
        <line x1="34" y1="76" x2="90" y2="76" stroke="var(--border)" strokeWidth="1.5" strokeDasharray="5 4" strokeLinecap="round" />

        {/* Top node (accent blue) */}
        <circle cx="64" cy="36" r="12" fill="var(--accent-subtle)" stroke="var(--accent)" strokeWidth="1.5" />
        <circle cx="64" cy="36" r="4"  fill="var(--accent)" />

        {/* Bottom-left node (GCR teal) */}
        <circle cx="32" cy="76" r="11" fill="var(--gcr-accent-subtle)" stroke="var(--gcr-accent)" strokeWidth="1.5" />
        <circle cx="32" cy="76" r="3.5" fill="var(--gcr-accent)" />

        {/* Bottom-right node (GCR teal) */}
        <circle cx="96" cy="76" r="11" fill="var(--gcr-accent-subtle)" stroke="var(--gcr-accent)" strokeWidth="1.5" />
        <circle cx="96" cy="76" r="3.5" fill="var(--gcr-accent)" />

        {/* Centre-bottom small node */}
        <circle cx="64" cy="92" r="8"  fill="var(--surface-3)" stroke="var(--border)" strokeWidth="1.5" />
        <circle cx="64" cy="92" r="2.5" fill="var(--text-subtle)" />
      </svg>

      {/* Text block */}
      <div style={{ maxWidth: 380 }}>
        <h2
          style={{
            margin: '0 0 10px',
            fontSize: 'var(--text-xl)',
            fontWeight: 700,
            color: 'var(--text)',
            letterSpacing: '-0.02em',
          }}
        >
          {t('overview.emptyTitle')}
        </h2>
        <p
          style={{
            margin: 0,
            fontSize: 'var(--text-base)',
            color: 'var(--text-muted)',
            lineHeight: 'var(--line-height)',
          }}
        >
          {t('overview.emptySubtitle')}
        </p>
      </div>

      {/* CTA buttons */}
      {(onLoadFile || onLoadCsv) && (
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'center' }}>
          {onLoadFile && (
            <button
              onClick={onLoadFile}
              style={{
                background: 'var(--accent)',
                color: '#ffffff',
                border: 'none',
                borderRadius: 'var(--radius-md)',
                padding: '9px 22px',
                fontWeight: 600,
                fontSize: 'var(--text-base)',
                cursor: 'pointer',
                boxShadow: '0 1px 4px rgba(0,0,0,0.25)',
              }}
            >
              {t('overview.emptyCtaJson')}
            </button>
          )}
          {onLoadCsv && (
            <button
              onClick={onLoadCsv}
              style={{
                background: 'var(--surface-3)',
                color: 'var(--text)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                padding: '9px 22px',
                fontWeight: 500,
                fontSize: 'var(--text-base)',
                cursor: 'pointer',
              }}
            >
              {t('overview.emptyCtaCsv')}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export default EmptyState
