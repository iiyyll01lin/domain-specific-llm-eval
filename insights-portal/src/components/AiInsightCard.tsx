/**
 * AiInsightCard — Auto-Insights Executive Summary card.
 *
 * Sends the current run's KPI payload to POST /api/v1/insights/generate on the
 * reporting service, then renders the LLM's Markdown response with a premium
 * animated gradient border effect.
 *
 * No external markdown library is required; a minimal inline renderer handles
 * the structured output format emitted by the backend System Prompt.
 */
import React, { useCallback, useRef, useState } from 'react'
import type { RunParsed, Thresholds } from '@/core/types'
import {
  AI_INSIGHTS_NO_KEY_CODE,
  AiInsightsError,
  generateAiInsights,
} from '@/app/lifecycle/api'
import type { AiInsightsResult } from '@/app/lifecycle/api'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Props {
  run: RunParsed
  thresholds: Thresholds
  verdict?: string
  failingMetrics?: string[]
}

type Status = 'idle' | 'generating' | 'done' | 'no-api-key' | 'error'

// ---------------------------------------------------------------------------
// Minimal Markdown → React renderer (no external deps)
// Handles: ##/### headings, **bold**, `code`, _em_, - lists, 1. ol, paragraphs
// ---------------------------------------------------------------------------

function renderInline(text: string, keyPrefix: string | number): React.ReactNode {
  const nodes: React.ReactNode[] = []
  const re = /(\*\*([^*\n]+)\*\*|`([^`\n]+)`|_([^_\n]+)_)/g
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index))
    if (m[2] !== undefined)
      nodes.push(<strong key={`${keyPrefix}-b${m.index}`}>{m[2]}</strong>)
    else if (m[3] !== undefined)
      nodes.push(
        <code
          key={`${keyPrefix}-c${m.index}`}
          style={{
            fontFamily: 'var(--font-mono)',
            background: 'var(--surface-3)',
            padding: '1px 5px',
            borderRadius: 3,
            fontSize: '0.88em',
          }}
        >
          {m[3]}
        </code>,
      )
    else if (m[4] !== undefined)
      nodes.push(<em key={`${keyPrefix}-e${m.index}`}>{m[4]}</em>)
    last = m.index + m[0].length
  }
  if (last < text.length) nodes.push(text.slice(last))
  return nodes.length === 1 ? nodes[0] : <>{nodes}</>
}

function renderMarkdown(markdown: string): React.ReactNode {
  const lines = markdown.split('\n')
  const output: React.ReactNode[] = []
  let listItems: React.ReactNode[] = []
  let listType: 'ul' | 'ol' | null = null
  let listCounter = 0

  const flushList = () => {
    if (!listItems.length) return
    const Tag = listType === 'ol' ? 'ol' : 'ul'
    output.push(
      <Tag key={`lst-${listCounter++}`} style={{ margin: '4px 0 10px 20px', lineHeight: 1.65 }}>
        {listItems}
      </Tag>,
    )
    listItems = []
    listType = null
  }

  lines.forEach((raw, i) => {
    const line = raw.trim()

    if (!line) {
      flushList()
      return
    }

    if (line.startsWith('### ')) {
      flushList()
      output.push(
        <h4 key={i} className="ai-md-h4">
          {renderInline(line.slice(4), i)}
        </h4>,
      )
      return
    }

    if (line.startsWith('## ')) {
      flushList()
      output.push(
        <h3 key={i} className="ai-md-h3">
          {renderInline(line.slice(3), i)}
        </h3>,
      )
      return
    }

    // Bare bold heading e.g. "**Key Findings**" or "**Recommended Actions**"
    if (/^\*\*[^*\n]+\*\*:?\s*$/.test(line)) {
      flushList()
      output.push(
        <p key={i} className="ai-md-section-label">
          {renderInline(line, i)}
        </p>,
      )
      return
    }

    if (line.startsWith('- ') || line.startsWith('* ')) {
      if (listType !== 'ul') { flushList(); listType = 'ul' }
      listItems.push(
        <li key={i} style={{ marginBottom: 3 }}>
          {renderInline(line.slice(2), i)}
        </li>,
      )
      return
    }

    const olMatch = line.match(/^(\d+)\.\s+(.+)$/)
    if (olMatch) {
      if (listType !== 'ol') { flushList(); listType = 'ol' }
      listItems.push(
        <li key={i} style={{ marginBottom: 3 }}>
          {renderInline(olMatch[2], i)}
        </li>,
      )
      return
    }

    flushList()
    output.push(
      <p key={i} className="ai-md-p">
        {renderInline(line, i)}
      </p>,
    )
  })

  flushList()
  return <div className="ai-insight-card__body">{output}</div>
}

// ---------------------------------------------------------------------------
// Skeleton rows while waiting for LLM
// ---------------------------------------------------------------------------

function InsightSkeleton() {
  return (
    <div className="ai-insight-skeleton" aria-busy="true" aria-label="Generating AI summary…">
      <div className="ai-insight-skeleton__row skeleton-pulse" style={{ width: '75%' }} />
      <div className="ai-insight-skeleton__row skeleton-pulse" style={{ width: '90%' }} />
      <div className="ai-insight-skeleton__row skeleton-pulse" style={{ width: '60%' }} />
      <div className="ai-insight-skeleton__row skeleton-pulse" style={{ width: '80%', marginTop: 8 }} />
      <div className="ai-insight-skeleton__row skeleton-pulse" style={{ width: '55%' }} />
      <div className="ai-insight-skeleton__row skeleton-pulse" style={{ width: '70%' }} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// No-API-key friendly message
// ---------------------------------------------------------------------------

function NoKeyMessage() {
  return (
    <div className="ai-insight-no-key" role="status">
      <span className="ai-insight-no-key__icon">🔑</span>
      <div>
        <strong>AI Insights not configured.</strong> To enable this feature, set your LLM API key
        in the <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9em' }}>.env</code> file
        at the repository root:
        <pre
          style={{
            marginTop: 8,
            padding: '6px 10px',
            background: 'var(--surface-3)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 'var(--text-xs)',
            fontFamily: 'var(--font-mono)',
            overflowX: 'auto',
          }}
        >
          {`OPENAI_API_KEY=sk-...your-key...`}
        </pre>
        Then rebuild the <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9em' }}>reporting</code>{' '}
        service container and reload the page. See the{' '}
        <strong>Developer Guide → Auto-Insights</strong> section for details.
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function AiInsightCard({ run, thresholds, verdict, failingMetrics }: Props) {
  const [status, setStatus] = useState<Status>('idle')
  const [result, setResult] = useState<AiInsightsResult | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const handleGenerate = useCallback(async () => {
    // Cancel any in-flight request
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setStatus('generating')
    setResult(null)
    setErrorMsg(null)

    const kpis = (run.kpis ?? {}) as Record<string, number>
    const thresholdMap: Record<string, { warning: number; critical: number }> = {}
    for (const [k, v] of Object.entries(thresholds ?? {})) {
      if (v) thresholdMap[k] = { warning: v.warning, critical: v.critical }
    }

    try {
      const res = await generateAiInsights(
        {
          run_id: run.id,
          kpis,
          counts: run.counts as Record<string, unknown>,
          verdict,
          failing_metrics: failingMetrics,
          thresholds: thresholdMap,
        },
        { signal: controller.signal },
      )
      if (!controller.signal.aborted) {
        setResult(res)
        setStatus('done')
      }
    } catch (err) {
      if (controller.signal.aborted) return
      if (err instanceof AiInsightsError && err.code === AI_INSIGHTS_NO_KEY_CODE) {
        setStatus('no-api-key')
      } else {
        setErrorMsg(err instanceof Error ? err.message : 'Unknown error.')
        setStatus('error')
      }
    }
  }, [run, thresholds, verdict, failingMetrics])

  // Cleanup on unmount
  React.useEffect(() => {
    return () => { abortRef.current?.abort() }
  }, [])

  const isGenerating = status === 'generating'

  const wrapperClass = `ai-insight-wrapper ai-insight-wrapper--${status}`

  return (
    <div className={wrapperClass}>
      <div className="ai-insight-card">
        {/* Header */}
        <div className="ai-insight-card__header">
          <span className="ai-insight-card__sparkle" aria-hidden="true">✨</span>
          <span className="ai-insight-card__title">AI Executive Summary</span>
          <div className="ai-insight-card__actions">
            {result && (
              <span className="ai-insight-card__model-badge">
                {result.model_used}
                {result.prompt_tokens != null && result.completion_tokens != null
                  ? ` · ${result.prompt_tokens + result.completion_tokens} tokens`
                  : ''}
              </span>
            )}
            <button
              className="ai-insight-card__cta"
              onClick={handleGenerate}
              disabled={isGenerating}
              aria-label={isGenerating ? 'Generating executive summary…' : 'Generate Executive Summary'}
            >
              {isGenerating ? (
                <>⏳ Generating…</>
              ) : status === 'done' ? (
                <>🔄 Regenerate</>
              ) : (
                <>✨ Generate Executive Summary</>
              )}
            </button>
          </div>
        </div>

        {/* Body: state-driven */}
        {status === 'idle' && (
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)', margin: 0 }}>
            Click <strong>Generate Executive Summary</strong> to get an AI-powered System Health
            Report for this run.
          </p>
        )}

        {status === 'generating' && <InsightSkeleton />}

        {status === 'done' && result && renderMarkdown(result.summary)}

        {status === 'no-api-key' && <NoKeyMessage />}

        {status === 'error' && (
          <p className="ai-insight-error" role="alert">
            ⚠️ {errorMsg ?? 'Failed to generate summary. Please try again.'}
          </p>
        )}
      </div>
    </div>
  )
}
