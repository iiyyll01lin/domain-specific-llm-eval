// Text highlighting utilities for the QA Debugger.
// Tier 1: Use explicit entity arrays from the item's extra payload.
// Tier 2: Heuristic — tokenize rag_answer and highlight word overlaps in contexts.

/** Keys checked in the `extra` field for explicit entity arrays (Tier 1). */
const ENTITY_KEYS = ['entities', 'extracted_entities', 'entity_list', 'named_entities'] as const

/** Extract explicit entity strings from the item's extra passthrough payload. */
export function extractEntities(extra: Record<string, unknown> | undefined): string[] {
  if (!extra) return []
  for (const key of ENTITY_KEYS) {
    const v = extra[key]
    if (Array.isArray(v) && v.length > 0 && typeof v[0] === 'string') {
      return (v as string[]).filter(Boolean)
    }
  }
  return []
}

/**
 * Heuristic fallback (Tier 2): tokenise the LLM answer into unique words ≥ 4 chars,
 * sorted longest-first for greedy matching, capped at 12 terms.
 */
export function extractAnswerTokens(answer: string | undefined): string[] {
  if (!answer) return []
  const words = answer
    .split(/[\s,.;:!?、。，！？·\-–—/\\()[\]{}"「」『』【】《》〈〉]+/)
    .filter((w) => w.length >= 4)
  return [...new Set(words)].sort((a, b) => b.length - a.length).slice(0, 12)
}

/**
 * Build highlighted HTML for a single context string.
 *
 * Security note: `text` and `terms` come from the user's own evaluation data files
 * loaded locally. The injected markup is always a hardcoded `<mark class="...">$1</mark>`
 * pattern with a fixed CSS class — no user-controlled HTML is injected.
 *
 * Returns the highlighted html string, the tier used, and the total match count.
 */
export function buildHighlightedContext(
  text: string,
  entities: string[],
  answerTokens: string[],
): { html: string; tier: 'entity' | 'overlap' | 'none'; matchCount: number } {
  const terms = entities.length > 0 ? entities : answerTokens
  const cls = entities.length > 0 ? 'hl-entity' : 'hl-overlap'
  const tier: 'entity' | 'overlap' | 'none' = entities.length > 0 ? 'entity' : answerTokens.length > 0 ? 'overlap' : 'none'

  if (!terms.length) return { html: text, tier: 'none', matchCount: 0 }

  let matchCount = 0
  let out = text

  for (const term of terms) {
    if (!term) continue
    const escaped = term.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')
    try {
      const re = new RegExp(`(${escaped})`, 'gi')
      const matches = out.match(re)
      if (matches) matchCount += matches.length
      out = out.replace(re, `<mark class="${cls}">$1</mark>`)
    } catch {
      // skip malformed regex (e.g. from unusual entity strings)
    }
  }

  return { html: out, tier, matchCount }
}
