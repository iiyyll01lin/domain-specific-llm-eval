import type { MetricKey } from '@/core/types'

// Metric display metadata and formatting helpers.
// Keep it UI-agnostic; UI can use i18n to translate label keys.

export type MetricFormat = (v: number | null | undefined, locale?: string) => string

export interface MetricMeta {
  key: MetricKey
  labelKey: string // i18n key e.g. metrics.ContextPrecision.label
  format?: MetricFormat
  helpKey?: string // i18n key for tooltip/help
  direction?: 'higher'|'lower' // Indicates improvement direction for coloring/deltas
}

const nf = (locale?: string, digits = 3) => new Intl.NumberFormat(locale || undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })

export const defaultFormat: MetricFormat = (v, locale) => {
  if (v == null || Number.isNaN(v as number)) return 'N/A'
  return nf(locale).format(v as number)
}

const known: Record<string, MetricMeta> = {
  ContextPrecision: { key: 'ContextPrecision', labelKey: 'metrics.ContextPrecision.label', helpKey: 'metrics.ContextPrecision.help', format: defaultFormat, direction: 'higher' },
  ContextRecall: { key: 'ContextRecall', labelKey: 'metrics.ContextRecall.label', helpKey: 'metrics.ContextRecall.help', format: defaultFormat, direction: 'higher' },
  Faithfulness: { key: 'Faithfulness', labelKey: 'metrics.Faithfulness.label', helpKey: 'metrics.Faithfulness.help', format: defaultFormat, direction: 'higher' },
  AnswerRelevancy: { key: 'AnswerRelevancy', labelKey: 'metrics.AnswerRelevancy.label', helpKey: 'metrics.AnswerRelevancy.help', format: defaultFormat, direction: 'higher' },
  AnswerSimilarity: { key: 'AnswerSimilarity', labelKey: 'metrics.AnswerSimilarity.label', helpKey: 'metrics.AnswerSimilarity.help', format: defaultFormat, direction: 'higher' },
  ContextualKeywordMean: { key: 'ContextualKeywordMean', labelKey: 'metrics.ContextualKeywordMean.label', helpKey: 'metrics.ContextualKeywordMean.help', format: defaultFormat, direction: 'higher' },
  // Graph Context Relevance metrics (Se / Sc / Ph)
  gcr_score: { key: 'gcr_score', labelKey: 'metrics.gcr_score.label', helpKey: 'metrics.gcr_score.help', format: defaultFormat, direction: 'higher' },
  entity_overlap: { key: 'entity_overlap', labelKey: 'metrics.entity_overlap.label', helpKey: 'metrics.entity_overlap.help', format: defaultFormat, direction: 'higher' },
  structural_connectivity: { key: 'structural_connectivity', labelKey: 'metrics.structural_connectivity.label', helpKey: 'metrics.structural_connectivity.help', format: defaultFormat, direction: 'higher' },
  hub_noise_penalty: { key: 'hub_noise_penalty', labelKey: 'metrics.hub_noise_penalty.label', helpKey: 'metrics.hub_noise_penalty.help', format: defaultFormat, direction: 'lower' },
}

export function getMetricMeta(key: MetricKey): MetricMeta {
  if (known[key]) return known[key]
  // Generic fallback: construct reasonable defaults for unseen metric keys.
  // Allow i18n layer to decide if translation exists; UI should fallback to key when missing.
  return {
    key,
    labelKey: `metrics.${key}.label`,
    helpKey: `metrics.${key}.help`,
    format: defaultFormat,
    direction: 'higher',
  }
}

export function metricDirection(key: MetricKey): 'higher'|'lower' {
  return getMetricMeta(key).direction || 'higher'
}
