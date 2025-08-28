import type { MetricKey } from '@/core/types'

// Metric display metadata and formatting helpers.
// Keep it UI-agnostic; UI can use i18n to translate label keys.

export type MetricFormat = (v: number | null | undefined, locale?: string) => string

export interface MetricMeta {
  key: MetricKey
  labelKey: string // i18n key e.g. metrics.ContextPrecision.label
  format?: MetricFormat
  helpKey?: string // i18n key for tooltip/help
}

const nf = (locale?: string, digits = 3) => new Intl.NumberFormat(locale || undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })

export const defaultFormat: MetricFormat = (v, locale) => {
  if (v == null || Number.isNaN(v as number)) return 'N/A'
  return nf(locale).format(v as number)
}

const known: Record<string, MetricMeta> = {
  ContextPrecision: { key: 'ContextPrecision', labelKey: 'metrics.ContextPrecision.label', helpKey: 'metrics.ContextPrecision.help', format: defaultFormat },
  ContextRecall: { key: 'ContextRecall', labelKey: 'metrics.ContextRecall.label', helpKey: 'metrics.ContextRecall.help', format: defaultFormat },
  Faithfulness: { key: 'Faithfulness', labelKey: 'metrics.Faithfulness.label', helpKey: 'metrics.Faithfulness.help', format: defaultFormat },
  AnswerRelevancy: { key: 'AnswerRelevancy', labelKey: 'metrics.AnswerRelevancy.label', helpKey: 'metrics.AnswerRelevancy.help', format: defaultFormat },
  AnswerSimilarity: { key: 'AnswerSimilarity', labelKey: 'metrics.AnswerSimilarity.label', helpKey: 'metrics.AnswerSimilarity.help', format: defaultFormat },
  ContextualKeywordMean: { key: 'ContextualKeywordMean', labelKey: 'metrics.ContextualKeywordMean.label', helpKey: 'metrics.ContextualKeywordMean.help', format: defaultFormat },
}

export function getMetricMeta(key: MetricKey): MetricMeta {
  return known[key] || { key, labelKey: `metrics.${key}.label`, helpKey: `metrics.${key}.help`, format: defaultFormat }
}
