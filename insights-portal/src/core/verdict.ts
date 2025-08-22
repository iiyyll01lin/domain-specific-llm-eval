import type { Thresholds, VerdictResult, MetricKey } from './types'

export function evaluateVerdict(
  kpis: Partial<Record<MetricKey, number | undefined>>,
  thresholds: Thresholds
): VerdictResult {
  const failingCritical: MetricKey[] = []
  const failingWarning: MetricKey[] = []
  for (const [mk, lv] of Object.entries(thresholds)) {
    const key = mk as MetricKey
    const v = kpis[key] ?? null
    if (v == null || lv == null) continue
    if (v < lv.critical) failingCritical.push(key)
    else if (v < lv.warning) failingWarning.push(key)
  }
  if (failingCritical.length) return { verdict: 'Blocked', triggeredRuleId: 'any_critical', failingMetrics: failingCritical }
  if (failingWarning.length) return { verdict: 'At Risk', triggeredRuleId: 'any_warning', failingMetrics: failingWarning }
  return { verdict: 'Ready', triggeredRuleId: 'all_warning_or_above', failingMetrics: [] }
}
