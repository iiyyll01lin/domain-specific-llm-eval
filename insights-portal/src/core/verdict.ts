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

// Threshold validation helpers
export function validateThresholdValue(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1
}

export function validateThresholdsShape(thresholds: Thresholds): { ok: boolean; errors: string[] } {
  const errors: string[] = []
  for (const [k, th] of Object.entries(thresholds)) {
    if (!th) continue
    const w = (th as any).warning
    const c = (th as any).critical
    if (!validateThresholdValue(w)) errors.push(`${k}.warning invalid`)
    if (!validateThresholdValue(c)) errors.push(`${k}.critical invalid`)
    if (validateThresholdValue(w) && validateThresholdValue(c) && c > w) errors.push(`${k}: critical must be <= warning`)
  }
  return { ok: errors.length === 0, errors }
}
