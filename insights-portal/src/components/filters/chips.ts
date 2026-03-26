import { usePortalStore } from '@/app/store/usePortalStore'

// chips.ts - helpers to build active filter chips from filters state
// All comments in English only.

export type MetricRange = { min?: number; max?: number };
export interface FiltersState {
  language?: string | null;
  latencyMs?: MetricRange; // expects milliseconds
  metrics?: Record<string, MetricRange>; // metric keys 0..1
}

export interface ChipItem {
  key: string; // unique key to clear this chip
  label: string; // display text
  type: 'language' | 'latency' | 'metric';
  metricKey?: string; // when type==='metric'
}

// Build user-facing chips from a FiltersState
export function buildActiveChips(filters: FiltersState, locale = 'en-US'): ChipItem[] {
  const chips: ChipItem[] = [];
  if (typeof filters.language === 'string' && filters.language.trim().length > 0) {
    chips.push({ key: 'language', type: 'language', label: `language: ${filters.language}` });
  }
  if (filters.latencyMs && (filters.latencyMs.min != null || filters.latencyMs.max != null)) {
    const { min, max } = filters.latencyMs;
    const minStr = min != null ? min.toLocaleString(locale) : 'min';
    const maxStr = max != null ? max.toLocaleString(locale) : 'max';
    chips.push({ key: 'latencyMs', type: 'latency', label: `latency: ${minStr}–${maxStr} ms` });
  }
  if (filters.metrics) {
    Object.entries(filters.metrics).forEach(([metricKey, range]) => {
      if (range && (range.min != null || range.max != null)) {
        const min = range.min != null ? range.min.toLocaleString(locale) : 'min';
        const max = range.max != null ? range.max.toLocaleString(locale) : 'max';
        chips.push({ key: `metric:${metricKey}`, type: 'metric', metricKey, label: `${metricKey}: ${min}–${max}` });
      }
    });
  }
  return chips;
}

export function buildFilterChips(filters: ReturnType<typeof usePortalStore.getState>['filters']): Array<{ key: string; label: string; onClear: () => void }> {
  const chips: Array<{ key: string; label: string; onClear: () => void }> = []
  const store = usePortalStore.getState()
  if (filters.language) {
    chips.push({ key: 'language', label: `lang:${filters.language}`, onClear: () => store.setFilters({ language: null }) })
  }
  if (filters.latencyRange) {
    const [lo, hi] = filters.latencyRange
    if (lo != null || hi != null) {
      const loS = lo != null ? `${lo}` : '-'
      const hiS = hi != null ? `${hi}` : '-'
      chips.push({ key: 'latency', label: `latency:${loS}..${hiS}ms`, onClear: () => store.setFilters({ latencyRange: [null, null] }) })
    }
  }
  if (filters.metricRanges) {
    for (const [k, [lo, hi]] of Object.entries(filters.metricRanges)) {
      if (lo != null || hi != null) {
        chips.push({ key: `metric:${k}`, label: `${k}:${lo ?? '-'}..${hi ?? '-'}`, onClear: () => {
          // Use dedicated action to clear a specific metric range key
          usePortalStore.getState().clearMetricRange(k)
        } })
      }
    }
  }
  return chips
}
