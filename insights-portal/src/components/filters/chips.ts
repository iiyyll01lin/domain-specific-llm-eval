import { usePortalStore } from '@/app/store/usePortalStore'

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
          const mr = { ...(store.filters.metricRanges || {}) }
          delete mr[k]
          store.setFilters({ metricRanges: mr })
        } })
      }
    }
  }
  return chips
}
