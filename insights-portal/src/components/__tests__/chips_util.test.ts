import { describe, it, expect } from 'vitest'
import { buildFilterChips } from '@/components/filters/chips'
import { usePortalStore } from '@/app/store/usePortalStore'

describe('buildFilterChips', () => {
  it('produces chips for language, latency, and metric ranges with proper labels', () => {
    // Reset and set filters
    usePortalStore.getState().clearFilters()
    usePortalStore.getState().setFilters({
      language: 'en',
      latencyRange: [100, 500],
      metricRanges: { Faithfulness: [0.5, 0.9] },
    })

    const filters = usePortalStore.getState().filters
    const chips = buildFilterChips(filters)
    const labels = chips.map((c) => c.label)

    expect(labels).toEqual(expect.arrayContaining([
      'lang:en',
      'latency:100..500ms',
      'Faithfulness:0.5..0.9',
    ]))
  })

  it('onClear clears the corresponding filter entry', () => {
    // Ensure state
    usePortalStore.getState().clearFilters()
    usePortalStore.getState().setFilters({
      language: 'en',
      latencyRange: [100, 500],
      metricRanges: { Faithfulness: [0.5, 0.9] },
    })

    const filters = usePortalStore.getState().filters
    const chips = buildFilterChips(filters)

    // Clear language
    const langChip = chips.find((c) => c.key === 'language')
    expect(langChip).toBeTruthy()
    langChip!.onClear()
    expect(usePortalStore.getState().filters.language).toBeNull()

    // Clear metric range
    const metricChip = buildFilterChips(usePortalStore.getState().filters).find((c) => c.key === 'metric:Faithfulness')
    expect(metricChip).toBeTruthy()
    metricChip!.onClear()
    expect(usePortalStore.getState().filters.metricRanges?.Faithfulness).toBeUndefined()

    // Clear latency
    const latencyChip = buildFilterChips(usePortalStore.getState().filters).find((c) => c.key === 'latency')
    expect(latencyChip).toBeTruthy()
    latencyChip!.onClear()
    expect(usePortalStore.getState().filters.latencyRange).toEqual([null, null])
  })
})
