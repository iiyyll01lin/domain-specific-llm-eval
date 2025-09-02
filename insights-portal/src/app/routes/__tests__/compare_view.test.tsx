// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from 'vitest'
import React from 'react'
import { render, screen } from '@testing-library/react'
import CompareView from '../CompareView'
import { usePortalStore } from '@/app/store/usePortalStore'

describe('CompareView', () => {
  beforeEach(() => {
    const set = usePortalStore.getState()
    set.setRuns({
      a: { items: [{ id: '1', language: 'en', latencyMs: 10, metrics: { Faithfulness: 0.5 } }], kpis: { Faithfulness: 0.5 }, counts: { total: 1 } } as any,
      b: { items: [{ id: '2', language: 'en', latencyMs: 12, metrics: { Faithfulness: 0.6 } }], kpis: { Faithfulness: 0.6 }, counts: { total: 1 } } as any,
    })
    set.setSelectedRuns(['a', 'b'])
  })

  it('renders baseline selector and exports', () => {
    render(<CompareView />)
  // Basic smoke: elements are queryable
  screen.getByText('Compare Runs')
  screen.getByTestId('compare-baseline')
  screen.getByLabelText('export-compare-csv')
  screen.getByLabelText('export-compare-xlsx')
  })
})
