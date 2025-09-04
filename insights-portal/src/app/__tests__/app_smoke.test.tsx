import { describe, it, expect } from 'vitest'
import React from 'react'
import { render } from '@testing-library/react'
import { TID } from '@/testing/testids'
import { App } from '../App'

describe('App smoke test', () => {
  it('renders navigation buttons by test id', () => {
    const { getByTestId } = render(<App />)
    expect(() => getByTestId(TID.nav.executive)).not.toThrow()
    expect(() => getByTestId(TID.nav.qa)).not.toThrow()
    expect(() => getByTestId(TID.nav.analytics)).not.toThrow()
  })
})
