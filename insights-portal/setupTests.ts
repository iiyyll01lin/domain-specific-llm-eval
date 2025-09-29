import { afterEach, beforeAll } from 'vitest'
import { cleanup } from '@testing-library/react'
import i18n from './src/app/i18n'

beforeAll(async () => {
  await i18n.changeLanguage('en-US')
})

afterEach(() => {
  cleanup()
})
