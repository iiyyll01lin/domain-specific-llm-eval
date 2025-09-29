import '@testing-library/jest-dom/vitest'

if (typeof window !== 'undefined') {
  await import('./src/app/i18n')
}
