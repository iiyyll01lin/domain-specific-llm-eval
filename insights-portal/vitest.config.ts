import { defineConfig } from 'vitest/config'
import path from 'node:path'

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./setupTests.ts'],
    exclude: [
  // Vitest defaults + project-specific
  '**/node_modules/**',
  '**/dist/**',
  '**/build/**',
  '**/.{idea,git,cache,output,temp}/**',
      'tests/e2e/**',
      'playwright-report/**',
      'test-results/**',
    ],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
})
