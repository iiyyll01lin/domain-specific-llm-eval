import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    fs: {
      // Allow serving files from the workspace root so we can fetch eval artifacts via /@fs/absolute/path in dev
      allow: [fileURLToPath(new URL('..', import.meta.url))],
    },
  },
})
