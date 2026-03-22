import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

const apiProxy = {
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ''),
  },
} as const

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    extensions: ['.tsx', '.ts', '.jsx', '.js', '.json'],
  },
  server: {
    // Allow Cursor / tunnel hostnames that are not "localhost"
    host: true,
    // Dev + HTTPS preview: same-origin /api → FastAPI (no mixed content)
    proxy: { ...apiProxy },
  },
  preview: {
    host: true,
    // `npm run preview` sets DEV=false; proxy must exist here too
    proxy: { ...apiProxy },
  },
  test: {
    environment: 'node',
  },
})

