import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

/** Match `PREDICT_TIMEOUT_MS` in src/api/chexit.ts — default http-proxy proxyTimeout is 2 min and drops long /predict. */
const API_PROXY_TIMEOUT_MS = 10 * 60 * 1000

const apiProxy = {
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ''),
    timeout: API_PROXY_TIMEOUT_MS,
    proxyTimeout: API_PROXY_TIMEOUT_MS,
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
    // Allow rotating ngrok free subdomains during local sharing
    allowedHosts: ['.ngrok-free.dev', '.ngrok-free.app'],
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

