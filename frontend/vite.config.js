import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000, // Keep frontend on port 3000 (backend is on 5173)
    proxy: {
      '/ws': {
        target: 'ws://localhost:5173',
        ws: true,
        changeOrigin: true,
      },
      '/images': {
        target: 'http://localhost:5173',
        changeOrigin: true,
      }
    }
  }
}) 