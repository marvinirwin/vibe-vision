import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy /api requests to our backend server
      '/api': {
        target: 'http://localhost:3001', // The address of your Python backend
        changeOrigin: true, // Recommended for virtual hosted sites
        // No rewrite needed as frontend already uses /api prefix
        // rewrite: (path) => path.replace(/^\/api/, '') 
      }
    }
  }
}) 