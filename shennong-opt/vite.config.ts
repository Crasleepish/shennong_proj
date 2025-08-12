import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000', // 后端地址
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, '') // 可选：重写路径
      }
    }
  }
})
