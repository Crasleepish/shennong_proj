import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: 'http://8.140.27.136', // 后端地址
        changeOrigin: true,
        rewrite: path => path.replace(/^\/api/, '') // 可选：重写路径
      }
    }
  }
})
