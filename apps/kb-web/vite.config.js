import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// /kb 开头的请求交给后端，浏览器只跟 5173 说话，避免跨域配置
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/kb': 'http://127.0.0.1:8001',
    },
  },
});
