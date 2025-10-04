<!-- src/App.vue -->
<template>
  <div>
    <div v-if="!authenticated" class="login-container">
      <h2>访问受限</h2>
      <input
        v-model="inputPassword"
        type="password"
        placeholder="请输入访问密码"
        @keyup.enter="verifyPassword"
      />
      <button @click="verifyPassword">进入系统</button>
      <p v-if="error" class="error">{{ error }}</p>
    </div>
    <div v-else>
      <nav class="navbar">
        <router-link to="/">首页</router-link>
        <router-link to="/asset_conf">目标资产管理</router-link>
        <router-link to="/factor_trend">因子趋势</router-link>
      </nav>
      <router-view />
    </div>
  </div>
</template>

<style scoped>
.navbar {
  display: flex;
  align-items: center;
  gap: 2rem;             /* 控制间距 */
  padding: 1rem;
  border-bottom: 1px solid #e5e7eb;
  background: #fff;
  box-shadow: 0 1px 2px rgb(0 0 0 / 0.04);
}
.navbar a {
  color: #374151;        /* text-gray-700 */
  text-decoration: none;
  transition: color .2s ease;
}
.navbar a:hover {
  color: #2563eb;        /* hover:text-blue-600 */
  text-decoration: underline;
}
/* 当前路由高亮（可选） */
.navbar :deep(.router-link-exact-active) {
  color: #2563eb;
  font-weight: 600;
}
.login-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  gap: 1rem;
}
input {
  padding: 0.5rem;
  font-size: 1rem;
  width: 200px;
}
button {
  padding: 0.5rem 1rem;
  cursor: pointer;
}
.error {
  color: red;
}
</style>

<script setup lang="ts">
import { ref } from 'vue'
import bcrypt from 'bcryptjs'

const inputPassword = ref('')
const authenticated = ref(false)
const error = ref('')

// 从 Vite 环境变量读取 hash
const ACCESS_HASH = import.meta.env.VITE_APP_ACCESS_HASH

function verifyPasswordSync(pw: string) {
  if (!ACCESS_HASH) {
    console.error('VITE_APP_ACCESS_HASH 未设置')
    return false
  }
  try {
    return bcrypt.compareSync(pw, ACCESS_HASH)
  } catch (e) {
    console.error('bcrypt compare 出错：', e)
    return false
  }
}

const verifyPassword = async () => {
  const ok = verifyPasswordSync(inputPassword.value)
  if (ok) {
    authenticated.value = true
    localStorage.setItem('auth_ok', '1')
  } else {
    error.value = '密码错误，请重试'
  }
}

// 页面加载时自动检查是否已通过验证
if (localStorage.getItem('auth_ok') === '1') {
  authenticated.value = true
}
</script>
