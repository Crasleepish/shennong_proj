<!-- src/App.vue -->
<template>
  <div>
    <nav class="navbar">
      <router-link to="/">首页</router-link>
      <router-link to="/asset_conf">目标资产管理</router-link>
      <router-link to="/factor_trend">因子趋势</router-link>
    </nav>
    <router-view />
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
</style>

<script setup lang="ts">
</script>
