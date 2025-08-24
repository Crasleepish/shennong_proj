import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import App from '../App.vue'
import AssetConf from '../views/AssetConf.vue'
import Home from '../views/Home.vue'

const routes: Array<RouteRecordRaw> = [
  { path: '/', component: Home },
  { path: '/asset_conf', component: AssetConf },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
