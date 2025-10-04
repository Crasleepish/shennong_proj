import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import App from '../App.vue'
import AssetConf from '../views/AssetConf.vue'
import Home from '../views/Home.vue'
import FactorTrend from '../views/FactorTrend.vue'

const routes: Array<RouteRecordRaw> = [
  { path: '/', component: Home },
  { path: '/asset_conf', component: AssetConf },
  { path: '/factor_trend', name: 'factor_trend', component: FactorTrend },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
