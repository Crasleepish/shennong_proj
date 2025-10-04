<template>
  <div class="factor-trend">
    <!-- 控制条 -->
    <div class="toolbar">
      <div class="row">
        <label>起始日期</label>
        <input type="date" v-model="start" @change="refreshAll" />
        <label>结束日期</label>
        <input type="date" v-model="end" @change="refreshAll" />

        <label>模式</label>
        <select v-model="mode" @change="refreshAll">
          <option value="daily">历史日线</option>
          <option value="with_intraday" :disabled="!meta.has_intraday">历史日线 + 盘中</option>
        </select>

        <button @click="updateIntraday" :disabled="updating">
          {{ updating ? '更新中...' : '更新盘中数据' }}
        </button>

        <span class="meta" v-if="meta.has_intraday">
          盘中更新时间：{{ meta.intraday_updated_at }}
        </span>
      </div>
      <div class="row">
        <label>Softprob资产</label>
        <select v-model="probAsset" @change="renderProbChart">
          <option v-for="opt in probAssets" :key="opt" :value="opt">{{ opt }}</option>
        </select>
      </div>
    </div>

    <!-- 图表区 -->
    <div class="charts">
      <div class="chart-card">
        <div class="chart-title">累计净值（首日归一）</div>
        <div ref="navRef" class="chart"></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">Softprob（prob0/1/2）</div>
        <div ref="probRef" class="chart"></div>
      </div>
    </div>

    <!-- 错误信息 -->
    <div v-if="error" class="error">{{ error }}</div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, watch, nextTick } from 'vue'
import * as echarts from 'echarts'

const baseUrl = import.meta.env.VITE_API_BASE_URL as string
const authHeader = import.meta.env.VITE_API_AUTH_HEADER as string

type Mode = 'daily' | 'with_intraday'

const navRef = ref<HTMLDivElement | null>(null)
const probRef = ref<HTMLDivElement | null>(null)
let navChart: echarts.ECharts | null = null
let probChart: echarts.ECharts | null = null

// ----- 状态 -----
const today = new Date()
const defaultEnd = today.toISOString().slice(0, 10)
const defaultStart = new Date(today.getTime() - 730 * 24 * 3600 * 1000) // 近两年
  .toISOString()
  .slice(0, 10)

const start = ref<string>(defaultStart)
const end = ref<string>(defaultEnd)
const mode = ref<Mode>('daily')

const meta = reactive({
  has_intraday: false,
  intraday_updated_at: '',
})

const updating = ref(false)
const error = ref<string>('')

// 接口返回数据
const navSeries = reactive<Record<string, Array<{ date: string; nav: number | null }>>>({})
const probSeries = reactive<Record<string, Array<{ date: string; prob0: number | null; prob1: number | null; prob2: number | null }>>>({})

// 概率图的资产选单
const probAssets = ref<string[]>([])
const probAsset = ref<string>('MKT')

// ----- 工具：Fetch -----
async function apiGet<T>(path: string): Promise<T> {
  const url = `${baseUrl}/${path}`
  const res = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/json'
      }
    })
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<T>
}
async function apiPost<T>(path: string, body?: any): Promise<T> {
  const url = `${baseUrl}/${path}`
  const res = await fetch(url, {
    method: 'POST',
    headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/json'
      },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json() as Promise<T>
}

// ----- 业务：拉数据 -----
async function fetchNav() {
  const qs = new URLSearchParams({ start: start.value, end: end.value, mode: mode.value })
  const url = `showdata/nav?${qs.toString()}`
  const json = await apiGet<any>(url)
  if (!json.ok) throw new Error(json.error || 'NAV 接口返回失败')
  // series: { MKT: [{date,nav}, ...], SMB: [...], ... }
  Object.keys(json.series || {}).forEach(k => (navSeries[k] = json.series[k]))
  meta.has_intraday = !!json.meta?.has_intraday
  meta.intraday_updated_at = json.meta?.intraday_updated_at || ''
}

async function fetchPredict() {
  const qs = new URLSearchParams({ start: start.value, end: end.value, mode: mode.value })
  const url = `showdata/predict?${qs.toString()}`
  const json = await apiGet<any>(url)
  if (!json.ok) throw new Error(json.error || 'Predict 接口返回失败')
  // probs: { MKT: [{date, prob0, prob1, prob2}, ...], ... }
  Object.keys(json.probs || {}).forEach(k => (probSeries[k] = json.probs[k]))
  meta.has_intraday = meta.has_intraday || !!json.meta?.has_intraday
  meta.intraday_updated_at = meta.intraday_updated_at || json.meta?.intraday_updated_at || ''

  // 刷新资产列表 & 默认选项
  const keys = Object.keys(probSeries)
  probAssets.value = keys
  if (!keys.includes(probAsset.value) && keys.length > 0) {
    probAsset.value = keys[0]
  }
}

// ----- 盘中刷新 -----
async function updateIntraday() {
  try {
    updating.value = true
    error.value = ''
    const res = await apiPost<any>('showdata/intraday/update')
    if (!res.ok) throw new Error(res.error || '更新盘中失败')
    meta.has_intraday = !!res.meta?.has_intraday
    meta.intraday_updated_at = res.meta?.intraday_updated_at || ''
    // 用户点了更新 => 自动切到 with_intraday
    mode.value = 'with_intraday'
    // 立刻刷新数据
    await refreshAll()
  } catch (e: any) {
    error.value = e?.message || String(e)
    console.error(e)
  } finally {
    updating.value = false
  }
}

// ----- 渲染图表 -----
function renderNavChart() {
  if (!navRef.value) return
  if (!navChart) navChart = echarts.init(navRef.value)

  const colors = [
    '#5470C6', '#91CC75', '#FAC858', '#EE6666', '#73C0DE', '#3BA272',
  ]
  const seriesDefs: echarts.SeriesOption[] = []
  const allDatesSet = new Set<string>()
  const legend: string[] = []

  const order = ['MKT', 'SMB', 'HML', 'QMJ', 'BOND10Y', 'GOLD']
  order.forEach((key, _idx) => {
    const arr = navSeries[key]
    if (!arr || !arr.length) return
    legend.push(key)
    arr.forEach(p => allDatesSet.add(p.date))
  })
  const allDates = Array.from(allDatesSet).sort()

  order.forEach((key, idx) => {
    const arr = navSeries[key]
    if (!arr || !arr.length) return
    const map = new Map(arr.map(p => [p.date, p.nav]))
    const data = allDates.map(d => [d, map.get(d) ?? null])
    seriesDefs.push({
      name: key,
      type: 'line',
      showSymbol: false,
      connectNulls: true,
      smooth: true,
      data,
      lineStyle: { width: 2 },
      emphasis: { focus: 'series' },
      itemStyle: { color: colors[idx % colors.length] },
    })
  })

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'axis' },
    legend: {
      type: 'scroll',            // 避免过长时换行挤占空间
      data: legend,
      top: 6,                    // 固定到上方
      left: 16,
      right: 16,
      padding: [2, 0, 2, 0]
    },
    grid: {
      left: 56, right: 24,
      top: 64,                   // ↑ 加大上边距，给 legend 留空间
      bottom: 64,                // ↑ 加大下边距，避免与 x 轴重叠
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: Array.from(new Set(allDates)).sort(),
      axisLabel: {
        hideOverlap: true,       // 自动避让
        margin: 12,              // 与轴线距离
        formatter: (v: string) => v
      }
    },
    yAxis: { type: 'value', name: 'NAV', scale: true },
    series: seriesDefs,
  }
  navChart.setOption(option)
  navChart.resize()
}

function renderProbChart() {
  if (!probRef.value) return
  if (!probChart) probChart = echarts.init(probRef.value)

  const key = probAsset.value
  const arr = probSeries[key] || []
  const dates = arr.map(d => d.date)
  // 语义标签：2=上涨，1=中性，0=下跌
  const fall = arr.map(d => d.prob0 ?? null)
  const flat = arr.map(d => d.prob1 ?? null)
  const rise = arr.map(d => d.prob2 ?? null)

  // 颜色定义（绿=下跌，灰=中性，红=上涨）
  const COLOR_DOWN = '#34A853'   // Google green
  const COLOR_NEUT = '#9E9E9E'   // Gray 500
  const COLOR_UP   = '#EA4335'   // Google red
  const AREA_ALPHA = 0.18        // 面积透明度

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'axis' },
    legend: {
      type: 'scroll',
      data: ['下跌', '中性', '上涨'],
      top: 6,
      left: 16,
      right: 16,
      padding: [2, 0, 2, 0]
    },
    grid: {
      left: 56, right: 24,
      top: 64,
      bottom: 64,
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: dates,
      axisLabel: { hideOverlap: true, margin: 12 }
    },
    yAxis: { type: 'value', name: '概率', min: 0, max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
    series: [
      // 用有序图例：下跌→中性→上涨，堆叠面积更直观
     {
       name: '下跌', type: 'line', stack: 'p', showSymbol: false, smooth: true, data: fall,
       lineStyle: { width: 2, color: COLOR_DOWN },
       itemStyle: { color: COLOR_DOWN },
       areaStyle: { color: COLOR_DOWN + Math.round(AREA_ALPHA * 255).toString(16).padStart(2, '0') } // 简易透明度
     },
     {
       name: '中性', type: 'line', stack: 'p', showSymbol: false, smooth: true, data: flat,
       lineStyle: { width: 2, color: COLOR_NEUT },
       itemStyle: { color: COLOR_NEUT },
       areaStyle: { color: COLOR_NEUT + Math.round(AREA_ALPHA * 255).toString(16).padStart(2, '0') }
     },
     {
       name: '上涨', type: 'line', stack: 'p', showSymbol: false, smooth: true, data: rise,
       lineStyle: { width: 2, color: COLOR_UP },
       itemStyle: { color: COLOR_UP },
       areaStyle: { color: COLOR_UP + Math.round(AREA_ALPHA * 255).toString(16).padStart(2, '0') }
     },
    ],
  }
  probChart.setOption(option)
  probChart.resize()
}

// ----- 刷新并渲染 -----
async function refreshAll() {
  try {
    error.value = ''
    await Promise.all([fetchNav(), fetchPredict()])
    await nextTick()
    renderNavChart()
    renderProbChart()
  } catch (e: any) {
    error.value = e?.message || String(e)
    console.error(e)
  }
}

onMounted(async () => {
  // 初始拉一次数据（daily 模式）
  await refreshAll()
  // 自适应窗口
  const onResize = () => {
    navChart?.resize()
    probChart?.resize()
  }
  window.addEventListener('resize', onResize)
})

// 当 probAsset 切换，重绘概率图
watch(probAsset, () => renderProbChart())
</script>

<style scoped>
.factor-trend {
  padding: 16px;
}
.toolbar {
  margin-bottom: 12px;
  background: #fafafa;
  border: 1px solid #eee;
  padding: 12px;
  border-radius: 8px;
}
.toolbar .row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.toolbar label {
  color: #666;
  font-size: 14px;
}
.toolbar input[type="date"],
.toolbar select {
  padding: 6px 8px;
  border: 1px solid #ddd;
  border-radius: 6px;
}
.toolbar button {
  padding: 6px 10px;
  border: none;
  border-radius: 6px;
  background: #409eff;
  color: #fff;
  cursor: pointer;
}
.toolbar button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.toolbar .meta {
  color: #888;
  font-size: 13px;
  margin-left: 8px;
}
.charts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.chart-card {
  border: 1px solid #eee;
  border-radius: 8px;
  background: #fff;
  padding: 8px 8px 12px 8px;
}
.chart-title {
  font-size: 14px;
  color: #444;
  margin: 4px 0 8px;
}
.chart {
  width: 100%;
  height: 420px;
}
.error {
  margin-top: 12px;
  color: #d93025;
  background: #fee;
  padding: 8px 10px;
  border: 1px solid #f9caca;
  border-radius: 6px;
}
@media (max-width: 1100px) {
  .charts {
    grid-template-columns: 1fr;
  }
}
</style>
