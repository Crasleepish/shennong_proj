<template>
  <div class="p-4 space-y-6">
    <!-- 1. 同步数据 -->
    <section>
      <h2 class="text-xl font-bold">一、同步数据</h2>
      <div class="flex space-x-4 items-center">
        <span>同步起始日期：</span>
        <el-date-picker v-model="syncStartDate" type="date" />
        <span>同步截止日期：</span>
        <el-date-picker v-model="syncEndDate" type="date" />
        <el-button :disabled="syncing" type="primary" @click="onSync">{{ syncing ? '同步中...' : '开始同步' }}</el-button>
        <el-button type="default" @click="onSkipSync">跳过</el-button>
      </div>
      <!-- 1.b 基本面数据同步：按季度 -->
      <div class="mt-6 space-y-2 border-t pt-4">
        <h3 class="font-semibold">基本面数据同步（按季度）</h3>

        <div class="flex flex-wrap items-center gap-3">
          <span>从</span>
          <!-- 起始：年份输入 -->
          <el-input-number
            v-model="fundStartYear"
            :min="1990" :max="2100" :step="1"
            placeholder="年份"
            controls-position="right"
            style="width:120px"
          />
          <span>年</span>

          <!-- 起始：季度选择 -->
          <el-select v-model="fundStartQuarter" placeholder="季度" style="width:120px">
            <el-option v-for="q in quarterOptions" :key="'s'+q.value" :label="q.label" :value="q.value" />
          </el-select>
          <span>季度</span>

          <span class="ml-4">到</span>

          <!-- 截止：年份输入 -->
          <el-input-number
            v-model="fundEndYear"
            :min="1990" :max="2100" :step="1"
            placeholder="年份"
            controls-position="right"
            style="width:120px"
          />
          <span>年</span>

          <!-- 截止：季度选择 -->
          <el-select v-model="fundEndQuarter" placeholder="季度" style="width:120px">
            <el-option v-for="q in quarterOptions" :key="'e'+q.value" :label="q.label" :value="q.value" />
          </el-select>
          <span>季度</span>

          <el-button
            type="primary"
            :disabled="fundSyncing"
            @click="onSyncFundamental"
          >
            {{ fundSyncing ? '同步中...' : '同步' }}
          </el-button>
        </div>

        <p class="text-xs text-gray-500">
          规则：Q1→YYYY0331，Q2→YYYY0630，Q3→YYYY0930，Q4→YYYY1231；将作为 <code>start_period</code>/<code>end_period</code> 传给后端。
        </p>
      </div>

      <div class="mt-6 space-y-2 border-t pt-4">
        <h3 class="font-semibold">同步 Beta（按日期）</h3>

        <div class="flex flex-wrap items-center gap-3">
          <span>开始日期：</span>
          <el-date-picker
            v-model="betaStartDate"
            type="date"
            placeholder="选择开始日期"
          />

          <span>结束日期：</span>
          <el-date-picker
            v-model="betaEndDate"
            type="date"
            placeholder="选择结束日期"
          />

          <el-button
            type="primary"
            :disabled="betaSyncing"
            @click="onSyncBeta"
          >
            {{ betaSyncing ? '同步中...' : '同步' }}
          </el-button>
        </div>

        <p class="text-xs text-gray-500">
          将把起止日期以 <code>YYYY-MM-DD</code> 传入 <code>start_date</code>/<code>end_date</code>；
          <code>asset_type</code> 固定为 <code>fund_info</code>。
        </p>
      </div>

    </section>

    <!-- 2. 优化组合 -->
    <section v-if="syncFinished">
      <h2 class="text-xl font-bold">二、优化当日组合</h2>
      <!-- 2.b 优化历史组合 -->
      <div class="mt-6 space-y-2 border-t pt-4">
        <h3 class="font-semibold">优化历史组合</h3>

        <div class="flex flex-wrap items-center gap-3">
          <span>开始日期：</span>
          <el-date-picker
            v-model="optHistStartDate"
            type="date"
            placeholder="可留空"
          />

          <span>截止至：</span>
          <el-date-picker
            v-model="optHistEndDate"
            type="date"
            placeholder="选择截止日期"
          />

          <el-button
            type="primary"
            :disabled="optHistRunning"
            @click="onOptimizeHistory"
          >
            {{ optHistRunning ? '优化中...' : '开始优化' }}
          </el-button>
        </div>

        <p class="text-xs text-gray-500">
          将从最近一次已优化的结果开始，依次补齐至所选日期；请求体：<code>{ end_date: 'YYYY-MM-DD' }</code>。
        </p>
      </div>
      <h3 class="font-semibold">优化今日组合</h3>
      <el-button :disabled="optimizing" type="primary" @click="onOptimize">{{ optimizing ? '优化中...' : '开始优化' }}</el-button>
      <el-button type="default" @click="onSkipOptimize">跳过</el-button>
    </section>

    <!-- 3. 调仓部分 -->
    <section v-if="optimizeFinished">
      <h2 class="text-xl font-bold">三、调仓</h2>

      <div class="flex items-center space-x-4">
        <span>选择查看日期：</span>
        <el-date-picker v-model="selectedDate" type="date" />
        <el-button type="info" @click="onViewLatestTransfer">查看最近调仓</el-button>
      </div>

      <div class="flex items-center space-x-4 mt-4">
        <span>最近调仓日：</span>
        <span>{{ latestRebalanceDate }}</span>
        <span>总价值：</span>
        <el-input-number v-model="totalValue" :min="0" @change="onTotalValueChange" />
      </div>

      <!-- 统一表格：当前持仓 + 目标权重 -->
      <div class="mt-4">
        <h3 class="font-semibold">当前与目标权重（可编辑）</h3>
        <el-table :data="mergedHoldings" border style="width: 100%" >
          <el-table-column v-for="col in columns" :key="col.prop" :prop="col.prop" :label="col.label" :width="col.width || undefined">
            <template #default="scope">
              <el-input
                v-model="scope.row[col.prop]"
                size="small"
                @change="() => onCellChange(scope.row, col.prop)"
              />
            </template>
          </el-table-column>
          <el-table-column fixed="right" label="操作" width="80">
            <template #default="scope">
              <el-button @click="removeRow(scope.$index)" type="danger" icon="el-icon-delete" circle size="small" />
            </template>
          </el-table-column>
        </el-table>
        <div class="mt-2">
          <el-button type="primary" @click="addRow">新增资产</el-button>
        </div>
      </div>

      <div class="mt-2" :style="{ color: parseFloat(deviationRateDisplay) > 0.002 ? 'red' : 'black' }">
        当前偏离度：{{ deviationRateDisplay }}<span v-if="parseFloat(deviationRateDisplay) > 0.002">，建议调仓</span>
      </div>

      <!-- 校验提示 -->
      <p v-if="!percentValid" class="text-red-500">⚠️ 目标占比之和需为 100%</p>

      <!-- 调仓按钮 -->
      <div class="mt-4">
        <el-button type="warning" @click="onCalculateRebalance">开始调仓</el-button>
      </div>

      <!-- 调仓结果 -->
      <div v-if="rebalanceResult.length" class="mt-6">
        <h3 class="font-semibold">调仓建议</h3>
        <el-table :data="rebalanceResult" border style="width: 100%">
          <el-table-column prop="from" label="卖出资产" />
          <el-table-column prop="to" label="买入资产" />
          <el-table-column prop="amount" label="卖出数量" />
        </el-table>
        <div class="mt-4">
          <el-button type="primary" @click="onConfirmRebalance">确认调仓</el-button>
        </div>
      </div>

    </section>
  </div>
</template>

<script setup>
import { ref, watch, computed } from 'vue'
import { ElMessage } from 'element-plus'

const baseUrl = import.meta.env.VITE_API_BASE_URL
const authHeader = import.meta.env.VITE_API_AUTH_HEADER

const syncing = ref(false)
const optimizing = ref(false)
const syncFinished = ref(false)
const optimizeFinished = ref(false)

const syncStartDate = ref('')
const syncEndDate = ref('')

const selectedDate = ref(null)
const latestRebalanceDate = ref('')
const totalValue = ref(0)
const totalValueDisplay = computed(() => totalValue.value.toFixed(2))

const mergedHoldings = ref([])
const rebalanceResult = ref([])
const percentValid = ref(true)

// === 基本面同步（季度）状态 ===
const fundSyncing = ref(false)
const fundStartYear = ref()
const fundStartQuarter = ref()
const fundEndYear = ref()
const fundEndQuarter = ref()

const quarterOptions = [
  { label: '一季度', value: 1 },
  { label: '二季度', value: 2 },
  { label: '三季度', value: 3 },
  { label: '四季度', value: 4 },
]

const betaSyncing = ref(false)
const betaStartDate = ref(null)
const betaEndDate = ref(null)

const optHistStartDate = ref(null)
const optHistEndDate = ref(null)
const optHistRunning = ref(false)

// 将 (year, quarter) → 'YYYYMMDD'（季末日）
function quarterEndDateStr(year, quarter) {
  if (!year || !quarter) return null
  const y = String(year).padStart(4, '0')
  switch (Number(quarter)) {
    case 1: return `${y}0331`
    case 2: return `${y}0630`
    case 3: return `${y}0930`
    case 4: return `${y}1231`
    default: return null
  }
}

// 由后端返回的 TE（偏离度）
const deviationRateRaw = ref(0)

const deviationRateDisplay = computed(() => deviationRateRaw.value.toFixed(6))


function formatDate(dateObj) {
  const d = new Date(dateObj)
  const year = d.getFullYear()
  const month = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function extractCodeFromLabel(label) {
  const match = label.match(/\(([^)]+)\)$/)
  return match ? match[1] : label
}

const columns = [
  { label: '资产', prop: 'asset' },
  { label: '名称', prop: 'name' },
  { label: '实际资产代码', prop: 'code' },
  { label: '价格', prop: 'price' },
  { label: '当前持仓量', prop: 'amount' },
  { label: '当前市值', prop: 'value' },
  { label: '当前占比(%)', prop: 'current_percent' },
  { label: '目标占比(%)', prop: 'target_percent' },
  { label: '目标市值', prop: 'target_value' },
  { label: '目标份额', prop: 'target_amount' }
]

function onTotalValueChange() {
  mergedHoldings.value.forEach(row => {
    updateTargetValues(row)
  })
}

function addRow() {
  mergedHoldings.value.push({
    asset: '', name: '', code: '', price: '', amount: '', value: '',
    current_percent: '', target_percent: '', target_value: '', target_amount: ''
  })
  recalculateTotal()
}

function removeRow(index) {
  mergedHoldings.value.splice(index, 1)
  recalculateTotal()
}

function onEdit(row, column, cell, event) {
  // 可选：聚焦时逻辑或格式处理
}

function onCellChange(row, prop) {
  debugger
  if (prop === 'code' && row.code) {
    fetch(`${baseUrl}/service/fund_names`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': authHeader
      },
      body: JSON.stringify({ codes: [row.code] })
    })
      .then(res => res.json())
      .then(data => {
        if (data && data[row.code]) {
          row.name = data[row.code]
        }
      })
      .catch(err => {
        console.error('自动填充基金名称失败', err)
      })

    fetch(`${baseUrl}/service/fund_prices`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': authHeader
      },
      body: JSON.stringify({ codes: [row.code] })
    })
      .then(res => res.json())
      .then(data => {
        if (data && data[row.code]) {
          row.price = data[row.code]
          recalculateTotal()
          if (row.amount) {
            row.value = (parseFloat(row.amount) * parseFloat(row.price)).toFixed(2)
            updateCurrentPercent(row)
          }
        }
      })
      .catch(err => {
        console.error('自动填充基金价格失败', err)
      })
  }

  if (prop === 'amount' && row.amount && row.price) {
    row.value = (parseFloat(row.amount) * parseFloat(row.price)).toFixed(2)
    updateCurrentPercent(row)
    recalculateTotal()
  }
  if (prop === 'target_percent') {
    updateTargetValues(row)
  }
  if (prop === 'price') {
    updateTargetValues(row)
  }
  validatePercent()
  mergedHoldings.value = [...mergedHoldings.value]
  triggerComputeDeviationDebounced()
}

function updateCurrentPercent(row) {
  const tval = mergedHoldings.value.reduce((s, r) => s + parseFloat(r.amount || 0) * parseFloat(r.price || 0), 0)
  if (tval > 0) {
    row.current_percent = ((parseFloat(row.value) / tval) * 100).toFixed(2)
  }
}

function updateTargetValues(row) {
  debugger
  const percent = parseFloat(row.target_percent || 0)
  const price = parseFloat(row.price || 0)
  const target_value = totalValue.value * (percent / 100)
  row.target_value = isNaN(target_value) ? '' : target_value.toFixed(2)
  if (price > 0) {
    row.target_amount = (target_value / price).toFixed(2)
  } else {
    row.target_amount = ''
  }
}


function recalculateTotal() {
  totalValue.value = mergedHoldings.value.reduce((s, r) => s + parseFloat(r.amount || 0) * parseFloat(r.price || 0), 0)
  // 更新所有目标市值和目标份额
  mergedHoldings.value.forEach(row => {
    updateTargetValues(row)
  })
}

function validatePercent() {
  const sum = mergedHoldings.value.reduce((s, row) => s + parseFloat(row.target_percent || 0), 0)
  const percentValid = Math.abs(sum - 100) < 0.01
  if (!percentValid) {
    ElMessage.warning(`目标占比合计为 ${sum.toFixed(2)}%，应为 100%`)
    return false
  }
  return true
}

async function onSync() {
  if (!syncStartDate.value || !syncEndDate.value) {
    ElMessage.warning('请先选择起始和截止日期')
    return
  }
  syncing.value = true
  try {
    await fetch(`${baseUrl}/fin_data/update_all`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': authHeader
      },
      body: JSON.stringify({
        mode: "realtime",
        start_date: formatDate(syncStartDate.value).replace(/-/g, ''),
        end_date: formatDate(syncEndDate.value).replace(/-/g, '')
      })
    })
    ElMessage.success('同步成功')
    syncFinished.value = true
  } catch (err) {
    console.error('同步失败', err)
    ElMessage.error('同步失败：控制台查看错误详情')
  } finally {
    syncing.value = false
  }
}

function onSkipSync() {
  syncFinished.value = true
  ElMessage.info('已跳过同步')
}

async function onSyncFundamental() {
  // 基础校验
  if (!fundStartYear.value || !fundStartQuarter.value || !fundEndYear.value || !fundEndQuarter.value) {
    ElMessage.warning('请完整选择起始与截止的 年份 和 季度')
    return
  }

  const start_period = quarterEndDateStr(fundStartYear.value, fundStartQuarter.value)
  const end_period = quarterEndDateStr(fundEndYear.value, fundEndQuarter.value)

  if (!start_period || !end_period) {
    ElMessage.error('季度参数不合法，请重新选择')
    return
  }
  if (Number(start_period) > Number(end_period)) {
    ElMessage.warning('起始季度不能晚于截止季度')
    return
  }

  fundSyncing.value = true
  try {
    const res = await fetch(`${baseUrl}/fin_data/fundamental/sync_one_period`, {
      method: 'POST',
      headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        start_period,
        end_period
      })
    })

    if (!res.ok) {
      const msg = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status} ${msg}`)
    }

    ElMessage.success(`基本面数据同步成功（${start_period} ~ ${end_period}）`)
  } catch (err) {
    console.error('基本面数据同步失败', err)
    ElMessage.error('基本面数据同步失败：请查看控制台日志')
  } finally {
    fundSyncing.value = false
  }
}

async function onOptimizeHistory() {
  if (!optHistEndDate.value) {
    ElMessage.warning('请先选择截止日期')
    return
  }
  const payload = {
    end_date: formatDate(optHistEndDate.value)
  }
  if (optHistStartDate.value) {
    payload.start_date = formatDate(optHistStartDate.value)
  }

  optHistRunning.value = true
  try {
    const res = await fetch(`${baseUrl}/service/portfolio_opt_hist`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': authHeader
      },
      body: JSON.stringify(payload)
    })
    if (!res.ok) {
      const msg = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status} ${msg}`)
    }
    ElMessage.success('历史组合优化完成')
  } catch (err) {
    console.error('历史组合优化失败', err)
    ElMessage.error('历史组合优化失败：请查看控制台日志')
  } finally {
    optHistRunning.value = false
  }
}

async function onSyncBeta() {
  if (!betaStartDate.value || !betaEndDate.value) {
    ElMessage.warning('请先选择开始和结束日期')
    return
  }

  const start_date = formatDate(betaStartDate.value)  // 复用现有工具，输出 YYYY-MM-DD
  const end_date = formatDate(betaEndDate.value)

  // 简单时序校验
  if (new Date(start_date) > new Date(end_date)) {
    ElMessage.warning('开始日期不能晚于结束日期')
    return
  }

  betaSyncing.value = true
  try {
    const res = await fetch(`${baseUrl}/fin_data/dynamic_beta`, {
      method: 'POST',
      headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        asset_type: 'fund_info',
        start_date,
        end_date
      })
    })

    if (!res.ok) {
      const msg = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status} ${msg}`)
    }

    ElMessage.success(`Beta 同步成功（${start_date} ~ ${end_date}）`)
  } catch (err) {
    console.error('同步 Beta 失败', err)
    ElMessage.error('同步 Beta 失败：请查看控制台日志')
  } finally {
    betaSyncing.value = false
  }
}

async function onOptimize() {
  optimizing.value = true
  try {
    const res = await fetch(`${baseUrl}/service/portfolio_opt_rt`, {
      method: 'POST',
      headers: { 'Authorization': authHeader }
    })
    const data = await res.json()
    optimizeFinished.value = true
    await loadHoldingsAndTargets(data)
    ElMessage.success('优化完成')
  } catch {
    ElMessage.error('优化失败')
  } finally {
    optimizing.value = false
  }
}

function onSkipOptimize() {
  optimizeFinished.value = true
  ElMessage.info('已跳过优化')
}

async function loadHoldingsAndTargets(weightsData) {
  const holdingsRes = await fetch(`${baseUrl}/service/current_position`, {
    headers: { 'Authorization': authHeader }
  })
  const holdings = await holdingsRes.json()

  // 计算总价值
  totalValue.value = holdings.reduce((s, r) => s + r.amount * r.price, 0)

  // 构建 asset -> [rows] 映射（支持多个 code 对应同一个 asset）
  const assetHoldingsMap = {}
  for (const h of holdings) {
    if (!assetHoldingsMap[h.asset]) {
      assetHoldingsMap[h.asset] = []
    }
    assetHoldingsMap[h.asset].push(h)
  }

  const convertedWeights = convertWeightsToPercent(weightsData.weights)

  const allAssets = new Set([...Object.keys(assetHoldingsMap), ...Object.keys(convertedWeights)])

  mergedHoldings.value = []

  for (const asset of allAssets) {
    const target_percent = parseFloat(convertedWeights[asset]) || 0
    const target_value = totalValue.value * (target_percent / 100)

    const holdingsList = assetHoldingsMap[asset] || [{}]

    for (const h of holdingsList) {
      const code = h.code || ''
      const price = h.price || ''
      const amount = h.amount || ''
      const name = h.name || ''
      const value = amount && price ? amount * price : ''
      const current_percent = value ? (value / totalValue.value * 100).toFixed(2) : ''
      const target_amount = code && price ? target_value / price : ''

      mergedHoldings.value.push({
        asset,
        name,
        code,
        price,
        amount,
        value,
        current_percent,
        target_percent: target_percent.toFixed(2),
        target_value: target_value.toFixed(2),
        target_amount: target_amount ? target_amount.toFixed(2) : ''
      })
    }
  }

  validatePercent()
  triggerComputeDeviationDebounced(0)
}

async function onCalculateRebalance() {
  if (!validatePercent()) {
    ElMessage.error('目标权重之和必须为 100%')
    return
  }
  const cleaned = mergedHoldings.value.map(row => ({
    ...row,
    amount: parseFloat(row.amount || 0),
    current_percent: parseFloat(row.current_percent || 0),
    target_amount: parseFloat(row.target_amount || 0),
    target_percent: parseFloat(row.target_percent || 0),
  })).filter(row => {
    const bothZero = row.amount === 0 && row.target_amount === 0
    const valid = row.code && parseFloat(row.price) > 0
    return !bothZero && valid
  })

  try {
    const res = await fetch(`${baseUrl}/service/calculate_rebalance`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': authHeader
      },
      body: JSON.stringify({ merged: cleaned })
    })
    const rawResult = await res.json()
    console.log('调仓建议:', rawResult)
    const codeSet = new Set()
    rawResult.forEach(item => {
      const fromCode = extractCodeFromLabel(item.from)
      const toCode = extractCodeFromLabel(item.to)
      if (fromCode) codeSet.add(fromCode)
      if (toCode) codeSet.add(toCode)
    })
    const codeList = Array.from(codeSet)
    const nameRes = await fetch(`${baseUrl}/service/fund_names`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': authHeader
      },
      body: JSON.stringify({ codes: codeList })
    })
    const nameMap = await nameRes.json()
    rebalanceResult.value = rawResult.map(item => {
      const fromCode = extractCodeFromLabel(item.from)
      const toCode = extractCodeFromLabel(item.to)
      const fromName = nameMap[fromCode] || ''
      const toName = nameMap[toCode] || ''
      return {
        ...item,
        from: `${fromName}(${fromCode})`,
        to: `${toName}(${toCode})`
      }
    })
    ElMessage.success('调仓建议已生成')
  } catch (err) {
    console.error('调仓计算失败', err)
    ElMessage.error('调仓计算失败')
  }
}

function onConfirmRebalance() {
  debugger
  const validRows = mergedHoldings.value.filter(r => {
    return r.asset && r.code && r.name && parseFloat(r.target_amount) > 0
  })

  if (validRows.length === 0) {
    ElMessage.warning('无有效持仓数据，无法提交')
    return
  }

  fetch(`${baseUrl}/service/submit_rebalance`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': authHeader
    },
    body: JSON.stringify({
      rebalance: rebalanceResult.value,
      new_holdings: validRows.map(r => ({
        asset: r.asset,
        code: r.code,
        name: r.name,
        amount: parseFloat(r.target_amount || 0)
      }))
    })
  })
    .then(res => res.json())
    .then(data => {
      ElMessage.success('调仓记录已保存')
    })
    .catch(err => {
      console.error('调仓记录提交失败', err)
      ElMessage.error('调仓提交失败')
    })
}

async function onViewLatestTransfer() {
  try {
    if (!selectedDate.value) {
      ElMessage.warning('请先选择日期')
      return
    }
    const dateStr = formatDate(selectedDate.value)
    const res = await fetch(`${baseUrl}/service/weights?date=${dateStr}`, {
      headers: { 'Authorization': authHeader }
    })
    const data = await res.json()
    await loadHoldingsAndTargets(data)
    ElMessage.success(`已加载 ${dateStr} 对应组合权重`)
  } catch (err) {
    console.error('获取组合权重失败', err)
    ElMessage.error('获取组合权重失败')
  }
}

// 替换展示用：优化结果中的目标占比字段（如 0.251 → 25.1）
function convertWeightsToPercent(weightsObj) {
  const converted = {}
  for (const k in weightsObj) {
    converted[k] = (weightsObj[k] * 100).toFixed(2)
  }
  return converted
}

function buildWeightMapsFromTable() {
  const current_w = {}
  const target_w = {}

  for (const row of mergedHoldings.value) {
    const asset = (row.asset || '').trim()
    if (!asset) continue

    const cp = parseFloat(row.current_percent || 0) // 百分数
    const tp = parseFloat(row.target_percent || 0)  // 百分数

    // 百分数 -> 权重（0~1），同名 asset 累加
    const cw = isFinite(cp) ? cp / 100 : 0
    const tw = isFinite(tp) ? tp / 100 : 0

    current_w[asset] = (current_w[asset] || 0) + cw
    target_w[asset]  = (target_w[asset]  || 0) + tw
  }

  return { current_w, target_w }
}

let _computeTimer = null
function triggerComputeDeviationDebounced(delay = 500) {
  clearTimeout(_computeTimer)
  _computeTimer = setTimeout(computeDeviation, delay)
}

async function computeDeviation() {
  // 需要有选定日期与至少一行有效数据
  if (!selectedDate.value || !mergedHoldings.value?.length) return
  const trade_date = formatDate(selectedDate.value) // YYYY-MM-DD
  const { current_w, target_w } = buildWeightMapsFromTable()

  try {
    const res = await fetch(`${baseUrl}/service/compute_diverge`, {
      method: 'POST',
      headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        portfolio_id: 1,
        trade_date,
        current_w,
        target_w
      })
    })
    // 容错：不同后端字段名
    const data = await res.json()
    const val = data?.data ?? data?.te ?? data?.value
    if (typeof val === 'number' && isFinite(val)) {
      deviationRateRaw.value = val
    } else {
      // 如果没有明确数值，保留原值并给出提示
      console.warn('compute_diverge: 返回值未包含有效数值字段', data)
    }
  } catch (e) {
    console.error('compute_diverge 调用失败', e)
  }
}

watch(selectedDate, () => {
  if (mergedHoldings.value?.length) {
    triggerComputeDeviationDebounced(0)
  }
})

</script>

<style scoped>
.el-table {
  font-size: 13px;
}
</style>
