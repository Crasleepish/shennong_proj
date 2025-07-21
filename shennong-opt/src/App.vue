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
    </section>

    <!-- 2. 优化组合 -->
    <section v-if="syncFinished">
      <h2 class="text-xl font-bold">二、优化当日组合</h2>
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

      <div class="mt-2" :style="{ color: parseFloat(deviationRateDisplay) > 0.12 ? 'red' : 'black' }">
        当前偏离度：{{ deviationRateDisplay }}<span v-if="parseFloat(deviationRateDisplay) > 0.12">，建议调仓</span>
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

const deviationRateRaw = computed(() => {
  const threshold = 3
  let total = 0
  let count = 0
  for (const row of mergedHoldings.value) {
    const target = parseFloat(row.target_percent || 0)
    const current = parseFloat(row.current_percent || 0)
    if (current >= threshold) {
      const ratio = current > target ? Math.max(current - target, 0) / current : 0
      if (isFinite(ratio)) {
        total += ratio
        count += 1
      }
    }
  }
  return count > 0 ? total / count : 0
})

const deviationRateDisplay = computed(() => deviationRateRaw.value.toFixed(2))


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

</script>

<style scoped>
.el-table {
  font-size: 13px;
}
</style>
