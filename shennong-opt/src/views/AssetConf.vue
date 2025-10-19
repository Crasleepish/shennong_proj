<template>
  <div class="p-4 space-y-4">
    <h2 class="text-xl font-bold">目标资产管理</h2>

    <!-- 组合ID 与操作按钮 -->
    <div class="flex items-center gap-3">
      <span>组合ID：</span>
      <el-input v-model.number="portfolioId" style="width: 160px" placeholder="例如 1" />
      <el-button type="primary" :loading="loading" @click="onQuery">查询</el-button>
      <el-button type="success" :loading="saving" @click="onSubmit">提交修改</el-button>
      <el-button @click="formatAll">格式化JSON</el-button>
    </div>

    <!-- portfolio_id 显示（可编辑） -->
    <div>
      <span class="text-sm text-gray-600">返回数据中的 portfolio_id：</span>
      <el-input v-model.number="respPortfolioId" style="width: 200px" />
    </div>

    <!-- asset_source_map -->
    <div>
      <div class="font-semibold mb-1">asset_source_map</div>
      <el-input
        v-model="assetSourceMapText"
        type="textarea"
        :autosize="{ minRows: 6 }"
        placeholder='例如：{"008114.OF":"factor","Au99.99.SGE":"index"}'
      />
    </div>

    <!-- code_factors_map -->
    <div>
      <div class="font-semibold mb-1">code_factors_map</div>
      <el-input
        v-model="codeFactorsMapText"
        type="textarea"
        :autosize="{ minRows: 8 }"
        placeholder='例如：{"008114.OF":["MKT","SMB","HML","QMJ"],"Au99.99.SGE":["GOLD"]}'
      />
    </div>

    <!-- view_codes -->
    <div>
      <div class="font-semibold mb-1">view_codes</div>
      <el-input
        v-model="viewCodesText"
        type="textarea"
        :autosize="{ minRows: 5 }"
        placeholder='例如：["H11004.CSI","Au99.99.SGE","008114.OF"]'
      />
    </div>

    <!-- view_codes -->
    <div>
      <div class="font-semibold mb-1">params</div>
      <el-input
        v-model="paramsText"
        type="textarea"
        :autosize="{ minRows: 5 }"
        placeholder='例如：{"post_view_tau": 0.07, "alpha": 0.1, "variance": 0.01}'
      />
    </div>
  </div>
  <!-- 查找全市场支撑资产 -->
  <div class="mt-8 border-t pt-4">
    <h3 class="font-semibold mb-2">查找全市场支撑资产</h3>

    <div class="flex items-center gap-3">
      <span>截面日期（可留空）：</span>
      <el-date-picker
        v-model="supportAsofDate"
        type="date"
        placeholder="选择日期或留空"
      />

      <span></span>
      <el-button
        type="primary"
        :loading="findingSupport"
        @click="onFindSupportAssets"
      >
        {{ findingSupport ? '查找中...' : '开始查找' }}
      </el-button>
    </div>

    <!-- 结果展示 -->
    <div v-if="supportAssets.length" class="mt-3">
      <div class="text-sm text-gray-600 mb-1">候选支撑资产代码：</div>
      <el-space wrap>
        <el-tag v-for="code in supportAssets" :key="code">{{ code }}</el-tag>
      </el-space>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'

// 复用项目中已有的环境变量（与 App.vue 同源）
const baseUrl = import.meta.env.VITE_API_BASE_URL as string
const authHeader = import.meta.env.VITE_API_AUTH_HEADER as string

// 输入/状态
const portfolioId = ref<number>(1)
const loading = ref(false)
const saving = ref(false)

// 文本框（原样展示 JSON 字符串）
const assetSourceMapText = ref<string>('')
const codeFactorsMapText = ref<string>('')
const viewCodesText = ref<string>('')
const paramsText = ref<string>('')

// 接口响应中的 portfolio_id
const respPortfolioId = ref<number | undefined>(undefined)

// --- 工具与校验 ---
function safeParseJSON<T = unknown>(txt: string, nameForMsg: string): T | null {
  try {
    const v = JSON.parse(txt)
    return v as T
  } catch (e) {
    ElMessage.error(`${nameForMsg} 不是有效的 JSON`)
    return null
  }
}

function prettyJSON(v: unknown): string {
  try {
    return JSON.stringify(v, null, 2)
  } catch {
    return ''
  }
}

function isRecordStringString(v: unknown): v is Record<string, string> {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return false
  return Object.entries(v as Record<string, unknown>).every(
    ([k, val]) => typeof k === 'string' && (typeof val === 'string' || (Array.isArray(val) && (val as unknown[]).every(i => typeof i === 'string')))
  )
}

function isRecordStringStringArray(v: unknown): v is Record<string, string[]> {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return false
  return Object.entries(v as Record<string, unknown>).every(
    ([k, val]) => typeof k === 'string' && Array.isArray(val) && (val as unknown[]).every(i => typeof i === 'string')
  )
}

function isStringArray(v: unknown): v is string[] {
  return Array.isArray(v) && v.every(i => typeof i === 'string')
}

function formatAll() {
  const a = safeParseJSON(assetSourceMapText.value || '{}', 'asset_source_map')
  const b = safeParseJSON(codeFactorsMapText.value || '{}', 'code_factors_map')
  const c = safeParseJSON(viewCodesText.value || '[]', 'view_codes')
  const d = safeParseJSON(paramsText.value || '{}', 'params')
  if (a) assetSourceMapText.value = prettyJSON(a)
  if (b) codeFactorsMapText.value = prettyJSON(b)
  if (c) viewCodesText.value = prettyJSON(c)
  if (d) paramsText.value = prettyJSON(d)
}

async function queryAndGetRaw(pid?: number): Promise<any | null> {
  const id = pid ?? portfolioId.value
  if (!id || id <= 0) {
    ElMessage.warning('请先输入有效的组合ID')
    return null
  }

  try {
    const res = await fetch(`${baseUrl}/service/portfolio_assets_query/${id}`, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/json'
      }
    })
    if (!res.ok) {
      const msg = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status} ${msg}`)
    }

    return await res.json()   // 👈 返回原始数据体
  } catch (err) {
    console.error('queryAndGetRaw 失败：', err)
    ElMessage.error('查询失败，请查看控制台日志')
    return null
  }
}

// --- 查询 ---
async function onQuery() {
  if (!portfolioId.value || portfolioId.value <= 0) {
    ElMessage.warning('请先输入有效的组合ID')
    return
  }
  loading.value = true
  try {
    const data = await queryAndGetRaw(portfolioId.value)
    if (!data) return

    // 赋值 & 美化显示
    respPortfolioId.value = data?.portfolio_id
    assetSourceMapText.value = prettyJSON(data?.asset_source_map ?? {})
    codeFactorsMapText.value = prettyJSON(data?.code_factors_map ?? {})
    viewCodesText.value = prettyJSON(data?.view_codes ?? [])
    paramsText.value = prettyJSON(data?.params ?? {})

    ElMessage.success('查询成功')
  } finally {
    loading.value = false
  }
}

// --- 提交修改 ---
async function onSubmit() {
  // 解析文本框
  const assetSourceMap = safeParseJSON(assetSourceMapText.value || '{}', 'asset_source_map')
  const codeFactorsMap = safeParseJSON(codeFactorsMapText.value || '{}', 'code_factors_map')
  const viewCodes = safeParseJSON(viewCodesText.value || '[]', 'view_codes')
  const params = safeParseJSON(paramsText.value || '{}', 'params')

  if (!assetSourceMap || !codeFactorsMap || !viewCodes || !params) return

  // 基本类型校验（可按需再严格一些）
  if (!isRecordStringString(assetSourceMap)) {
    ElMessage.error('asset_source_map 必须是 { [code: string]: string }')
    return
  }
  if (!isRecordStringStringArray(codeFactorsMap)) {
    ElMessage.error('code_factors_map 必须是 { [code: string]: string[] }')
    return
  }
  if (!isStringArray(viewCodes)) {
    ElMessage.error('view_codes 必须是 string[]')
    return
  }
  if (!isRecordStringString(params)) {
    ElMessage.error('params 必须是 { [code: string]: string or string[] }')
    return
  }

  const pid = portfolioId.value || respPortfolioId.value
  if (!pid || pid <= 0) {
    ElMessage.warning('组合ID缺失或无效，请先填写或查询')
    return
  }

  saving.value = true
  try {
    const res = await fetch(`${baseUrl}/service/portfolio_assets_upsert/${pid}`, {
      method: 'POST',
      headers: {
        'Authorization': authHeader, // 按你项目约定使用环境变量中的密钥
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        asset_source_map: assetSourceMap,
        code_factors_map: codeFactorsMap,
        view_codes: viewCodes,
        params: params
      })
    })
    if (!res.ok) {
      const msg = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status} ${msg}`)
    }
    ElMessage.success('提交成功')
  } catch (e) {
    console.error(e)
    ElMessage.error('提交失败，请查看控制台日志')
  } finally {
    saving.value = false
  }
}

// === 支撑资产查找 ===
const findingSupport = ref(false)
const supportAssets = ref<string[]>([])
const supportAsofDate = ref<Date | null>(null)

// 简单日期格式化：Date -> 'YYYY-MM-DD'
function formatDateYMD(d: Date | null | undefined): string | undefined {
  if (!d) return undefined
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}


async function onFindSupportAssets() {
  findingSupport.value = true
  supportAssets.value = []

  try {
    // 1) 读取 params（从“目标资产管理”的查询接口）
    const raw = await queryAndGetRaw()
    // 有的后端返回里才有 params；没有就视为可空
    const params = raw?.params ?? {}
    const blacklist: string[] | undefined = Array.isArray(params?.blacklist) ? params.blacklist : undefined
    const whitelist: string[] | undefined = Array.isArray(params?.whitelist) ? params.whitelist : undefined

    // 2) 组装 body
    const body: Record<string, unknown> = {}
    const asof = formatDateYMD(supportAsofDate.value)
    if (asof) body.asof_date = asof
    if (blacklist && blacklist.length) body.blacklist = blacklist
    if (whitelist && whitelist.length) body.whitelist = whitelist

    // 3) 请求后端
    const res = await fetch(`${baseUrl}/service/portfolio_assets/find_support_assets`, {
      method: 'POST',
      headers: {
        'Authorization': authHeader,       // 确保是后端期望的 Authorization 形式
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    })

    if (!res.ok) {
      const msg = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status} ${msg}`)
    }

    const data = await res.json()
    const arr = Array.isArray(data?.data) ? data.data : []
    supportAssets.value = arr

    ElMessage.success('支撑资产查找成功')
  } catch (e) {
    console.error('find_support_assets 失败：', e)
    ElMessage.error('查找失败，请查看控制台日志')
  } finally {
    findingSupport.value = false
  }
}
</script>
