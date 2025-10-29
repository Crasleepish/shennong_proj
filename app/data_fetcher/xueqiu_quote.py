# app/data_fetcher/xueqiu_quote.py
import logging
import akshare as ak
import math
import logging
from typing import Dict, Any, List, Tuple, Optional

import requests
import pandas as pd
from tqdm import tqdm

from app.data_fetcher.xueqiu_token import XueqiuTokenManager
from app.data_fetcher.xueqiu_token import XueqiuTokenManager

logger = logging.getLogger(__name__)

def stock_individual_spot_xq_safe(symbol: str, timeout: float = 8.0):
    """
    1) 带缓存 token 调用 AkShare 的雪球个股/指数/基金实时行情
    2) 失败或 4xx/空数据 → 强制刷新 token 重试一次
    """
    # 第一次用缓存/自动刷新
    token = XueqiuTokenManager.get_token(allow_refresh=True)
    try:
        df = ak.stock_individual_spot_xq(symbol=symbol, token=token, timeout=timeout)
        # 有些场景接口不会抛错，但 df 为空；这里一并兜底
        if df is None or (hasattr(df, "empty") and df.empty):
            raise RuntimeError("Empty result from Xueqiu API")
        return df
    except Exception as e1:
        logger.warning("Xueqiu API 调用失败，尝试强制刷新 token 后重试一次：%s", e1)
        token = XueqiuTokenManager.force_refresh()
        df = ak.stock_individual_spot_xq(symbol=symbol, token=token, timeout=timeout)
        if df is None or (hasattr(df, "empty") and df.empty):
            raise RuntimeError("Empty result after token refresh")
        return df


logger = logging.getLogger(__name__)

_XQ_LIST_URL = "https://stock.xueqiu.com/v5/stock/screener/quote/list.json"
_DEVICE_ID = "db51044c2cd4031de562070d95303379"
_PAGE_SIZE = 90

_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,ja;q=0.6,zh-TW;q=0.5",
    "origin": "https://xueqiu.com",
    "referer": "https://xueqiu.com/",
    "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/141.0.0.0 Safari/537.36"),
    "cache-control": "no-cache",
}

def _fetch_page(token: str, page: int, market_type: str, timeout: float) -> Tuple[int, List[Dict[str, Any]]]:
    """
    拉取单页；返回 (total_count, list)
    抛出异常由上层处理（用于触发强制刷新）
    """
    params = {
        "page": page,
        "size": _PAGE_SIZE,
        "order": "asc",
        "order_by": "symbol",
        "market": "CN",
        "type": market_type,
    }
    cookies = {
        "xq_a_token": token,
        "device_id": _DEVICE_ID,
    }
    resp = requests.get(_XQ_LIST_URL, headers=_HEADERS, cookies=cookies, params=params, timeout=timeout)
    resp.raise_for_status()
    j = resp.json()
    if not isinstance(j, dict) or j.get("error_code") != 0 or "data" not in j or "list" not in j["data"]:
        raise RuntimeError(f"Unexpected response: {j!r}")
    data = j["data"]
    total = int(data.get("count", 0) or 0)
    items = data.get("list", []) or []
    return total, items

def _normalize_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    选取并英文化字段；全部返回为基础数值/字符串（无“万/亿”单位），百分比用数值（例如 213.49）
    """
    out = []
    for x in items:
        row = {
            "symbol":              x.get("symbol"),
            "name":                x.get("name"),
            "price":               x.get("current"),
            "change":              x.get("chg"),
            "pct_chg":             x.get("percent"),               # 例如 213.49
            "ytd_pct":             x.get("current_year_percent"),  # 例如 213.49
            "volume":              x.get("volume"),                # 成交量（股）
            "amount":              x.get("amount"),                # 成交额（元）
            "turnover":            x.get("turnover_rate"),         # %
            "pe_ttm":              x.get("pe_ttm"),
            "div_yield":           x.get("dividend_yield"),        # %
            "market_cap":          x.get("market_capital"),        # 总市值（元）
        }
        out.append(row)
    return out

def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def stock_zh_a_xq_list(timeout: float = 30.0) -> pd.DataFrame:
    """
    获取沪深全部A股实时快照（雪球筛选器接口）
    - 分页遍历 size=90
    - 使用 XueqiuTokenManager.get_token()，失败一次后 force_refresh() 再试
    - 按 symbol 去重
    - 返回 DataFrame（英文字段名，数值列为纯数值类型）
    """
    # 1) 先用缓存 token 拉第1页；失败则强刷重试一次
    token = XueqiuTokenManager.get_token(allow_refresh=True)

    def _fetch_all_with_token(tok: str, market_type: str) -> List[Dict[str, Any]]:
        total, items = _fetch_page(tok, page=1, market_type=market_type, timeout=timeout)
        rows = _normalize_rows(items)
        if total <= _PAGE_SIZE:
            return rows
        total_pages = int(math.ceil(total / _PAGE_SIZE))

        try:
            page_iter = tqdm(range(2, total_pages + 1), desc="Fetching pages", ncols=80)
        except Exception:
            page_iter = range(2, total_pages + 1)

        for p in page_iter:
            _, items_p = _fetch_page(tok, page=p, market_type=market_type, timeout=timeout)
            rows.extend(_normalize_rows(items_p))
        return rows

    market_type_list = ["sha", "kcb", "sza", "shb", "szb", "bj"]

    try:
        rows = [row for market_type in market_type_list for row in _fetch_all_with_token(token, market_type)]
    except Exception as e1:
        logger.warning("首次使用缓存 token 抓取失败，尝试强制刷新 token 后重试一次：%s", e1)
        token = XueqiuTokenManager.force_refresh()
        rows = _fetch_all_with_token(token)

    # 2) 组装 DataFrame、去重、类型转换
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["symbol"], keep="last")
        num_cols = ["price", "change", "pct_chg", "ytd_pct", "volume", "amount", "turnover",
                    "pe_ttm", "div_yield", "market_cap"]
        df = _to_numeric(df, num_cols)
        # 强制列顺序（可读性更好）
        cols = ["symbol", "name", "price", "change", "pct_chg", "ytd_pct",
                "volume", "amount", "turnover", "pe_ttm", "div_yield", "market_cap"]
        df = df.reindex(columns=cols)
    return df