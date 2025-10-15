import threading
import pandas as pd
import akshare as ak
from datetime import datetime
from app.database import get_db
from app.models.index_models import IndexHist
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
import logging
from typing import Optional, Dict, Tuple
import tushare as ts

logger = logging.getLogger(__name__)

# ===== 实时指数行情缓存：默认 5 分钟 =====
REALTIME_CACHE_TTL_SECONDS = 300  # 5 minutes


class IndexDataReader:
    """指数数据读取器（单例 + TTL缓存 + AkShare→Tushare 兜底）"""

    _instance = None
    _lock = threading.Lock()  # 单例创建锁

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # 双重检查
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, realtime_cache_ttl: int = REALTIME_CACHE_TTL_SECONDS):
        # 防止单例重复初始化
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # ---- 实时行情缓存（按 index_code 分桶）----
        # { index_code: (df, cache_time) }
        self._rt_cache: Dict[str, Tuple[pd.DataFrame, pd.Timestamp]] = {}
        self._rt_cache_ttl = int(realtime_cache_ttl)

    # ---------- 工具 ----------
    @staticmethod
    def _now_ts() -> pd.Timestamp:
        return pd.Timestamp.now(tz=None)

    def _rt_cache_valid(self, index_code: str) -> bool:
        item = self._rt_cache.get(index_code)
        if not item:
            return False
        _, t = item
        return (self._now_ts() - t).total_seconds() < self._rt_cache_ttl

    def clear_realtime_cache(self, index_code: Optional[str] = None):
        """清空实时缓存；若提供 index_code，则仅清该指数"""
        if index_code is None:
            self._rt_cache.clear()
        else:
            self._rt_cache.pop(index_code, None)

    # ---------- 交易日 ----------
    def _get_current_trade_date(self) -> pd.Timestamp:
        today = pd.Timestamp.today().normalize()
        trade_dates = TradeCalendarReader.get_trade_dates(end=today.strftime("%Y-%m-%d"))
        return trade_dates[-1] if today not in trade_dates else today

    # ---------- 最新收盘价（不缓存） ----------
    def fetch_latest_close_prices(self, index_code, latest_trade_date=None) -> pd.DataFrame:
        if latest_trade_date is None:
            latest_trade_date = self._get_current_trade_date()
        with get_db() as db:
            query = db.query(IndexHist.date).filter(IndexHist.index_code == index_code)
            if latest_trade_date:
                query = query.filter(IndexHist.date <= latest_trade_date)
            latest_date = query.order_by(IndexHist.date.desc()).limit(1).scalar()
            if not latest_date:
                logger.warning(f"未获取到指数 {index_code} 的历史数据")
                return pd.DataFrame(columns=["index_code", "close", "vol", "amount"])

            row = (
                db.query(IndexHist)
                .filter(IndexHist.index_code == index_code, IndexHist.date == latest_date)
                .first()
            )
            return pd.DataFrame(
                [
                    {
                        "index_code": row.index_code,
                        "date": row.date,
                        "close": row.close,
                        "vol": row.volume,
                        "amount": row.amount,
                    }
                ]
            )

    # ---------- 代码规范化 ----------
    @staticmethod
    def _to_symbol_for_ak(index_code: str) -> str:
        """
        ak.index_zh_a_hist_min_em 需要不带后缀的 symbol，例如 '000985'、'399001'
        """
        dot = index_code.rfind(".")
        return index_code[:dot] if dot > 0 else index_code

    @staticmethod
    def _to_ts_index_code(index_code: str) -> str:
        """
        规范为 Tushare realtime_quote 需要的 ts_code：
        - 若已是 *.SH/*.SZ，原样返回
        - 若是 *.CSI / *.INDEX / *.CI / *.CS 或无后缀：
            - 以 '399' 开头 -> '.SZ'
            - 其他 -> '.SH'
        例：'000985.CSI' -> '000985.SH'；'399001.CSI' -> '399001.SZ'
        """
        if "." in index_code:
            root, suf = index_code.split(".", 1)
            suf = suf.upper()
            if suf in ("SH", "SZ"):
                return f"{root}.{suf}"
            if suf in ("CSI", "INDEX", "CI", "CS"):
                if root.startswith("399"):
                    return f"{root}.SZ"
                return f"{root}.SH"
            # 未知后缀，尽量推断
            if root.startswith("399"):
                return f"{root}.SZ"
            return f"{root}.SH"
        # 无后缀，推断
        if index_code.startswith("399"):
            return f"{index_code}.SZ"
        return f"{index_code}.SH"

    # ---------- 实时行情（AkShare -> Tushare Fallback，带 TTL 缓存） ----------
    def fetch_realtime_prices(
        self,
        index_code: str,
        *,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        获取指数当日盘中最新数据，统一返回列：
        ["index_code", "close", "vol", "amount"]
        流程：
          1) AkShare index_zh_a_hist_min_em (1分钟线，从 09:30:00 起)
          2) 失败或为空时，Tushare ts.realtime_quote(ts_code=..., src='sina') 兜底
        成功则写入 TTL 缓存（默认 5 分钟，按 index_code 分桶）。
        """
        if use_cache and not force_refresh and self._rt_cache_valid(index_code):
            logger.info("命中指数实时缓存（%s，≤%ds）。", index_code, self._rt_cache_ttl)
            return self._rt_cache[index_code][0].copy()

        trade_date = self._get_current_trade_date()
        start_dt = trade_date.strftime("%Y-%m-%d 09:30:00")
        symbol = self._to_symbol_for_ak(index_code)

        # ===== 尝试 1：AkShare =====
        df_out = pd.DataFrame(columns=["index_code", "close", "vol", "amount"])
        got = False
        try:
            df = ak.index_zh_a_hist_min_em(symbol=symbol, period="1", start_date=start_dt)
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                df_out = pd.DataFrame(
                    [
                        {
                            "index_code": index_code,
                            "close": latest.get("收盘"),
                            "vol": latest.get("成交量"),
                            "amount": latest.get("成交额"),
                        }
                    ]
                ).dropna(subset=["index_code", "close"])
                if not df_out.empty:
                    got = True
        except Exception as e:
            logger.warning("AkShare 指数实时失败(%s)：%s", index_code, e)

        # ===== 尝试 2：Tushare realtime_quote 兜底 =====
        if not got:
            try:
                ts_code = self._to_ts_index_code(index_code)
                df_ts = ts.realtime_quote(ts_code=ts_code, src='sina')
                if isinstance(df_ts, pd.DataFrame) and not df_ts.empty:
                    # 大小写兼容
                    cols = {
                        "TS_CODE": "stock_code", "ts_code": "stock_code",
                        "PRICE": "close", "price": "close",
                        "VOLUME": "vol", "volume": "vol",
                        "AMOUNT": "amount", "amount": "amount",
                    }
                    df_ts = df_ts.rename(columns=cols)
                    row = df_ts.iloc[0]
                    price = row.get("close", 0) or 0
                    if float(price) > 0:  # 09:00:00 前常出现 0，视为无效
                        df_out = pd.DataFrame(
                            [
                                {
                                    "index_code": index_code,
                                    "close": float(row.get("close")),
                                    "vol": row.get("vol"),
                                    "amount": row.get("amount"),
                                }
                            ]
                        ).dropna(subset=["index_code", "close"])
                        if not df_out.empty:
                            got = True
            except Exception as e:
                logger.warning("Tushare realtime_quote 兜底失败(%s)：%s", index_code, e)

        if not got:
            logger.error("指数实时获取失败：AkShare -> Tushare 均未取到有效数据（%s）。返回空表。", index_code)
            return pd.DataFrame(columns=["index_code", "close", "vol", "amount"])

        # 写缓存
        if use_cache or force_refresh:
            self._rt_cache[index_code] = (df_out.copy(), self._now_ts())
            logger.info("指数实时行情已刷新缓存：%s @ %s", index_code, self._rt_cache[index_code][1])

        return df_out