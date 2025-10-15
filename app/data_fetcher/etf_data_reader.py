import threading
import pandas as pd
import tushare as ts
from app.models.etf_model import EtfInfo, EtfHist
from app.database import get_db
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# 实时ETF行情缓存：默认 5 分钟
REALTIME_CACHE_TTL_SECONDS = 300  # 5 minutes


class EtfDataReader:
    """ETF 数据读取器（单例 + TTL 缓存 + tushare 主源dc / 兜底sina）"""

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

        # ---- 实时行情缓存（按 etf_code 分桶）----
        # { etf_code: (df, cache_time) }
        self._rt_cache: Dict[str, Tuple[pd.DataFrame, pd.Timestamp]] = {}
        self._rt_cache_ttl = int(realtime_cache_ttl)

    # ---------- 工具 ----------
    @staticmethod
    def _now_ts() -> pd.Timestamp:
        return pd.Timestamp.now(tz=None)

    def _rt_cache_valid(self, etf_code: str) -> bool:
        item = self._rt_cache.get(etf_code)
        if not item:
            return False
        _, t = item
        return (self._now_ts() - t).total_seconds() < self._rt_cache_ttl

    def clear_realtime_cache(self, etf_code: Optional[str] = None):
        """清空实时缓存；若提供 etf_code，则仅清该ETF"""
        if etf_code is None:
            self._rt_cache.clear()
        else:
            self._rt_cache.pop(etf_code, None)

    # ---------- 交易日 ----------
    def _get_current_trade_date(self) -> pd.Timestamp:
        today = pd.Timestamp.today().normalize()
        trade_dates = TradeCalendarReader.get_trade_dates(end=today.strftime("%Y-%m-%d"))
        return trade_dates[-1] if today not in trade_dates else today

    # ---------- 最新收盘价（不缓存） ----------
    def fetch_latest_close_prices(self, etf_code: str, latest_trade_date=None) -> pd.DataFrame:
        if latest_trade_date is None:
            latest_trade_date = self._get_current_trade_date()
        with get_db() as db:
            query = db.query(EtfHist.date).filter(EtfHist.etf_code == etf_code)
            if latest_trade_date:
                query = query.filter(EtfHist.date <= latest_trade_date)
            latest_date = query.order_by(EtfHist.date.desc()).limit(1).scalar()
            if not latest_date:
                logger.warning(f"未获取到ETF {etf_code} 的历史数据")
                return pd.DataFrame(columns=["etf_code", "close", "vol", "amount"])

            row = (
                db.query(EtfHist)
                .filter(EtfHist.etf_code == etf_code, EtfHist.date == latest_date)
                .first()
            )
            return pd.DataFrame(
                [{
                    "etf_code": row.etf_code,
                    "date": row.date,
                    "close": row.close,
                    "vol": row.volume,
                    "amount": row.amount
                }]
            )

    # ---------- 实时行情（主源dc + fallback sina + TTL缓存） ----------
    def fetch_realtime_prices(
        self,
        etf_code: str,
        *,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        使用 tushare 实时行情接口获取单个 ETF 的最新成交价、成交量与成交额。
        优先 src='dc'，失败或为空时 fallback src='sina'。
        返回列：["etf_code", "close", "vol", "amount"]
        """
        if use_cache and not force_refresh and self._rt_cache_valid(etf_code):
            logger.info("命中ETF实时缓存（%s，≤%ds）。", etf_code, self._rt_cache_ttl)
            return self._rt_cache[etf_code][0].copy()

        df_out = pd.DataFrame(columns=["etf_code", "close", "vol", "amount"])
        got = False

        # ===== 尝试 1：tushare dc 源 =====
        try:
            df = ts.realtime_quote(ts_code=etf_code, src='dc')
            if isinstance(df, pd.DataFrame) and not df.empty:
                row = df.iloc[0]
                df_out = pd.DataFrame([{
                    "etf_code": row.get("TS_CODE", etf_code),
                    "date": row.get("DATE"),
                    "close": float(row.get("PRICE", 0.0)) / 10.0,   # 单位厘 -> 元
                    "vol": float(row.get("VOLUME", 0.0)) * 100.0,   # 手 -> 份
                    "amount": float(row.get("AMOUNT", 0.0))
                }]).dropna(subset=["etf_code", "close"])
                if not df_out.empty:
                    got = True
                    source = "dc"
        except Exception as e:
            logger.warning(f"ETF实时行情(dc)失败 {etf_code}: {e}")

        # ===== 尝试 2：tushare sina 源 fallback =====
        if not got:
            try:
                df = ts.realtime_quote(ts_code=etf_code, src='sina')
                if isinstance(df, pd.DataFrame) and not df.empty:
                    row = df.iloc[0]
                    df_out = pd.DataFrame([{
                        "etf_code": row.get("TS_CODE", etf_code),
                        "date": row.get("DATE"),
                        "close": float(row.get("PRICE", 0.0)),        # 已是元
                        "vol": float(row.get("VOLUME", 0.0)),         # 已是股/份
                        "amount": float(row.get("AMOUNT", 0.0))
                    }]).dropna(subset=["etf_code", "close"])
                    if not df_out.empty:
                        got = True
                        source = "sina"
            except Exception as e:
                logger.warning(f"ETF实时行情(sina)兜底失败 {etf_code}: {e}")

        if not got:
            logger.error("ETF实时行情获取失败（dc + sina 均无数据）：%s", etf_code)
            return pd.DataFrame(columns=["etf_code", "close", "vol", "amount"])

        # 写入缓存
        if use_cache or force_refresh:
            self._rt_cache[etf_code] = (df_out.copy(), self._now_ts())
            logger.info("ETF实时行情已刷新缓存（%s，来源：%s）@ %s",
                        etf_code, source, self._rt_cache[etf_code][1])

        return df_out

    # ---------- 其他查询 ----------
    @staticmethod
    def get_etf_info_for_beta_regression() -> pd.DataFrame:
        with get_db() as db:
            query = db.query(EtfInfo).filter(
                (EtfInfo.invest_type.in_(["被动指数型", "增强指数型"]))
            )
            df = pd.read_sql(query.statement, db.bind)
        return df

    @staticmethod
    def get_etf_hist_by_code(etf_code: str, start_date: str = None, end_date: str = None):
        with get_db() as db:
            query = db.query(EtfHist).filter(EtfHist.etf_code == etf_code)
            if start_date:
                query = query.filter(EtfHist.date >= start_date)
            if end_date:
                query = query.filter(EtfHist.date <= end_date)
            df = pd.read_sql(query.statement, db.bind)
        return df