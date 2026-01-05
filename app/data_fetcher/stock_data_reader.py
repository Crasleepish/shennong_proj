import threading
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import aliased
from app.database import get_db
from app.models.stock_models import StockInfo, StockHistUnadj, AdjFactor
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
import tushare as ts
import akshare as ak
from datetime import datetime, timedelta
import logging
from sqlalchemy import select, desc, literal_column
from sqlalchemy.dialects.postgresql import dialect
from typing import Optional, List
from app.data_fetcher.xueqiu_quote import stock_zh_a_xq_list
from app.dao.stock_info_dao import AdjFactorDao
from api_key import TUSHARE_API_KEY

logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
ts.set_token(TUSHARE_API_KEY)

# ======== 实时行情缓存：默认 5 分钟 ========
REALTIME_CACHE_TTL_SECONDS = 300  # 5 minutes
TS_REALTIME_BATCH = 50            # tushare realtime_quote 每批最多50只


class StockDataReader:
    """股票数据读取器（单例 + 实时行情缓存 + 多层fallback）"""

    _instance = None
    _lock = threading.Lock()  # 用于线程安全的单例创建

    def __new__(cls, *args, **kwargs):
        """单例实现：线程安全"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # 双重检查锁
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, realtime_cache_ttl: int = REALTIME_CACHE_TTL_SECONDS):
        # 避免多次初始化（单例模式下 __init__ 可能被多次调用）
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # ---- 实时行情缓存 ----
        self._rt_cache_df: Optional[pd.DataFrame] = None
        self._rt_cache_time: Optional[pd.Timestamp] = None
        self._rt_cache_ttl = int(realtime_cache_ttl)
        # ---- 连接 ----
        self._ts_pro = ts.pro_api()

    # ---------- 内部工具 ----------
    @staticmethod
    def _now_ts() -> pd.Timestamp:
        return pd.Timestamp.now(tz=None)

    def _rt_cache_valid(self) -> bool:
        if self._rt_cache_df is None or self._rt_cache_time is None:
            return False
        age = (self._now_ts() - self._rt_cache_time).total_seconds()
        return age < self._rt_cache_ttl

    def clear_realtime_cache(self):
        """手动清空实时缓存（如盘后或调试场景）"""
        self._rt_cache_df = None
        self._rt_cache_time = None

    # ---------- 日期 & 复权 ----------
    def _get_current_trade_date(self, offset=0) -> pd.Timestamp:
        today = pd.Timestamp.today().normalize()
        trade_dates = TradeCalendarReader.get_trade_dates(end=today.strftime("%Y-%m-%d"))
        return trade_dates[-1 + offset]

    def _get_adj_factors(self, trade_date: pd.Timestamp) -> pd.DataFrame:
        adj_factor_dao = AdjFactorDao()
        trade_date_str = trade_date.strftime("%Y%m%d")
        return adj_factor_dao.get_adj_factor_dataframe(stock_code=None, start_date=trade_date_str, end_date=trade_date_str)

    # ---------- 最新收盘价（不缓存） ----------
    def fetch_latest_close_prices(self, exchange_filter=None, list_status_filter=None, latest_trade_date=None, adjust_trade_date=None) -> pd.DataFrame:
        """
        从本地DB读取【最近一个≤latest_trade_date的收盘价】并做前复权到 T 日。
        注意：本函数不做缓存（每次都查库），以保证口径一致且便于调试。
        """
        if latest_trade_date is None:
            latest_trade_date = self._get_current_trade_date()
        if adjust_trade_date is None:
            adjust_trade_date = latest_trade_date
        with get_db() as db:
            shu = aliased(StockHistUnadj)
            si = aliased(StockInfo)
            af = aliased(AdjFactor)

            latest_hist_subq = (
                select(
                    shu.date.label("date"),
                    shu.close.label("close"),
                    shu.total_shares.label("total_shares"),
                )
                .where(shu.stock_code == si.stock_code)
                .order_by(desc(shu.date))
                .limit(1)
                .correlate(si)
            )
            if latest_trade_date:
                latest_hist_subq = latest_hist_subq.where(shu.date <= latest_trade_date)
            latest_hist_subq = latest_hist_subq.lateral()

            adj_factor_subq = (
                select(af.adj_factor.label("adj_factor"))
                .where(
                    af.stock_code == si.stock_code,
                    af.date == literal_column("shu.date")
                )
                .limit(1)
                .lateral()
            )

            stmt = (
                select(
                    si.stock_code.label("stock_code"),
                    si.stock_name.label("stock_name"),
                    si.exchange.label("exchange"),
                    literal_column("shu.date").label("date"),
                    literal_column("shu.close").label("close"),
                    literal_column("shu.total_shares").label("total_shares"),
                    literal_column("adj.adj_factor").label("adj_factor_d"),
                )
                .select_from(si)
                .join(latest_hist_subq.alias("shu"), literal_column("true"))
                .join(adj_factor_subq.alias("adj"), literal_column("true"))
            )

            if exchange_filter:
                stmt = stmt.where(si.exchange.in_(exchange_filter))
            if list_status_filter:
                stmt = stmt.where(si.list_status.in_(list_status_filter))

            compiled = stmt.compile(dialect=dialect(), compile_kwargs={"literal_binds": True})
            sql_str = str(compiled)
            df = pd.read_sql(sql_str, db.bind)

        if df.empty:
            logger.warning("未能获取任何收盘价数据")
            return df

        # 前复权到 T 日
        adj_factor_T_df = self._get_adj_factors(adjust_trade_date)
        adj_T_map = dict(zip(adj_factor_T_df.stock_code, adj_factor_T_df.adj_factor))
        if adj_factor_T_df.empty:
            # 若缺少最新交易日的复权因子，则使用各个股票所查到的有收盘价的那一日对应的复权因子，即不需再复权
            logger.warning("未能获取最新交易日的复权因子，将使用各个股票所查到的收盘价，不再复权。如果某支股票最近几天刚好有分红，可能会低估收益率。")
            adj_T_map = dict(zip(df.stock_code, df.adj_factor_d))


        missing_codes = []

        def compute_adj_close(row):
            code = row["stock_code"]
            adj_d = row["adj_factor_d"]
            adj_t = adj_T_map.get(code)
            if pd.notna(adj_d) and adj_t and adj_t != 0:
                return row["close"] * adj_d / adj_t
            missing_codes.append(code)
            return row["close"]

        df["adj_close"] = df.apply(compute_adj_close, axis=1)
        if missing_codes:
            logger.warning("以下股票缺少复权因子，使用原始 close 值代替：%s", sorted(set(missing_codes)))

        return df[["stock_code", "stock_name", "exchange", "date", "adj_close", "total_shares"]] \
            .rename(columns={"adj_close": "close"}) \
            .dropna(subset=["stock_code", "close"])

    def fetch_latest_close_prices_from_cache(self, exchange_filter=None, list_status_filter=None, latest_trade_date=None, adjust_trade_date=None) -> pd.DataFrame:
        """
        兼容旧接口：现在不再缓存，直接调用 fetch_latest_close_prices。
        """
        logger.debug("fetch_latest_close_prices_from_cache 已废弃缓存逻辑，直接走实时查询。")
        return self.fetch_latest_close_prices(exchange_filter, list_status_filter, latest_trade_date, adjust_trade_date)

    # ---------- 实时行情价（带TTL缓存） ----------
    # ===== 统一规范化：不同来源 -> 标准列 =====
    @staticmethod
    def _normalize_from_ak(df_raw: pd.DataFrame, valid_codes_set: set, code_map: dict) -> pd.DataFrame:
        if df_raw.empty:
            return pd.DataFrame(columns=["stock_code", "close", "vol", "amount"])
        df = df_raw.rename(columns={"代码": "code", "最新价": "close", "成交量": "vol", "成交额": "amount"})
        # ak返回"600000"这类代码，需要映射到"600000.SH"等
        df["stock_code"] = df["code"].map(code_map)
        df = df.dropna(subset=["stock_code", "close"])
        # 仅保留本地StockInfo存在且list_status='L'的代码
        df = df[df["stock_code"].isin(valid_codes_set)]
        return df[["stock_code", "close", "vol", "amount"]]

    @staticmethod
    def _normalize_from_rt_k(df_raw: pd.DataFrame, valid_codes_set: set) -> pd.DataFrame:
        if df_raw is None or df_raw.empty:
            return pd.DataFrame(columns=["stock_code", "close", "vol", "amount"])
        # rt_k 已为 ts_code+close/vol/amount
        cols = {"ts_code": "stock_code"}
        pick = ["ts_code", "close", "vol", "amount"]
        df = df_raw.loc[:, [c for c in pick if c in df_raw.columns]].rename(columns=cols)
        df = df.dropna(subset=["stock_code", "close"])
        df = df[df["stock_code"].isin(valid_codes_set)]
        return df[["stock_code", "close", "vol", "amount"]]

    @staticmethod
    def _normalize_from_realtime_quote(df_raw_list: List[pd.DataFrame], valid_codes_set: set) -> pd.DataFrame:
        if not df_raw_list:
            return pd.DataFrame(columns=["stock_code", "close", "vol", "amount"])
        df = pd.concat(df_raw_list, ignore_index=True)
        cols = {"TS_CODE": "stock_code", "PRICE": "close", "VOLUME": "vol", "AMOUNT": "amount",
                "ts_code": "stock_code", "price": "close", "volume": "vol", "amount": "amount"}
        df = df.rename(columns=cols)
        if "stock_code" not in df or "close" not in df:
            return pd.DataFrame(columns=["stock_code", "close", "vol", "amount"])
        df = df.dropna(subset=["stock_code", "close"])
        df = df[df["stock_code"].isin(valid_codes_set)]
        return df[["stock_code", "close", "vol", "amount"]]
    
    @staticmethod
    def _normalize_from_ak_spot(df_raw: pd.DataFrame, valid_codes_set: set) -> pd.DataFrame:
        """
        规范化 ak.stock_zh_a_spot() 的输出为标准列：
        ["stock_code", "close", "vol", "amount"]
        代码形如 sh600000/sz000001/bj920000 → 600000.SH/000001.SZ/920000.BJ
        """
        if df_raw is None or df_raw.empty:
            return pd.DataFrame(columns=["stock_code", "close", "vol", "amount"])

        df = df_raw.rename(columns={
            "代码": "raw_code",
            "最新价": "close",
            "成交量": "vol",
            "成交额": "amount",
        })

        def to_std_code(raw: str):
            if not isinstance(raw, str) or len(raw) < 3:
                return None
            pfx = raw[:2].lower()
            digits = "".join(ch for ch in raw if ch.isdigit())
            if not digits:
                return None
            if pfx == "sh":
                suf = "SH"
            elif pfx == "sz":
                suf = "SZ"
            elif pfx == "bj":
                suf = "BJ"
            else:
                return None
            return f"{digits}.{suf}"

        df["stock_code"] = df["raw_code"].map(to_std_code)
        df = df.dropna(subset=["stock_code", "close"])
        df = df[df["stock_code"].isin(valid_codes_set)]
        return df[["stock_code", "close", "vol", "amount"]]
    
    @staticmethod
    def _normalize_from_xq_list(df_raw: pd.DataFrame, valid_codes_set: set) -> pd.DataFrame:
        """
        规范化雪球 stock_zh_a_xq_list() 输出为标准列：
        输入列：symbol(如 'SH600519'/'SZ000001'/'BJ920000'), price, volume, amount
        输出列：['stock_code','close','vol','amount']，并仅保留本地有效代码集合
        """
        if df_raw is None or df_raw.empty:
            return pd.DataFrame(columns=["stock_code", "close", "vol", "amount"])

        def to_std_code(sym: str):
            # 'SH600519' -> '600519.SH'；'SZ000001' -> '000001.SZ'；'BJ920000' -> '920000.BJ'
            if not isinstance(sym, str) or len(sym) < 3:
                return None
            pfx = sym[:2].upper()
            digits = sym[2:]
            if not digits.isdigit():
                return None
            if pfx == "SH":
                suf = "SH"
            elif pfx == "SZ":
                suf = "SZ"
            elif pfx == "BJ":
                suf = "BJ"
            else:
                return None
            return f"{digits}.{suf}"

        df = df_raw.rename(columns={
            "symbol": "sym",
            "price": "close",
            "volume": "vol",
            "amount": "amount",
        }).copy()

        df["stock_code"] = df["sym"].map(to_std_code)
        df = df.dropna(subset=["stock_code", "close"])
        df = df[df["stock_code"].isin(valid_codes_set)]
        return df[["stock_code", "close", "vol", "amount"]]

    # ===== 主函数：实时价 + 多级兜底 + TTL缓存 =====
    def fetch_realtime_prices(self, *, use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        if use_cache and not force_refresh and self._rt_cache_valid():
            logger.info("命中实时行情缓存（≤%ds）。", self._rt_cache_ttl)
            return self._rt_cache_df.copy()

        # 1) 先准备本地有效代码集合 & 两个映射
        with get_db() as db:
            # 仅保留“在市”的标的
            rows = db.query(StockInfo.stock_code, StockInfo.list_status).all()
            valid_codes = [c for c, s in rows if (c and s == 'L')]
            valid_codes_set = set(valid_codes)
            # ak -> 证券代码（无后缀）到 带后缀 的映射
            code_map = {}
            for c in valid_codes:
                if "." in c:
                    short = c.split(".")[0]
                    code_map[short] = c

        # ==== 尝试 1：AkShare ====
        # 因反爬收紧，不再使用ak.stock_zh_a_spot()
        raise_success = False

        # ==== 尝试 2：Tushare Pro rt_k（全市场） ====
        if not raise_success:
            try:
                rt_df = self._ts_pro.rt_k(ts_code='3*.SZ,6*.SH,0*.SZ,9*.BJ')
                df = self._normalize_from_rt_k(rt_df, valid_codes_set)
                if not df.empty:
                    source = "tushare_rt_k"
                    raise_success = True
                else:
                    raise_success = False
            except Exception as e:
                logger.warning("Tushare rt_k 兜底失败：%s", e)
                raise_success = False

        # ==== 尝试 3: AkShare stock_zh_a_spot() ====
        if not raise_success:
            try:
                spot_df = ak.stock_zh_a_spot()
                df = self._normalize_from_ak_spot(spot_df, valid_codes_set)
                if not df.empty:
                    source = "akshare_spot"
                    raise_success = True
                else:
                    raise_success = False
            except Exception as e:
                logger.warning("AkShare stock_zh_a_spot 兜底失败：%s", e)
                raise_success = False

        # ==== 尝试 4：Xueqiu stock_zh_a_xq_list() ====
        if not raise_success:
            try:
                # 可按需传参：timeout=10.0 等；默认 8s
                xq_df = stock_zh_a_xq_list()  # 或 stock_zh_a_xq_list(timeout=10.0)
                df = self._normalize_from_xq_list(xq_df, valid_codes_set)
                if not df.empty:
                    source = "xueqiu_list"
                    raise_success = True
                else:
                    raise_success = False
            except Exception as e:
                logger.warning("Xueqiu stock_zh_a_xq_list 兜底失败：%s", e)
                raise_success = False

        # 如果都失败，给空表（并不写缓存）
        if not raise_success:
            logger.error("实时行情获取失败：已尝试 akshare -> rt_k -> realtime_quote。返回空结果。")
            return pd.DataFrame(columns=["stock_code", "close", "vol", "amount"])

        # 去重、清洗
        df = df.drop_duplicates(subset=["stock_code"], keep="first")
        df = df.dropna(subset=["stock_code", "close"])

        # 写缓存
        if use_cache or force_refresh:
            self._rt_cache_df = df.copy()
            self._rt_cache_time = self._now_ts()
            logger.info("实时行情已刷新缓存（来源：%s），时间：%s", source, self._rt_cache_time)

        return df