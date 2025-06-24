import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import aliased
from app.database import get_db
from app.models.stock_models import StockInfo, StockHistUnadj, AdjFactor
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
import tushare as ts
import akshare as ak
from datetime import datetime
import logging
from sqlalchemy import select, desc, literal_column
from sqlalchemy.dialects.postgresql import dialect

logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
ts.set_token('2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211')

class StockDataReader:
    def __init__(self):
        self._last_df_cache = None
        self._last_cache_time = None
        self._last_cache_trade_date = None
        self._last_cache_filter_key = None
        self._ts_pro = ts.pro_api()

    def _get_current_trade_date(self) -> pd.Timestamp:
        today = pd.Timestamp.today().normalize()
        trade_dates = TradeCalendarReader.get_trade_dates(end=today.strftime("%Y-%m-%d"))
        return trade_dates[-1] if today not in trade_dates else today

    def _get_adj_factors(self, trade_date: pd.Timestamp) -> pd.DataFrame:
        trade_date_str = trade_date.strftime("%Y%m%d")
        return self._ts_pro.adj_factor(trade_date=trade_date_str)
    
    def fetch_latest_close_prices(self, exchange_filter=None, list_status_filter=None) -> pd.DataFrame:
        with get_db() as db:
            # 获取当前交易日 T 日
            trade_date = self._get_current_trade_date()

            shu = aliased(StockHistUnadj)
            si = aliased(StockInfo)
            af = aliased(AdjFactor)

            # 子查询 1：每只股票最新的行情（LIMIT 1 按日期倒序）
            latest_hist_subq = (
                select(
                    shu.date.label("date"),
                    shu.close.label("close"),
                    shu.total_shares.label("total_shares"),
                )
                .where(shu.stock_code == si.stock_code)
                .order_by(desc(shu.date))
                .limit(1)
                .lateral()
            )

            # 子查询 2：该日复权因子（stock_code + date）
            adj_factor_subq = (
                select(af.adj_factor.label("adj_factor"))
                .where(
                    af.stock_code == si.stock_code,
                    af.date == literal_column("shu.date")  # 关键：引用 LATERAL 子查询列
                )
                .limit(1)
                .lateral()
            )

            # 主查询
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

            # 过滤条件（可选）
            if exchange_filter:
                stmt = stmt.where(si.exchange.in_(exchange_filter))
            if list_status_filter:
                stmt = stmt.where(si.list_status.in_(list_status_filter))

            # PostgreSQL 需要字面量 SQL 来完全复制 psql 逻辑（可选）
            compiled = stmt.compile(dialect=dialect(), compile_kwargs={"literal_binds": True})
            sql_str = str(compiled)

            # 查询为 DataFrame
            df = pd.read_sql(sql_str, db.bind)

        if df.empty:
            logger.warning("未能获取任何收盘价数据")
            return df
        
        # 拉取 T 日复权因子
        adj_factor_T_df = self._get_adj_factors(trade_date)
        adj_T_map = dict(zip(adj_factor_T_df.ts_code, adj_factor_T_df.adj_factor))

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

        self._last_df_cache = df
        self._last_cache_time = pd.Timestamp.now()
        self._last_cache_trade_date = trade_date
        self._last_cache_filter_key = (tuple(exchange_filter or []), tuple(list_status_filter or []))

        return df[[
            "stock_code", "stock_name", "exchange", "date",
            "adj_close", "total_shares"
        ]].rename(columns={"adj_close": "close"}).dropna(subset=["stock_code", "close"])
    
    def fetch_latest_close_prices_from_cache(self, exchange_filter=None, list_status_filter=None) -> pd.DataFrame:
        trade_date = self._get_current_trade_date()
        current_key = (tuple(exchange_filter or []), tuple(list_status_filter or []))

        if (
            self._last_df_cache is None or
            self._last_cache_trade_date != trade_date or
            self._last_cache_filter_key != current_key
        ):
            logger.warning("缓存未命中，重新获取数据...")
            return self.fetch_latest_close_prices(exchange_filter, list_status_filter)

        return self._last_df_cache

    def fetch_realtime_prices(self) -> pd.DataFrame:
        df = ak.stock_zh_a_spot_em()
        if df.empty:
            logger.warning("未获取到任何实时数据")
            return pd.DataFrame(columns=["stock_code", "close", "vol", "amount"])

        # 获取带市场标识映射表
        with get_db() as db:
            rows = db.query(StockInfo.stock_code).all()
            code_map = {
                code.split(".")[0]: code for (code,) in rows if "." in code
            }

        df = df.rename(columns={
            "代码": "stock_code",
            "最新价": "close",
            "成交量": "vol",
            "成交额": "amount"
        })

        df["stock_code"] = df["stock_code"].map(code_map)
        df = df.dropna(subset=["stock_code", "close"])

        return df[["stock_code", "close", "vol", "amount"]]
