import pandas as pd
import tushare as ts
from datetime import datetime
from app.models.etf_model import EtfInfo, EtfHist
from app.database import get_db
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
import logging

logger = logging.getLogger(__name__)

class EtfDataReader:
    def __init__(self):
        self._last_df_cache = None
        self._last_cache_trade_date = None
        self._last_cache_etf_code = None

    def _get_current_trade_date(self) -> pd.Timestamp:
        today = pd.Timestamp.today().normalize()
        trade_dates = TradeCalendarReader.get_trade_dates(end=today.strftime("%Y-%m-%d"))
        return trade_dates[-1] if today not in trade_dates else today

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

            row = db.query(EtfHist).filter(EtfHist.etf_code == etf_code, EtfHist.date == latest_date).first()
            return pd.DataFrame([{
                "etf_code": row.etf_code,
                "date": row.date,
                "close": row.close,
                "vol": row.volume,
                "amount": row.amount
            }])

    def fetch_latest_close_prices_from_cache(self, etf_code: str, latest_trade_date=None) -> pd.DataFrame:
        if latest_trade_date is None:
            latest_trade_date = self._get_current_trade_date()
        if (
            self._last_df_cache is None or
            self._last_cache_trade_date != latest_trade_date or
            self._last_cache_etf_code != etf_code
        ):
            logger.info("缓存未命中，重新加载最新ETF行情")
            df = self.fetch_latest_close_prices(etf_code, latest_trade_date)
            self._last_df_cache = df
            self._last_cache_trade_date = latest_trade_date
            self._last_cache_etf_code = etf_code
        return self._last_df_cache

    def fetch_realtime_prices(self, etf_code: str) -> pd.DataFrame:
        """
        使用 tushare 实时行情接口（dc 源）获取单个 ETF 的最新成交价、成交量与成交额
        """
        try:
            df = ts.realtime_quote(ts_code=etf_code, src='dc')
        except Exception as e:
            logger.warning(f"实时ETF行情获取失败: {e}")
            return pd.DataFrame(columns=["etf_code", "close", "vol", "amount"])

        if df.empty:
            return pd.DataFrame(columns=["etf_code", "close", "vol", "amount"])

        row = df.iloc[0]
        return pd.DataFrame([{
            "etf_code": row["TS_CODE"],
            "date": row["DATE"],
            "close": row["PRICE"] / 10.0,     # 最新价格
            "vol": row["VOLUME"] * 100.0,      # 成交量（单位：份）
            "amount": row["AMOUNT"]    # 成交额（单位：元）
        }])
    
    
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