import pandas as pd
import akshare as ak
from datetime import datetime
from app.database import get_db
from app.models.index_models import IndexHist
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
import logging

logger = logging.getLogger(__name__)

class IndexDataReader:
    def __init__(self):
        self._last_df_cache = None
        self._last_cache_trade_date = None
        self._last_cache_index_code = None

    def _get_current_trade_date(self) -> pd.Timestamp:
        today = pd.Timestamp.today().normalize()
        trade_dates = TradeCalendarReader.get_trade_dates(end=today.strftime("%Y-%m-%d"))
        return trade_dates[-1] if today not in trade_dates else today

    def fetch_latest_close_prices(self, index_code) -> pd.DataFrame:
        with get_db() as db:
            latest_date = db.query(IndexHist.date).filter(IndexHist.index_code == index_code).order_by(IndexHist.date.desc()).limit(1).scalar()
            if not latest_date:
                logger.warning(f"未获取到指数 {index_code} 的历史数据")
                return pd.DataFrame(columns=["stock_code", "close", "vol", "amount"])

            row = db.query(IndexHist).filter(IndexHist.index_code == index_code, IndexHist.date == latest_date).first()
            return pd.DataFrame([{
                "stock_code": row.index_code,
                "close": row.close,
                "vol": row.volume,
                "amount": row.amount
            }])

    def fetch_latest_close_prices_from_cache(self, index_code) -> pd.DataFrame:
        trade_date = self._get_current_trade_date()
        if (
            self._last_df_cache is None
            or self._last_cache_trade_date != trade_date
            or self._last_cache_index_code != index_code
        ):
            logger.info("缓存未命中，重新加载最新指数行情")
            df = self.fetch_latest_close_prices(index_code)
            self._last_df_cache = df
            self._last_cache_trade_date = trade_date
            self._last_cache_index_code = index_code
        return self._last_df_cache

    def fetch_realtime_prices(self, index_code) -> pd.DataFrame:
        trade_date = self._get_current_trade_date()
        start_dt = trade_date.strftime("%Y-%m-%d 09:30:00")

        last_dot_index = index_code.rfind('.')
        symbol = index_code[:last_dot_index] if last_dot_index > 0 else index_code

        try:
            df = ak.index_zh_a_hist_min_em(symbol=symbol, period="1", start_date=start_dt)
        except Exception as e:
            logger.warning(f"实时指数行情获取失败: {e}")
            return pd.DataFrame(columns=["stock_code", "close", "vol", "amount"])

        if df.empty:
            return pd.DataFrame(columns=["stock_code", "close", "vol", "amount"])

        latest = df.iloc[-1]
        return pd.DataFrame([{
            "stock_code": index_code,
            "close": latest["收盘"],
            "vol": latest["成交量"],
            "amount": latest["成交额"]
        }])
