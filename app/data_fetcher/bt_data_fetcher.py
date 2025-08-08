from abc import ABC, abstractmethod
import pandas as pd
import os
import pickle
from typing import List, Dict, Optional
import logging
from sqlalchemy.orm import Session
from datetime import timedelta
from app.database import get_db
from app.models.stock_models import StockHistUnadj
from app.models.stock_models import AdjFactor
from app.models.stock_models import FundamentalData
from app.dao.stock_info_dao import StockInfoDao

logger = logging.getLogger(__name__)


class DataFetcher:

    @staticmethod
    def get_stock_info_df() -> pd.DataFrame:
        """
        返回股票基本信息数据，包括股票代码、上市日期、所属市场等。
        
        DataFrame 格式要求：
        - 索引为股票代码（字符串）
        - 至少包含一列 listing_date（datetime64[ns]），以及市场信息，例如 exchange, listing_date, list_status
        
        样例输出：
                    listing_date     exchange
        stock_code                        
        600012       1990-05-01      SZSE
        600016       1988-06-15      SSE
        600018       2000-09-30      SSE
        ...
        """
        df = StockInfoDao.select_dataframe_all()
        df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
        df = df.set_index("stock_code", drop=True)
        df = df[["exchange", "listing_date", "list_status", "industry"]]
        return df

    def fetch_mkt_cap_on(self, stock_codes: List[str], date: str) -> pd.Series:
        try:
            with get_db() as db:
                query = db.query(StockHistUnadj.stock_code, StockHistUnadj.mkt_cap)
                query = query.filter(StockHistUnadj.date == date)
                if stock_codes:
                    query = query.filter(StockHistUnadj.stock_code.in_(stock_codes))
                df = pd.read_sql(query.statement, db.bind)
        except Exception as e:
            logger.error(f"Failed to fetch mkt_cap on {date}: {e}")
            return pd.Series(dtype='float32')

        if df.empty:
            return pd.Series(dtype='float32')

        df = df.dropna(subset=['mkt_cap'])
        return df.set_index('stock_code')['mkt_cap'].astype('float32')

    def fetch_price(self, field: str, start_date: str, end_date: str) -> pd.DataFrame:
        current = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        result = []

        while current <= end:
            chunk_end = min(current + pd.DateOffset(days=30), end)
            try:
                with get_db() as db:
                    query = db.query(StockHistUnadj.date, StockHistUnadj.stock_code, getattr(StockHistUnadj, field))
                    query = query.filter(StockHistUnadj.date >= current.strftime('%Y-%m-%d'), StockHistUnadj.date <= chunk_end.strftime('%Y-%m-%d'))
                    df = pd.read_sql(query.statement, db.bind)
                    if not df.empty:
                        df = df.dropna()
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.pivot(index='date', columns='stock_code', values=field)
                        result.append(df.astype('float32'))
            except Exception as e:
                logger.error(f"Failed to fetch price field {field} from {current} to {chunk_end}: {e}")
            current = chunk_end + timedelta(days=1)

        if result:
            return pd.concat(result).sort_index()
        return pd.DataFrame()

    def fetch_adjfactor(self, start_date: str, end_date: str) -> pd.DataFrame:
        current = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        result = []

        while current <= end:
            chunk_end = min(current + pd.DateOffset(days=30), end)
            try:
                with get_db() as db:
                    query = db.query(AdjFactor.date, AdjFactor.stock_code, AdjFactor.adj_factor)
                    query = query.filter(AdjFactor.date >= current.strftime('%Y-%m-%d'), AdjFactor.date <= chunk_end.strftime('%Y-%m-%d'))
                    df = pd.read_sql(query.statement, db.bind)
                    if not df.empty:
                        df = df.dropna()
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.pivot(index='date', columns='stock_code', values='adj_factor')
                        result.append(df.astype('float32'))
            except Exception as e:
                logger.error(f"Failed to fetch adj_factor from {current} to {chunk_end}: {e}")
            current = chunk_end + timedelta(days=1)

        if result:
            return pd.concat(result).sort_index().ffill()
        return pd.DataFrame()

    def fetch_adj_hist(self, field: str, start_date: str, end_date: str) -> pd.DataFrame:
        pivot_df_hist = self.fetch_price(field, start_date, end_date)
        pivot_adjf = self.fetch_adjfactor(start_date, end_date)

        # 对齐列
        if set(pivot_df_hist.columns) != set(pivot_adjf.columns):
            logger.warning("Columns of price and adj_factor mismatch, intersecting them.")
        common_stocks = pivot_df_hist.columns.intersection(pivot_adjf.columns)
        pivot_df_hist = pivot_df_hist[common_stocks]
        pivot_adjf = pivot_adjf[common_stocks]

        # 对齐索引
        pivot_adjf = pivot_adjf.reindex(pivot_df_hist.index)
        pivot_adjf = pivot_adjf.ffill()

        # 最新复权因子
        latest_adj_factors = pivot_adjf.iloc[-1]
        adj_ratios = pivot_adjf.div(latest_adj_factors, axis='columns')

        adjusted_prices = pivot_df_hist * adj_ratios
        adjusted_prices.index = pd.to_datetime(adjusted_prices.index)
        return adjusted_prices.astype('float32')

    def fetch_fundamentals_on(self, field: str, start_date: str, end_date: str) -> pd.DataFrame:
        adj_start = pd.to_datetime(start_date) - pd.Timedelta(days=120)
        adj_end = pd.to_datetime(end_date) - pd.Timedelta(days=120)
        try:
            with get_db() as db:
                query = db.query(FundamentalData.report_date, FundamentalData.stock_code, getattr(FundamentalData, field))
                query = query.filter(FundamentalData.report_date >= adj_start, FundamentalData.report_date <= adj_end)
                df = pd.read_sql(query.statement, db.bind)
        except Exception as e:
            logger.error(f"Failed to fetch fundamental field {field} from {start_date} to {end_date}: {e}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df = df.dropna()
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.pivot(index='report_date', columns='stock_code', values=field)
        return df.astype('float32')
    
    def fetch_fundamentals_on_all(self, start_date: str, end_date: str, fields: List[str]) -> pd.DataFrame:
        adj_start = pd.to_datetime(start_date) - pd.Timedelta(days=500)
        adj_end = pd.to_datetime(end_date) - pd.Timedelta(days=120)
        try:
            with get_db() as db:
                query = db.query(FundamentalData.report_date, FundamentalData.stock_code, *[getattr(FundamentalData, f) for f in fields])
                query = query.filter(FundamentalData.report_date >= adj_start, FundamentalData.report_date <= adj_end)
                df = pd.read_sql(query.statement, db.bind)
        except Exception as e:
            logger.error(f"Failed to fetch selected fundamental fields from {start_date} to {end_date}: {e}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df = df.dropna(subset=['stock_code', 'report_date'])
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.set_index(['report_date', 'stock_code'])
        return df.astype('float32')
