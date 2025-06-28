# app/data_fetcher/gold_data_fetcher.py

import pandas as pd
import tushare as ts
import datetime
from app.data_fetcher.base_fetcher import BaseFetcher
from app.models.index_models import IndexInfo, IndexHist
from app.database import get_db

tspro = ts.pro_api()

GOLD_INDEX_CODES = [
    "Au99.95",  # 黄金9995
    "Au99.99",  # 黄金9999
    "iAu99.99"  # 国际板黄金9999
]


class GoldDataFetcher:
    """
    支持从 Tushare 获取黄金指数历史数据，并从数据库查询本地存储的历史数据。
    支持传入 index_code -> DataFrame 的映射用于补充盘中估算值。
    """

    def __init__(self, additional_map: dict[str, pd.DataFrame] = None):
        """
        :param additional_map: 可选的 index_code -> DataFrame 映射
        """
        self.additional_map = additional_map or {}

    @staticmethod
    def fetch_sge_index_data(symbol: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        all_chunks = []
        offset = 0
        limit = 2000

        while True:
            df = tspro.sge_daily(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date,close,open,high,low,price_avg,change,pct_change,vol,amount",
                offset=offset,
                limit=limit
            )
            if df.empty:
                break
            all_chunks.append(df)
            offset += limit

        if not all_chunks:
            return pd.DataFrame(), pd.DataFrame()

        df = pd.concat(all_chunks, ignore_index=True)

        df = df.rename(columns={
            "ts_code": "index_code",
            "trade_date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "vol": "volume",
            "amount": "amount",
            "change": "change",
            "pct_change": "change_percent"
        })

        df_info = pd.DataFrame({
            "index_code": [symbol + ".SGE"],
            "index_name": [symbol],
            "market": ["SGE"]
        })

        df["index_code"] = symbol + ".SGE"
        df_hist = df[[
            "index_code", "date", "open", "close", "high", "low", "volume",
            "amount", "change_percent", "change"
        ]].copy()

        df_hist = BaseFetcher.standardize_dates(df_hist, "date", "%Y%m%d")
        df_hist["change_percent"] = df_hist["change_percent"] / 100
        df_hist = df_hist.sort_values("date")
        return df_info, df_hist

    @staticmethod
    def fetch_and_store_sge_index_data(start_date: str, end_date: str) -> None:
        for symbol in GOLD_INDEX_CODES:
            df_info, df_hist = GoldDataFetcher.fetch_sge_index_data(symbol, start_date, end_date)
            with get_db() as db:
                BaseFetcher.write_to_db_no_date(df_info, IndexInfo, db)
                BaseFetcher.write_to_db(df_hist, IndexHist, db, drop_na_row=False)

    def get_data_by_code_and_date(self, code: str = None, start: str = None, end: str = None) -> pd.DataFrame:
        """
        从数据库中读取指定指数代码和日期范围的数据，并合并该 code 对应的补充数据。

        :param code: 指数代码，如 "Au99.99.SGE"。若为 None，则不过滤。
        :param start: 开始日期，字符串格式 "YYYY-MM-DD"。若为 None，则不过滤。
        :param end: 结束日期，字符串格式 "YYYY-MM-DD"。若为 None，则不过滤。
        :return: 满足条件的 DataFrame。
        """
        with get_db() as db:
            query = db.query(IndexHist)

            if code:
                query = query.filter(IndexHist.index_code == code)
            if start:
                start_date = datetime.datetime.strptime(start, "%Y-%m-%d").date()
                query = query.filter(IndexHist.date >= start_date)
            if end:
                end_date = datetime.datetime.strptime(end, "%Y-%m-%d").date()
                query = query.filter(IndexHist.date <= end_date)

            df = pd.read_sql(query.statement, db.bind)

        if code and code in self.additional_map:
            df_extra = self.additional_map[code]
            if not df_extra.empty:
                df = pd.concat([df, df_extra], axis=0)

        return df.sort_values("date").dropna(how="all")
