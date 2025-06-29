# app/data_fetcher/csi_index_data_fetcher.py

import pandas as pd
import akshare as ak
import datetime
from app.data_fetcher.base_fetcher import BaseFetcher
from app.models.index_models import IndexInfo, IndexHist
from app.database import get_db

CSI_INDEX_CODES = [
    "H11001",  # 中证综合债
    "H11004",  # 中证10债
    # 可扩展更多
]


class CSIIndexDataFetcher:
    """
    支持从 AkShare 获取中证指数历史数据，并从数据库查询本地存储的历史数据。
    支持根据 index_code -> DataFrame 的形式传入补充数据（如盘中估算值）。
    """

    def __init__(self, additional_map: dict[str, pd.DataFrame] = None):
        """
        :param additional_map: 可选补充数据映射，如 { "H11001.CSI": df }
        """
        self.additional_map = additional_map or {}

    @staticmethod
    def fetch_csi_index_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = ak.stock_zh_index_hist_csindex(
            symbol=symbol, start_date=start_date, end_date=end_date
        )
        df = df.rename(columns={
            "日期": "date",
            "指数代码": "index_code",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交金额": "amount",
            "涨跌幅": "change_percent",
            "涨跌": "change",
            "指数中文简称": "index_name"
        })

        df_info = df[["index_code", "index_name"]].copy()
        df_info["market"] = "CSI"
        df_info["index_code"] = df_info["index_code"].astype(str) + ".CSI"
        df_info = df_info.drop_duplicates()

        df_hist = df[["index_code", "date", "open", "close", "high", "low", "volume", "amount", "change_percent", "change"]].copy()
        df_hist = BaseFetcher.standardize_dates(df_hist, "date", "%Y-%m-%d")
        df_hist["volume"] = df_hist["volume"] * 10000     # 万手 → 股
        df_hist["amount"] = df_hist["amount"] * 1e8       # 亿元 → 元
        df_hist["index_code"] = df_hist["index_code"].astype(str) + ".CSI"
        df_hist = df_hist.sort_values("date")
        return df_info, df_hist

    @staticmethod
    def fetch_and_store_csi_index_data(start_date: str, end_date: str) -> None:
        for code in CSI_INDEX_CODES:
            df_info, df_hist = CSIIndexDataFetcher.fetch_csi_index_data(code, start_date, end_date)
            with get_db() as db:
                BaseFetcher.write_to_db_no_date(df_info, IndexInfo, db)
                BaseFetcher.write_to_db(df_hist, IndexHist, db, drop_na_row=False)

    def get_data_by_code_and_date(self, code: str = None, start: str = None, end: str = None) -> pd.DataFrame:
        """
        从数据库中读取指定指数代码和日期范围的数据，并合并该 code 对应的补充数据。

        :param code: 指数代码，如 "H11001.CSI"。若为 None，则不过滤。
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
                df = df.reset_index(drop=True)
                

        return df.sort_values("date").dropna(how="all")
