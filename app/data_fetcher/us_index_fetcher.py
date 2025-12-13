# app/data_fetcher/us_index_fetcher.py

import datetime
import logging
from typing import Optional

import pandas as pd
import tushare as ts
from sqlalchemy import text

from app.data_fetcher.base_fetcher import BaseFetcher
from app.database import get_db

logger = logging.getLogger(__name__)

tspro = ts.pro_api()

US_INDEX_CODE = "USDOLLAR.FXCM"


class USIndexFetcher(BaseFetcher):
    """
    从 Tushare 拉取美元指数 USDOLLAR.FXCM（FXCM 美元指数），
    写入本地表 us_index，并提供简单的增量同步逻辑。

    约定：
    - Tushare 接口：tspro.fx_daily(ts_code="USDOLLAR.FXCM", ...)
    - us_index 表结构与 fx_daily 返回字段兼容，至少包含：
        date, ts_code, trade_date, bid_open, bid_close, bid_high, bid_low,
        ask_open, ask_close, ask_high, ask_low, tick_qty
    """

    @staticmethod
    def fetch_us_index_data(start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Tushare 拉取美元指数原始数据，并做基础预处理。

        参数
        ----
        start_date : str
            起始日期（YYYYMMDD），直接传给 tspro.fx_daily。
        end_date : str
            结束日期（YYYYMMDD），直接传给 tspro.fx_daily。

        返回
        ----
        DataFrame
            已标准化 date 列且按日期排序的美元指数数据。
        """
        all_chunks = []
        offset = 0
        limit = 2000

        fields = (
            "ts_code,trade_date,"
            "bid_open,bid_close,bid_high,bid_low,"
            "ask_open,ask_close,ask_high,ask_low,"
            "tick_qty"
        )

        while True:
            df = tspro.fx_daily(
                ts_code=US_INDEX_CODE,
                start_date=start_date,
                end_date=end_date,
                fields=fields,
                offset=offset,
                limit=limit,
            )

            if df is None or df.empty:
                break

            all_chunks.append(df)
            offset += limit

        if not all_chunks:
            logger.warning(
                "USIndexFetcher.fetch_us_index_data: no data from Tushare for "
                "USDOLLAR.FXCM between %s and %s",
                start_date,
                end_date,
            )
            return pd.DataFrame()

        df = pd.concat(all_chunks, ignore_index=True)

        # 标准化日期：trade_date -> date（datetime.date）
        df = df.rename(columns={"trade_date": "date"})
        df = BaseFetcher.standardize_dates(df, "date", "%Y%m%d")

        # 去重 + 排序
        df = df.dropna(subset=["date"])
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.sort_values("date")

        return df

    @staticmethod
    def fetch_and_store_us_index_data(start_date: str, end_date: str) -> None:
        """
        拉取 [start_date, end_date] 区间内的美元指数数据，并写入 us_index 表。

        - 会先查询 us_index 中该日期区间已有的 date，做简单去重，
          避免重复插入相同交易日。
        - 若表尚未创建，df.to_sql 会抛错，由调用方根据需要创建表结构。
        """
        df = USIndexFetcher.fetch_us_index_data(start_date, end_date)
        if df.empty:
            logger.info(
                "USIndexFetcher.fetch_and_store_us_index_data: "
                "no data to insert for %s ~ %s",
                start_date,
                end_date,
            )
            return

        with get_db() as db:
            engine = db.bind

            min_date = df["date"].min()
            max_date = df["date"].max()

            # 读取已存在的日期，避免重复插入
            existing_dates = pd.DataFrame()
            try:
                sql = text(
                    "SELECT date FROM us_index "
                    "WHERE date >= :start_date AND date <= :end_date"
                )
                existing_dates = pd.read_sql(
                    sql, engine, params={"start_date": min_date, "end_date": max_date}
                )
            except Exception as e:
                logger.warning(
                    "USIndexFetcher: failed to load existing dates from us_index, "
                    "will try raw append. error=%s",
                    e,
                )

            if not existing_dates.empty and "date" in existing_dates.columns:
                existed = pd.to_datetime(existing_dates["date"]).dt.date
                df = df[~df["date"].isin(existed)]

            if df.empty:
                logger.info(
                    "USIndexFetcher.fetch_and_store_us_index_data: "
                    "no new rows to insert for %s ~ %s",
                    start_date,
                    end_date,
                )
                return

            # 写入数据库：依赖 us_index 表结构已创建好
            df.to_sql("us_index", engine, if_exists="append", index=False)
            logger.info(
                "USIndexFetcher: inserted %d rows into us_index for %s ~ %s",
                len(df),
                start_date,
                end_date,
            )
