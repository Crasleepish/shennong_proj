# app/data_fetcher/us_index_data_reader.py

import datetime
import logging
from typing import Optional

import pandas as pd
from sqlalchemy import text

from app.database import get_db

logger = logging.getLogger(__name__)


class USIndexDataReader:
    """
    从数据库 us_index 表中读取美元指数时间序列。

    - 默认按 date 升序返回
    - 可选 start / end 过滤
    - 返回 DataFrame，index 为 date（datetime.date）
    """

    @staticmethod
    def read_us_index(
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        读取美元指数数据。

        参数
        ----
        start : str, optional
            起始日期，格式 "YYYY-MM-DD"；若为 None，则不过滤下界。
        end : str, optional
            结束日期，格式 "YYYY-MM-DD"；若为 None，则不过滤上界。

        返回
        ----
        DataFrame
            index 为 date（datetime.date），列包含：
            ts_code, bid_open, bid_close, bid_high, bid_low,
            ask_open, ask_close, ask_high, ask_low, tick_qty 等。
        """
        with get_db() as db:
            engine = db.bind

            conditions = []
            params = {}

            if start:
                try:
                    start_date = datetime.datetime.strptime(
                        start, "%Y-%m-%d"
                    ).date()
                except ValueError:
                    logger.warning(
                        "USIndexDataReader.read_us_index: invalid start=%s, "
                        "expected YYYY-MM-DD, ignored.",
                        start,
                    )
                else:
                    conditions.append("date >= :start_date")
                    params["start_date"] = start_date

            if end:
                try:
                    end_date = datetime.datetime.strptime(end, "%Y-%m-%d").date()
                except ValueError:
                    logger.warning(
                        "USIndexDataReader.read_us_index: invalid end=%s, "
                        "expected YYYY-MM-DD, ignored.",
                        end,
                    )
                else:
                    conditions.append("date <= :end_date")
                    params["end_date"] = end_date

            base_sql = "SELECT * FROM us_index"
            if conditions:
                base_sql += " WHERE " + " AND ".join(conditions)
            base_sql += " ORDER BY date"

            df = pd.read_sql(text(base_sql), engine, params=params)

        if df.empty:
            return df

        # 统一用 date 作为索引
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df.set_index("date")

        return df
