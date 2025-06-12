# app/data_fetcher/base_fetcher.py

import pandas as pd
from sqlalchemy.orm import Session
import logging
from typing import Dict, List, Type
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseFetcher:
    """
    通用数据预处理与入库工具类。
    """

    @staticmethod
    def standardize_dates(df: pd.DataFrame, date_col: str, format: str = None) -> pd.DataFrame:
        """
        将指定列转换为标准 datetime 类型。
        :param df: 原始 DataFrame
        :param date_col: 日期列名
        :param format: 可选格式，如 '%Y%m'
        """
        if format:
            df[date_col] = pd.to_datetime(df[date_col], format=format)
        else:
            df[date_col] = pd.to_datetime(df[date_col])
        return df

    @staticmethod
    def rename_columns(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
        return df.rename(columns=rename_map)

    @staticmethod
    def log_shape(df: pd.DataFrame, name: str):
        logger.info("获取 [%s] 共 %d 条", name, len(df))

    @staticmethod
    def write_to_db(df: pd.DataFrame, orm_class: Type, db: Session, drop_na_row: bool = True):
        # 仅过滤非法日期 NaT
        df = df[df["date"].notna()]

        count = 0
        for _, row in df.iterrows():
            # 若其他字段为空也一并跳过（可选）
            if drop_na_row and row.drop(labels=["date"]).isna().any():
                continue
            data_dict = row.to_dict()
            # 将 Pandas 的 NaN 转为 Python None，确保可映射到 SQL NULL
            clean_data = data_dict
            db.merge(orm_class(**clean_data))
            count += 1

        logger.info("写入数据库 [%s] 共 %d 条记录（允许部分字段为 NULL）", orm_class.__tablename__, count)

    @staticmethod
    def write_to_db_no_date(df: pd.DataFrame, orm_class: Type, db: Session):
        count = 0
        for _, row in df.iterrows():
            data_dict = row.to_dict()
            # 将 Pandas 的 NaN 转为 Python None，确保可映射到 SQL NULL
            clean_data = data_dict
            db.merge(orm_class(**clean_data))
            count += 1

        logger.info("写入数据库 [%s] 共 %d 条记录（允许部分字段为 NULL）", orm_class.__tablename__, count)