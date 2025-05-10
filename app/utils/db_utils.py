import pandas as pd
from sqlalchemy.orm import Query
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.sql import ClauseElement
from typing import List, Optional, Type
from app.database import get_db
import logging

logger = logging.getLogger(__name__)

def chunked_query_to_dataframe(
    base_query: Query,
    table_name: str,
    chunksize: int = 10000
) -> pd.DataFrame:
    """
    支持联表（join）的通用分块读取工具函数，带进度日志。

    :param base_query: 已配置完 join、filter、order_by 的 SQLAlchemy Query 对象
    :param table_name: 日志中展示的表名
    :param chunksize: 每块读取多少条记录
    :return: 拼接后的 Pandas DataFrame
    """
    with get_db() as db:
        try:
            logger.info("开始分块读取 [%s]，每块 %d 条...", table_name, chunksize)
            chunks = pd.read_sql(base_query.statement, db.bind, chunksize=chunksize)
            df_list = []
            chunk_index = 0
            total_rows = 0
            for chunk in chunks:
                chunk_index += 1
                row_count = len(chunk)
                total_rows += row_count
                logger.info("读取第 %d 块，记录数：%d，累计：%d", chunk_index, row_count, total_rows)
                df_list.append(chunk)

            logger.info("读取完成，总记录数：%d，共 %d 块", total_rows, chunk_index)
            return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
        except Exception as e:
            logger.error("Error during chunked join query for [%s]: %s", table_name, e)
            return pd.DataFrame()