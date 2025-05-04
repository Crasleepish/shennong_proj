import pandas as pd
from sqlalchemy.orm import Query
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.sql import ClauseElement
from typing import List, Optional, Type
from app.database import get_db
import logging

logger = logging.getLogger(__name__)

def chunked_query_to_dataframe(
    model: Type[DeclarativeMeta],
    filters: Optional[List[ClauseElement]] = None,
    order_by: Optional[ClauseElement] = None,
    columns: Optional[List[str]] = None,
    chunksize: int = 10000
) -> pd.DataFrame:
    """
    通用的按条件分块读取工具函数，带进度日志。
    
    :param model: ORM 模型类
    :param filters: 查询条件列表
    :param order_by: 排序字段
    :param columns: 要选择的列名列表（默认为所有列）
    :param chunksize: 每块读取多少条记录
    :return: 拼接后的 Pandas DataFrame
    """
    with get_db() as db:
        if columns is not None:
            try:
                query: Query = db.query(*[getattr(model, col) for col in columns])
            except AttributeError as e:
                logger.error("Invalid column specified: %s", e)
                return pd.DataFrame()
        else:
            query: Query = db.query(model)

        if filters is not None:
            for cond in filters:
                query = query.filter(cond)
        if order_by is not None:
            query = query.order_by(order_by)

        try:
            logger.info("开始分块读取 [%s]，每块 %d 条...", model.__tablename__, chunksize)
            chunks = pd.read_sql(query.statement, db.bind, chunksize=chunksize)
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
            logger.error("Error during chunked query for %s: %s", model.__tablename__, e)
            return pd.DataFrame()
        