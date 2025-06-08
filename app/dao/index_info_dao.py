from typing import List, Union
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, tuple_
from app.models.index_models import IndexInfo, IndexHist
from app.database import get_db
from app.utils.data_utils import process_in_batches
import logging
import pandas as pd
import datetime
import akshare as ak

logger = logging.getLogger(__name__)

class IndexInfoDao:

    _instance = None  # 用于保存单例对象

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def load_index_info(self, code_list: List = None) -> List[IndexInfo]:
        try:
            with get_db() as db:
                if code_list:
                    index_info_lst = db.query(IndexInfo).filter(IndexInfo.index_code.in_(code_list)).all()
                else:
                    index_info_lst = db.query(IndexInfo).all()
                return index_info_lst
        except Exception as e:
            logger.error(e)
        
    def batch_insert(self, index_info_lst: List[IndexInfo]):
        try:
            with get_db() as db:
                def insert_one_batch(batch: List[IndexInfo]):
                    db.add_all(batch)
                    db.commit()
                    return batch
                process_in_batches(index_info_lst, insert_one_batch)
                return index_info_lst
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e
    
    def select_dataframe_all(self) -> pd.DataFrame:
        try:
            with get_db() as db:
                # 构造查询条件
                query = db.query(IndexInfo)
                # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            return pd.DataFrame()
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(IndexInfo).delete()
                db.commit()
        except Exception as e:
            logger.error("Error deleting all IndexInfo: %s", e)
            db.rollback()
            raise e
                
class IndexHistDao:

    _instance = None  # 用于保存单例对象

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def get_latest_date(self, index_code: str):
        """
        查询指定指数在历史行情表中的最新日期，如果没有数据则返回 None。
        """
        try:
            with get_db() as db:
                logger.info(f"Using database bind: {db.bind}")
                result = db.query(IndexHist.date).filter(IndexHist.index_code == index_code).order_by(IndexHist.date.desc()).first()
                if result:
                    return result[0]
                else:
                    return None
        except Exception as e:
            logger.error("Error querying latest date for %s: %s", index_code, e)
            return None

    def batch_insert(self, records: List[IndexHist]):
        """
        批量插入历史数据记录到数据库。
        """
        
        try:
            with get_db() as db:
                logger.info(f"Using database bind: {db.bind}")
                def insert_one_batch(batch: List[IndexHist]):
                    db.add_all(batch)
                    db.commit()
                    return batch
                process_in_batches(records, insert_one_batch)
                return records
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e

    def batch_upsert(self, records: List[IndexHist]):
        """
        应用层批量 upsert。如果主键存在则更新，不存在则插入。
        分批执行，每批默认处理1000条。
        """
        try:
            with get_db() as db:
                logger.info(f"Using database bind: {db.bind}")

                def upsert_one_batch(batch: List[IndexHist]):
                    # 提取所有主键
                    key_set = {(rec.index_code, rec.date) for rec in batch}

                    # 查询已存在记录
                    existing_records = db.query(IndexHist).filter(
                        IndexHist.index_code.in_([k[0] for k in key_set]),
                        IndexHist.date.in_([k[1] for k in key_set])
                    ).all()

                    existing_dict = {(rec.index_code, rec.date): rec for rec in existing_records}

                    for rec in batch:
                        key = (rec.index_code, rec.date)
                        if key in existing_dict:
                            existing = existing_dict[key]
                            existing.open = rec.open
                            existing.close = rec.close
                            existing.high = rec.high
                            existing.low = rec.low
                            existing.volume = rec.volume
                            existing.amount = rec.amount
                            existing.change = rec.change
                            existing.change_percent = rec.change_percent
                        else:
                            db.add(rec)

                    db.commit()
                    return batch

                return process_in_batches(records, upsert_one_batch)

        except Exception as e:
            logger.error("Error during index_hist batch upsert: %s", e)
            db.rollback()
            raise e
    def select_dataframe_all(self):
        try:
            with get_db() as db:
                # 构造查询条件
                query = db.query(IndexHist)
                # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            return pd.DataFrame()
        
    def select_dataframe_by_code(self, index_code: str):
        """
        查询指定指数的所有历史数据，并返回为 Pandas DataFrame。
        
        :param index_code: 指数代码，例如 "600012"
        :return: 包含查询结果的 DataFrame，如果没有数据则返回空 DataFrame。
        """
        try:
            with get_db() as db:
                # 构造查询条件
                query = db.query(IndexHist).filter(IndexHist.index_code == index_code).order_by(IndexHist.date)
                # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def select_after_date_as_dataframe(self, index_code: str, date: Union[str, datetime.date]):
        """
        查询指定指数代码在当前日期之后的数据，并返回一个包含查询结果的 DataFrame。
        :param index_code: 股票代码
        :return: 包含查询结果的 DataFrame，如果没有数据则返回空 DataFrame。
        """
        try:
            with get_db() as db:
                # 构造查询条件，查找 index_code 相等且 date 大于指定日期的记录
                query = db.query(IndexHist).filter(
                    IndexHist.index_code == index_code,
                    IndexHist.date >= date
                )
                # 使用 pd.read_sql 将查询结果转换为 DataFrame，
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            logger.error("Error querying after date for %s: %s", index_code, e)
            return pd.DataFrame()
    
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(IndexHist).delete()
                db.commit()
        except Exception as e:
            logger.error("Error deleting allrecords", e)
            db.rollback()
            raise e


IndexInfoDao._instance = object.__new__(IndexInfoDao)
IndexInfoDao._instance.__init__()
IndexHistDao._instance = object.__new__(IndexHistDao)
IndexHistDao._instance.__init__()
