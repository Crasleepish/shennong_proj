from typing import List, Union
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, tuple_
from app.models.fund_models import FundInfo, FundHist
from app.database import get_db
from app.utils.data_utils import process_in_batches
import logging
import pandas as pd
import datetime
import akshare as ak

logger = logging.getLogger(__name__)

class FundInfoDao:

    _instance = None  # 用于保存单例对象

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def load_fund_info(self) -> List[FundInfo]:
        try:
            with get_db() as db:
                fund_info_lst = db.query(FundInfo).all()
                return fund_info_lst
        except Exception as e:
            logger.error(e)
        
    def batch_insert(self, fund_info_lst: List[FundInfo]):
        try:
            with get_db() as db:
                db.add_all(fund_info_lst)
                db.commit()
                return fund_info_lst
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e
    
    def select_dataframe_all(self) -> pd.DataFrame:
        try:
            with get_db() as db:
                # 构造查询条件
                query = db.query(FundInfo)
                # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            return pd.DataFrame()
        
    def select_dataframe_by_code(self, fund_code: str) -> pd.DataFrame:
        try:
            with get_db() as db:
                # 构造查询条件
                query = db.query(FundInfo).filter(FundInfo.fund_code == fund_code)
                # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            return pd.DataFrame()
        
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(FundInfo).delete()
                db.commit()
        except Exception as e:
            logger.error("Error deleting all FundInfo: %s", e)
            db.rollback()
            raise e
                
class FundHistDao:

    _instance = None  # 用于保存单例对象

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def get_latest_date(self, fund_code: str):
        """
        查询指定指数在历史行情表中的最新日期，如果没有数据则返回 None。
        """
        try:
            with get_db() as db:
                logger.info(f"Using database bind: {db.bind}")
                result = db.query(FundHist.date).filter(FundHist.fund_code == fund_code).order_by(FundHist.date.desc()).first()
                if result:
                    return result[0]
                else:
                    return None
        except Exception as e:
            logger.error("Error querying latest date for %s: %s", fund_code, e)
            return None

    def batch_insert(self, records: List[FundHist]):
        """
        批量插入历史数据记录到数据库。
        """
        
        try:
            with get_db() as db:
                logger.info(f"Using database bind: {db.bind}")
                def insert_one_batch(batch: List[FundHist]):
                    db.add_all(batch)
                    db.commit()
                    return batch
                process_in_batches(records, insert_one_batch)
                return records
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e
        
    def select_dataframe_all(self):
        try:
            with get_db() as db:
                # 构造查询条件
                query = db.query(FundHist)
                # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            return pd.DataFrame()
        
    def select_dataframe_by_code(self, fund_code: str):
        """
        查询指定指数的所有历史数据，并返回为 Pandas DataFrame。
        
        :param fund_code: 指数代码，例如 "600012"
        :return: 包含查询结果的 DataFrame，如果没有数据则返回空 DataFrame。
        """
        try:
            with get_db() as db:
                # 构造查询条件
                query = db.query(FundHist).filter(FundHist.fund_code == fund_code).order_by(FundHist.date)
                # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def select_after_date_as_dataframe(self, fund_code: str, date: Union[str, datetime.date]):
        """
        查询指定指数代码在当前日期之后的数据，并返回一个包含查询结果的 DataFrame。
        :param fund_code: 股票代码
        :return: 包含查询结果的 DataFrame，如果没有数据则返回空 DataFrame。
        """
        try:
            with get_db() as db:
                # 构造查询条件，查找 fund_code 相等且 date 大于指定日期的记录
                query = db.query(FundHist).filter(
                    FundHist.fund_code == fund_code,
                    FundHist.date >= date
                )
                # 使用 pd.read_sql 将查询结果转换为 DataFrame，
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            logger.error("Error querying after date for %s: %s", fund_code, e)
            return pd.DataFrame()
    
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(FundHist).delete()
                db.commit()
        except Exception as e:
            logger.error("Error deleting allrecords", e)
            db.rollback()
            raise e


FundInfoDao._instance = object.__new__(FundInfoDao)
FundInfoDao._instance.__init__()
FundHistDao._instance = object.__new__(FundHistDao)
FundHistDao._instance.__init__()
