from typing import List, Union
from sqlalchemy.orm import Session, joinedload
from app.models.stock_models import StockInfo, StockHistUnadj, UpdateFlag, FutureTask, CompanyAction, StockHistAdj
from app.database import get_db
from app.utils.data_utils import process_in_batches
import logging
import pandas as pd
import datetime

logger = logging.getLogger(__name__)

class StockInfoDao:

    _instance = None  # 用于保存单例对象

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def load_stock_info(self) -> List[StockInfo]:
        try:
            with get_db() as db:
                stock_info_lst = db.query(StockInfo).all()
                return stock_info_lst
        except Exception as e:
            logger.error(e)
        
    def batch_insert(self, stock_info_lst: List[StockInfo]):
        try:
            with get_db() as db:
                db.add_all(stock_info_lst)
                db.commit()
                return stock_info_lst
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e
        

class UpdateFlagDao:

    _instance = None  # 用于保存单例对象

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def insert_one(self, obj: UpdateFlag):
        try:
            with get_db() as db:
                db.add(obj)
                db.commit()
                return obj
        except Exception as e:
            logger.error("Error during insert: %s", e)
            db.rollback()
            raise e
        
    def select_one_by_code(self, stock_code: str):
        try:
            with get_db() as db:
                result = db.query(UpdateFlag).filter(UpdateFlag.stock_code == stock_code).first()
                if result:
                    return {
                        'stock_code': result.stock_code,
                        'action_update_flag': result.action_update_flag,
                    }
        except Exception as e:
            logger.error("Error querying update flag for %s: %s", stock_code, e)
            return None
        
    def update_action_flag(self, stock_code: str, action_update_flag: str):
        try:
            with get_db() as db:
                lines = db.query(UpdateFlag).filter(UpdateFlag.stock_code == stock_code).update({UpdateFlag.action_update_flag: action_update_flag})
                db.commit()
                return lines
        except Exception as e:
            logger.error("Error updating action flag for %s:%s", stock_code, e)
            db.rollback()
            raise e


class FutureTaskDao:

    _instance = None

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def select_by_code_date(self, stock_code, date):
        """
        根据股票代码和日期查询FutureTask对象
        """
        try:
            with get_db() as db:
                result = db.query(FutureTask).options(joinedload(
                    FutureTask.task_id, FutureTask.task_type, FutureTask.stock_code, FutureTask.task_date)).filter(
                    FutureTask.stock_code == stock_code, 
                    FutureTask.task_date == date
                ).all()
                return result
        except Exception as e:
            logger.error("Error querying update flag for %s: %s", stock_code, e)
            return None
        
    def select_by_code_date_type(self, stock_code, date, task_type) -> List[FutureTask]:
        """
        根据股票代码、日期和任务类型查询FutureTask对象
        """
        try:
            with get_db() as db:
                result = db.query(FutureTask).filter(
                    FutureTask.stock_code == stock_code, 
                    FutureTask.task_date <= date,
                    FutureTask.task_type == task_type
                ).all()
                return result
        except Exception as e:
            logger.error("Error querying update flag for %s: %s", stock_code, e)
            return None
        
    def insert_one(self, obj: FutureTask):
        try:
            with get_db() as db:
                db.add(obj)
                db.commit()
                return obj
        except Exception as e:
            logger.error("Error during insert: %s", e)
            db.rollback()
            raise e
        
    def update_status_by_id(self, task_id: int, task_status: str):
        '''
        Update the task status by task_id
        '''
        try:
            with get_db() as db:
                lines = db.query(FutureTask).filter(FutureTask.task_id == task_id).update({FutureTask.task_status: task_status})
                db.commit()
                return lines
        except Exception as e:
            logger.error("Error updating task status for task %s:%s", task_id, e)
            db.rollback()
            raise e


class StockHistUnadjDao:

    _instance = None  # 用于保存单例对象

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def get_latest_date(self, stock_code: str):
        """
        查询指定股票在历史行情表中的最新日期，如果没有数据则返回 None。
        """
        try:
            with get_db() as db:
                result = db.query(StockHistUnadj.date).filter(StockHistUnadj.stock_code == stock_code).order_by(StockHistUnadj.date.desc()).first()
                if result:
                    return result[0]
                else:
                    return None
        except Exception as e:
            logger.error("Error querying latest date for %s: %s", stock_code, e)
            return None

    def batch_insert(self, records: List[StockHistUnadj]):
        """
        批量插入历史数据记录到数据库。
        """
        
        try:
            with get_db() as db:
                def insert_one_batch(batch: List[StockHistUnadj]):
                    db.add_all(batch)
                    db.commit()
                    return batch
                results = process_in_batches(records, insert_one_batch)
                return [item for sublist in results for item in sublist]
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e
        
    def select_all_as_dataframe(self, stock_code: str):
        """
        查询指定股票的所有前复权历史数据，并返回为 Pandas DataFrame。
        
        :param stock_code: 股票代码，例如 "600012"
        :return: 包含查询结果的 DataFrame，如果没有数据则返回空 DataFrame。
        """
        with get_db() as db:
            # 构造查询条件
            query = db.query(StockHistUnadj).filter(StockHistUnadj.stock_code == stock_code)
            # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
            df = pd.read_sql(query.statement, db.bind)
        return df
    
    def select_after_date_as_dataframe(self, stock_code: str, date: Union[str, datetime.date]):
        """
        查询指定股票代码在当前日期之后的公司行动数据，并返回一个包含查询结果的 DataFrame。
        :param stock_code: 股票代码
        :return: 包含查询结果的 DataFrame，如果没有数据则返回空 DataFrame。
        """
        with get_db() as db:
            # 构造查询条件，查找 stock_code 相等且 date 大于指定日期的记录
            query = db.query(StockHistUnadj).filter(
                StockHistUnadj.stock_code == stock_code,
                StockHistUnadj.date >= date
            )
            # 使用 pd.read_sql 将查询结果转换为 DataFrame，
            df = pd.read_sql(query.statement, db.bind)
        return df


class CompanyActionDao:

    _instance = None  # 用于保存单例对象

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def select_by_code_and_date(self, stock_code: str, ex_date) -> CompanyAction:
        try:
            with get_db() as db:
                record = db.query(CompanyAction).filter(
                    CompanyAction.stock_code == stock_code,
                    CompanyAction.ex_dividend_date == ex_date
                ).first()
                return record
        except Exception as e:
            logger.error("Error query for stock %s, date %s: %s", stock_code, ex_date, e)
            return None

    def batch_insert(self, records: List[CompanyAction]):
        """
        批量插入公司行动数据记录到数据库。
        """
        
        try:
            with get_db() as db:
                def insert_one_batch(batch: List[CompanyAction]):
                    db.add_all(batch)
                    db.commit()
                    return batch
                results = process_in_batches(records, insert_one_batch)
                return [item for sublist in results for item in sublist]
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e
        
    def select_all_as_dataframe(self, stock_code: str):
        """
        查询指定股票的所有公司行动数据，并返回为 Pandas DataFrame。
        
        :param stock_code: 股票代码，例如 "600012"
        :return: 包含查询结果的 DataFrame，如果没有数据则返回空 DataFrame。
        """
        with get_db() as db:
            # 构造查询条件
            query = db.query(CompanyAction).filter(CompanyAction.stock_code == stock_code)
            # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
            df = pd.read_sql(query.statement, db.bind)
        return df
        

class StockHistAdjDao:

    _instance = None  # 用于保存单例对象

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def get_latest_date(self, stock_code: str):
        """
        查询指定股票在历史行情表中的最新日期，如果没有数据则返回 None。
        """
        try:
            with get_db() as db:
                result = db.query(StockHistAdj.date).filter(StockHistAdj.stock_code == stock_code).order_by(StockHistAdj.date.desc()).first()
                if result:
                    return result[0]
                else:
                    return None
        except Exception as e:
            logger.error("Error querying latest date for %s: %s", stock_code, e)
            return None

    def batch_insert(self, records: List[StockHistAdj]):
        """
        批量插入历史数据记录到数据库。
        """
        
        try:
            with get_db() as db:
                def insert_one_batch(batch: List[StockHistAdj]):
                    db.add_all(batch)
                    db.commit()
                    return batch
                results = process_in_batches(records, insert_one_batch)
                return [item for sublist in results for item in sublist]
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e
    
    def select_all_as_dataframe(self, stock_code: str):
        """
        查询指定股票的所有无复权历史数据，并返回为 Pandas DataFrame。
        
        :param stock_code: 股票代码，例如 "600012"
        :return: 包含查询结果的 DataFrame，如果没有数据则返回空 DataFrame。
        """
        with get_db() as db:
            # 构造查询条件
            query = db.query(StockHistAdj).filter(StockHistAdj.stock_code == stock_code)
            # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
            df = pd.read_sql(query.statement, db.bind)
        return df

    def delete_by_stock_code(self, stock_code: str):
        try:
            with get_db() as db:
                db.query(StockHistAdj).filter(StockHistAdj.stock_code == stock_code).delete()
                db.commit()
        except Exception as e:
            logger.error("Error during delete: %s", e)
            db.rollback()
            raise e
        

StockInfoDao._instance = object.__new__(StockInfoDao)
StockInfoDao._instance.__init__()
UpdateFlagDao._instance = object.__new__(UpdateFlagDao)
UpdateFlagDao._instance.__init__()
FutureTaskDao._instance = object.__new__(FutureTaskDao)
FutureTaskDao._instance.__init__()
StockHistUnadjDao._instance = object.__new__(StockHistUnadjDao)
StockHistUnadjDao._instance.__init__()
CompanyActionDao._instance = object.__new__(CompanyActionDao)
CompanyActionDao._instance.__init__()
StockHistAdjDao._instance = object.__new__(StockHistAdjDao)
StockHistAdjDao._instance.__init__()