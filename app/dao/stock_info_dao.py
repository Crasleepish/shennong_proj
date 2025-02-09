from typing import List
from sqlalchemy.orm import Session
from app.models.stock_models import StockInfo, StockHistUnadj, UpdateFlag, CompanyAction
from app.database import get_db
from app.utils.data_utils import process_in_batches
import logging

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
        

StockInfoDao._instance = object.__new__(StockInfoDao)
StockInfoDao._instance.__init__()
UpdateFlagDao._instance = object.__new__(UpdateFlagDao)
UpdateFlagDao._instance.__init__()
StockHistUnadjDao._instance = object.__new__(StockHistUnadjDao)
StockHistUnadjDao._instance.__init__()
CompanyActionDao._instance = object.__new__(CompanyActionDao)
CompanyActionDao._instance.__init__()