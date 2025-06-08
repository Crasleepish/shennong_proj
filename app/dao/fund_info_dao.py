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

    def load_index_fund_info(self) -> List[FundInfo]:
        try:
            with get_db() as db:
                condition = or_(
                    FundInfo.invest_type.in_(['被动指数型', '增强指数型']),
                    FundInfo.fund_type == '商品型'
                )
                fund_info_lst = db.query(FundInfo).filter(condition).all()
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
                condition = or_(
                    FundInfo.invest_type.in_(['被动指数型', '增强指数型']),
                    FundInfo.fund_type == '商品型'
                )
                query = db.query(FundInfo).filter(condition)
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
        if cls._instance is None:
            cls._instance = super(FundHistDao, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def get_latest_date(self, fund_code: str):
        try:
            with get_db() as db:
                result = db.query(FundHist.date).filter(FundHist.fund_code == fund_code).order_by(FundHist.date.desc()).first()
                return result[0] if result else None
        except Exception as e:
            logger.error("Error querying latest date for %s: %s", fund_code, e)
            return None

    def batch_upsert(self, records: List[FundHist]):
        try:
            with get_db() as db:
                def upsert_one_batch(batch: List[FundHist]):
                    existing = db.query(FundHist).filter(
                        FundHist.fund_code.in_([r.fund_code for r in batch]),
                        FundHist.date.in_([r.date for r in batch])
                    ).all()
                    existing_dict = {(r.fund_code, r.date): r for r in existing}
                    for r in batch:
                        key = (r.fund_code, r.date)
                        if key in existing_dict:
                            existing_rec = existing_dict[key]
                            existing_rec.value = r.value
                            existing_rec.net_value = r.net_value
                            existing_rec.change_percent = r.change_percent
                        else:
                            db.add(r)
                    db.commit()
                    return batch
                return process_in_batches(records, upsert_one_batch)
        except Exception as e:
            logger.error("Error during fund_hist batch upsert: %s", e)
            raise e

    def delete_all(self):
        try:
            with get_db() as db:
                db.query(FundHist).delete()
                db.commit()
        except Exception as e:
            logger.error("Error deleting all fund_hist records: %s", e)
            db.rollback()
            raise e
        
    def select_dataframe_by_code(self, fund_code: str) -> pd.DataFrame:
        """
        查询指定基金代码的所有历史行情数据，返回为 Pandas DataFrame。
        """
        try:
            with get_db() as db:
                query = db.query(FundHist).filter(FundHist.fund_code == fund_code).order_by(FundHist.date.asc())
                df = pd.read_sql(query.statement, db.bind)
                return df
        except Exception as e:
            logger.error("Error querying fund_hist for %s: %s", fund_code, e)
            return pd.DataFrame()
        
    def select_dataframe_by_code_and_date(self, fund_codes: List[str], date: datetime.date) -> pd.DataFrame:
        try:
            with get_db() as db:
                query = db.query(FundHist).filter(
                    FundHist.fund_code.in_(fund_codes),
                    FundHist.date == date
                )
                return pd.read_sql(query.statement, db.bind)
        except Exception as e:
            logger.error("Error querying fund_hist for previous date: %s", e)
            return pd.DataFrame()
        



FundInfoDao._instance = object.__new__(FundInfoDao)
FundInfoDao._instance.__init__()
FundHistDao._instance = object.__new__(FundHistDao)
FundHistDao._instance.__init__()
