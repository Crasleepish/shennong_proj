from typing import List, Union
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, tuple_
from app.models.stock_models import StockInfo, StockHistUnadj, UpdateFlag, FutureTask, CompanyAction, StockHistAdj, FundamentalData, SuspendData, StockShareChangeCNInfo
from app.database import get_db
from app.utils.data_utils import process_in_batches
import logging
import pandas as pd
import datetime
import akshare as ak

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
    
    def select_dataframe_all(self) -> pd.DataFrame:
        try:
            with get_db() as db:
                # 构造查询条件
                query = db.query(StockInfo)
                # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            return pd.DataFrame()
        
    def update_all_industry(self):
        logger.info("update all industry ...")
        def safe_value(val):
            if pd.isna(val) or val.strip() == '-':
                return None 
            else:
                return val
        try:
            with get_db() as db:
                # 构造查询条件
                stock_info_lst = db.query(StockInfo).all()
                for stock_info in stock_info_lst:
                    stock_code = stock_info.stock_code
                    try:
                        info_df = ak.stock_individual_info_em(symbol=stock_code)
                    except Exception as e:
                        logger.error("更新行业信息失败，股票代码：%s，错误信息：%s", stock_code, e)
                        continue
                    industry = info_df[info_df["item"] == "行业"].iloc[0]['value']
                    stock_info.industry = safe_value(industry)
                    logger.info(f"更新行业信息成功，股票代码：{stock_code}，行业：{industry}")
                db.commit()
        except Exception as e:
            logger.error("更新行业信息失败，错误信息：%s", e)
            db.rollback()
            raise e
        
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(StockInfo).delete()
                db.commit()
        except Exception as e:
            logger.error("Error deleting all StockInfo: %s", e)
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

    def upsert_one(self, obj: UpdateFlag) -> UpdateFlag:
        """
        根据主键 stock_code 判断记录是否存在，
        如果存在则更新（更新 action_update_flag 与 fundamental_update_flag 字段），
        如果不存在则插入记录。
        """
        try:
            with get_db() as db:
                existing = db.query(UpdateFlag).filter(UpdateFlag.stock_code == obj.stock_code).first()
                if existing:
                    # 更新已有记录的字段
                    existing.action_update_flag = obj.action_update_flag
                    existing.fundamental_update_flag = obj.fundamental_update_flag
                    db.commit()
                    db.refresh(existing)
                    logger.info("Updated update flag for stock %s", obj.stock_code)
                    return existing
                else:
                    db.add(obj)
                    db.commit()
                    db.refresh(obj)
                    logger.info("Inserted new update flag for stock %s", obj.stock_code)
                    return obj
        except Exception as e:
            logger.error("Error during upsert: %s", e)
            # 若在 with 块中发生异常，get_db() 应该自动回滚，但可以手动调用 rollback() 如下：
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
                        'fundamental_update_flag': result.fundamental_update_flag
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
        
    def update_fundamental_flag(self, stock_code: str, fundamental_update_flag: str):
        try:
            with get_db() as db:
                lines = db.query(UpdateFlag).filter(UpdateFlag.stock_code == stock_code).update({UpdateFlag.fundamental_update_flag: fundamental_update_flag})
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
            return pd.DataFrame()
        
    def select_by_code_date_type(self, stock_code, date, task_type, task_status) -> List[FutureTask]:
        """
        根据股票代码、日期和任务类型查询FutureTask对象
        """
        try:
            with get_db() as db:
                if task_status is not None:
                    result = db.query(FutureTask).filter(
                        FutureTask.stock_code == stock_code, 
                        FutureTask.task_date <= date,
                        FutureTask.task_type == task_type, 
                        FutureTask.task_status == task_status
                    ).all()
                else:
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
        
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(FutureTask).delete()
                db.commit()
        except Exception as e:
            logger.error("Error deleting all tasks: %s", e)
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
                logger.info(f"Using database bind: {db.bind}")
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
                logger.info(f"Using database bind: {db.bind}")
                def insert_one_batch(batch: List[StockHistUnadj]):
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
                query = db.query(StockHistUnadj)
                # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            return pd.DataFrame()
        
    def select_dataframe_by_code(self, stock_code: str):
        """
        查询指定股票的所有前复权历史数据，并返回为 Pandas DataFrame。
        
        :param stock_code: 股票代码，例如 "600012"
        :return: 包含查询结果的 DataFrame，如果没有数据则返回空 DataFrame。
        """
        try:
            with get_db() as db:
                # 构造查询条件
                query = db.query(StockHistUnadj).filter(StockHistUnadj.stock_code == stock_code)
                # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def select_after_date_as_dataframe(self, stock_code: str, date: Union[str, datetime.date]):
        """
        查询指定股票代码在当前日期之后的公司行动数据，并返回一个包含查询结果的 DataFrame。
        :param stock_code: 股票代码
        :return: 包含查询结果的 DataFrame，如果没有数据则返回空 DataFrame。
        """
        try:
            with get_db() as db:
                # 构造查询条件，查找 stock_code 相等且 date 大于指定日期的记录
                query = db.query(StockHistUnadj).filter(
                    StockHistUnadj.stock_code == stock_code,
                    StockHistUnadj.date >= date
                )
                # 使用 pd.read_sql 将查询结果转换为 DataFrame，
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            logger.error("Error querying after date for %s: %s", stock_code, e)
            return pd.DataFrame()
    
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(StockHistUnadj).delete()
                db.commit()
        except Exception as e:
            logger.error("Error deleting allrecords", e)
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
                def insert_one_batch(batch: List[CompanyAction]):
                    db.add_all(batch)
                    db.commit()
                    return batch
                process_in_batches(records, insert_one_batch)
                return records
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e
        
    def batch_insert_if_not_exists(self, records: List[CompanyAction]):
        """
        批量插入公司行动数据记录到数据库，如果记录已存在则跳过。
        每个批次提交一次。
        """
        try:
            with get_db() as db:
                inserted_all = []  # 用于保存所有批次中实际插入的记录

                def insert_one_batch(batch: List[CompanyAction]):
                    # 查询现有记录，检查是否存在相同 stock_code 和 ex_dividend_date
                    existing_records = db.query(CompanyAction).filter(
                        CompanyAction.stock_code.in_([record.stock_code for record in batch]),
                        CompanyAction.ex_dividend_date.in_([record.ex_dividend_date for record in batch])
                    ).all()
                    
                    # 从返回的查询结果中提取已存在的记录的组合（stock_code, ex_dividend_date）
                    existing_keys = set((record.stock_code, record.ex_dividend_date) for record in existing_records)
                    
                    # 仅保留不存在的记录
                    records_to_insert = [
                        record for record in batch
                        if (record.stock_code, record.ex_dividend_date) not in existing_keys
                    ]
                    
                    if records_to_insert:
                        db.add_all(records_to_insert)
                        db.commit()
                    
                    return records_to_insert
                
                # 处理批次插入
                batch_results  = process_in_batches(records, insert_one_batch)
                for sublist in batch_results:
                    inserted_all.extend(sublist)
                return inserted_all
        except Exception as e:
            logger.error("Error during batch insert if not exists: %s", e)
            db.rollback()
            raise e
        
    def select_dataframe_by_code(self, stock_code: str):
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
    
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(CompanyAction).delete()
                db.commit()
        except Exception as e:
            logger.error("Error deleting all records", e)
            db.rollback()
            raise e
        

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
                process_in_batches(records, insert_one_batch)
                return records
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e
        
    def select_dataframe_all(self):
        """
        查询所有股票的所有无复权历史数据，并返回为 Pandas DataFrame。
        """
        with get_db() as db:
            query = db.query(StockHistAdj)
            # 设置每次读取10000条记录
            chunksize = 10000
            logger.info("开始分块读取数据，每次读取 %d 条记录", chunksize)
            chunks = pd.read_sql(query.statement, db.bind, chunksize=chunksize)
            
            df_list = []
            chunk_index = 0
            for chunk in chunks:
                chunk_index += 1
                logger.info("已读取第 %d 块数据，大小：%d", chunk_index, len(chunk))
                df_list.append(chunk)
                
            df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
            logger.info("数据读取完毕，总记录数：%d", len(df))
            return df
    
    def select_dataframe_by_code(self, stock_code: str):
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
    
    def select_dataframe_by_date_range(self, start_date: str, end_date: str):
        """
        查询所有股票在指定日期范围内的的所有无复权历史数据（含start_date，不含end_date），并返回为 Pandas DataFrame。
        
        :param start_date: 开始日期，例如 "2020-01-01"
        :param end_date: 结束日期，例如 "2020-01-01"
        :return: 包含查询结果的 DataFrame，如果没有数据则返回空 DataFrame。
        """
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        with get_db() as db:
            query = db.query(StockHistAdj).filter(StockHistAdj.date >= start_date, StockHistAdj.date < end_date)
            # 设置每次读取10000条记录
            chunksize = 10000
            logger.info("开始分块读取数据，每次读取 %d 条记录", chunksize)
            chunks = pd.read_sql(query.statement, db.bind, chunksize=chunksize)
            
            df_list = []
            chunk_index = 0
            for chunk in chunks:
                chunk_index += 1
                logger.info("已读取第 %d 块数据，大小：%d", chunk_index, len(chunk))
                df_list.append(chunk)
                
            df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
            logger.info("数据读取完毕，总记录数：%d", len(df))
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
        
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(StockHistAdj).delete()
                db.commit()
        except Exception as e:
            logger.error("Error during delete: %s", e)
            db.rollback()
            raise e


class FundamentalDataDao:

    _instance = None  # 用于保存单例对象

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def get_latest_report_date(self, stock_code: str):
        try:
            with get_db() as db:
                result = db.query(FundamentalData.report_date).filter(FundamentalData.stock_code == stock_code).order_by(FundamentalData.report_date.desc()).first()
                if result:
                    return result[0]
                else:
                    return None
        except Exception as e:
            logger.error("Error querying latest report date for %s: %s", stock_code, e)
            return None

    def batch_upsert(self, records: List[FundamentalData]) -> List[FundamentalData]:
        """
        批量 upsert 基本面数据记录到数据库：
        根据 stock_code 和 report_date 查询，如果存在则更新，否则插入。
        优化：在每个批次内一次性查询出所有存在记录，减少数据库访问次数。
        """
        try:
            with get_db() as db:
                def upsert_one_batch(batch: List[FundamentalData]) -> List[FundamentalData]:
                    # 1. 构造联合主键列表
                    keys = [(rec.stock_code, rec.report_date) for rec in batch]
                    # 2. 一次性查询出数据库中已存在的记录
                    existing_records = db.query(FundamentalData).filter(
                        tuple_(FundamentalData.stock_code, FundamentalData.report_date).in_(keys)
                    ).all()
                    # 构建字典，键为 (stock_code, report_date)
                    existing_dict = {
                        (rec.stock_code, rec.report_date): rec for rec in existing_records
                    }
                    # 3. 遍历批次记录，进行 upsert
                    for rec in batch:
                        key = (rec.stock_code, rec.report_date)
                        if key in existing_dict:
                            existing = existing_dict[key]
                            # 更新字段
                            existing.total_equity = rec.total_equity
                            existing.total_assets = rec.total_assets
                            existing.current_liabilities = rec.current_liabilities
                            existing.noncurrent_liabilities = rec.noncurrent_liabilities
                            existing.net_profit = rec.net_profit
                            existing.operating_profit = rec.operating_profit
                            existing.total_revenue = rec.total_revenue
                            existing.total_cost = rec.total_cost
                            existing.net_cash_from_operating = rec.net_cash_from_operating
                            existing.cash_for_fixed_assets = rec.cash_for_fixed_assets
                        else:
                            db.add(rec)
                    db.commit()
                    return batch
                process_in_batches(records, upsert_one_batch)
                return records
        except Exception as e:
            logger.error("Error during batch upsert: %s", e)
            db.rollback()
            raise e
        
    def select_dataframe_by_code(self, stock_code: str):
        with get_db() as db:
            # 构造查询条件
            query = db.query(FundamentalData).filter(FundamentalData.stock_code == stock_code)
            # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
            df = pd.read_sql(query.statement, db.bind)
        return df
    
    def select_dataframe_all(self):
        with get_db() as db:
            # 构造查询条件
            query = db.query(FundamentalData)
            # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
            df = pd.read_sql(query.statement, db.bind)
        return df
    
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(FundamentalData).delete()
                db.commit()
        except Exception as e:
            logger.error("Error during delete: %s", e)
            db.rollback()
            raise e


class SuspendDataDao:
    _instance = None

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance

    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def get_by_stock_code_and_suspend_date(self, stock_code: str, suspend_date):
        try:
            with get_db() as db:
                record = db.query(SuspendData).filter(
                    SuspendData.stock_code == stock_code,
                    SuspendData.suspend_date == suspend_date
                ).first()
            return record
        except Exception as e:
            logger.error("Error query suspend data for stock %s, date %s: %s", stock_code, suspend_date, e)
            return None

    def insert(self, record: SuspendData):
        try:
            with get_db() as db:
                db.add(record)
                db.commit()
            return record
        except Exception as e:
            logger.error("Error insert suspend data: %s", e)
            db.rollback()
            raise e

    def update(self, existing: SuspendData, new_data: dict):
        try:
            with get_db() as db:
                for key, value in new_data.items():
                    setattr(existing, key, value)
                db.commit()
            return existing
        except Exception as e:
            logger.error("Error update suspend data: %s", e)
            db.rollback()
            raise e

    def batch_upsert(self, records: list):
        """
        对传入的 SuspendData 列表进行 upsert 操作：
        对于每条记录，根据股票代码和停牌时间判断是否存在，存在则更新，否则插入。
        """
        results = []
        for record in records:
            existing = self.get_by_stock_code_and_suspend_date(record.stock_code, record.suspend_date)
            if existing:
                update_data = {
                    "resume_date": record.resume_date,
                    "suspend_period": record.suspend_period,
                    "suspend_reason": record.suspend_reason,
                    "market": record.market,
                }
                updated = self.update(existing, update_data)
                results.append(updated)
            else:
                inserted = self.insert(record)
                results.append(inserted)
        return results
    
    def select_dataframe_all(self):
        with get_db() as db:
            # 构造查询条件
            query = db.query(SuspendData)
            # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
            df = pd.read_sql(query.statement, db.bind)
        return df
    
    def get_suspended_stocks_by_date(self, query_date: datetime.date):
        """
        查询指定日期发生停牌或正在停牌的股票列表，返回所有字段的记录。
        
        条件：
          - suspend_date <= query_date
          - resume_date 为 NULL 或 resume_date >= query_date
          
        :param query_date: 查询日期，类型为 datetime.date
        :return: 满足条件的 DataFrame
        """
        with get_db() as db:
            query = db.query(SuspendData).filter(
                SuspendData.suspend_date <= query_date,
                or_(SuspendData.resume_date == None, SuspendData.resume_date >= query_date)
            )
            df = pd.read_sql(query.statement, db.bind)
            return df
        
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(SuspendData).delete()
                db.commit()
        except Exception as e:
            logger.error("Error during delete: %s", e)
            db.rollback()
            raise e

class StockShareChangeCNInfoDao:
    _instance = None

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance

    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def insert(self, record: StockShareChangeCNInfo):
        try:
            with get_db() as db:
                db.add(record)
                db.commit()
            return record
        except Exception as e:
            logger.error("Error insert suspend data: %s", e)
            db.rollback()
            raise e
        
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(StockShareChangeCNInfo).delete()
                db.commit()
        except Exception as e:
            logger.error("Error delete all stock share change records: %s", e)
            db.rollback()
            raise e
        
    def select_dataframe_all(self):
        try:
            with get_db() as db:
                query = db.query(StockShareChangeCNInfo)
                df = pd.read_sql(query.statement, db.bind)
                return df
        except Exception as e:
            return pd.DataFrame()
    
    def select_dataframe_by_stock_code(self, stock_code: str):
        try:
            with get_db() as db:
                query = db.query(StockShareChangeCNInfo).filter(StockShareChangeCNInfo.SECCODE == stock_code)
                df = pd.read_sql(query.statement, db.bind)
                return df
        except Exception as e:
            return pd.DataFrame()


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
FundamentalDataDao._instance = object.__new__(FundamentalDataDao)
FundamentalDataDao._instance.__init__()
SuspendDataDao._instance = object.__new__(SuspendDataDao)
SuspendDataDao._instance.__init__()
StockShareChangeCNInfoDao._instance = object.__new__(StockShareChangeCNInfoDao)
StockShareChangeCNInfoDao._instance.__init__()