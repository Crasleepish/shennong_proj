from typing import List, Union
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, tuple_
from app.models.stock_models import StockInfo, StockHistUnadj, UpdateFlag, FutureTask, CompanyAction, StockHistAdj, AdjFactor, FundamentalData, SuspendData, StockShareChangeCNInfo, MarketFactors
from app.database import get_db
from app.utils.data_utils import process_in_batches
from app.utils.db_utils import chunked_query_to_dataframe
import logging
import pandas as pd
import datetime
import akshare as ak

logger = logging.getLogger(__name__)

class StockInfoDao:

    @staticmethod
    def load_stock_info() -> List[StockInfo]:
        try:
            with get_db() as db:
                stock_info_lst = db.query(StockInfo).all()
                return stock_info_lst
        except Exception as e:
            logger.error(e)
        
    @staticmethod
    def batch_insert(stock_info_lst: List[StockInfo]):
        try:
            with get_db() as db:
                db.add_all(stock_info_lst)
                db.commit()
                return stock_info_lst
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e
        
    @staticmethod
    def batch_upsert(stock_info_lst: List[StockInfo]):
        """
        批量 upsert（存在则更新，不存在则插入）StockInfo 记录
        """
        if not stock_info_lst:
            logger.info("batch_upsert 传入空列表，跳过处理")
            return []

        try:
            with get_db() as db:
                # 获取所有传入记录的 stock_code
                input_codes = [s.stock_code for s in stock_info_lst]

                # 查找数据库中已存在的记录
                existing_records = db.query(StockInfo).filter(StockInfo.stock_code.in_(input_codes)).all()
                existing_dict = {r.stock_code: r for r in existing_records}

                inserted = 0
                updated = 0
                for new_rec in stock_info_lst:
                    if new_rec.stock_code in existing_dict:
                        existing = existing_dict[new_rec.stock_code]
                        # 更新字段（保留主键 stock_code）
                        existing.stock_name = new_rec.stock_name
                        existing.market = new_rec.market
                        existing.exchange = new_rec.exchange
                        existing.industry = new_rec.industry
                        existing.listing_date = new_rec.listing_date
                        existing.list_status = new_rec.list_status
                        updated += 1
                    else:
                        db.add(new_rec)
                        inserted += 1

                db.commit()
                logger.info(f"StockInfo batch_upsert 完成：新增 {inserted} 条，更新 {updated} 条")
                return stock_info_lst
        except Exception as e:
            logger.error("StockInfo batch_upsert 失败: %s", e)
            db.rollback()
            raise e
    
    @staticmethod
    def select_dataframe_all() -> pd.DataFrame:
        try:
            with get_db() as db:
                # 构造查询条件
                query = db.query(StockInfo)
                # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
                df = pd.read_sql(query.statement, db.bind)
            return df
        except Exception as e:
            return pd.DataFrame()
        
    @staticmethod
    def update_industry_by_mapping(stock_codes: List[str], industry_name: str):
        """
        将指定的一组股票代码的行业字段统一更新为 industry_name
        """
        updated = 0
        if not stock_codes:
            logger.warning("传入的股票代码列表为空，跳过行业更新")
            return
        try:
            with get_db() as db:
                stock_info_list = db.query(StockInfo).filter(StockInfo.stock_code.in_(stock_codes)).all()
                for stock in stock_info_list:
                    stock.industry = industry_name
                    updated += 1
                db.commit()
            logger.info(f"成功将 {updated} 支股票的行业更新为：{industry_name}")
        except Exception as e:
            logger.error("更新行业信息失败: %s", e)
            raise e

    @staticmethod
    def delete_all():
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

    @staticmethod
    def get_latest_date(stock_code: str):
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

    @staticmethod
    def batch_insert(records: List[StockHistUnadj]):
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
        
    @staticmethod
    def batch_upsert(records: List[StockHistUnadj]):
        if not records:
            return []

        try:
            with get_db() as db:
                logger.info(f"Using database bind: {db.bind}")

                def upsert_one_batch(batch: List[StockHistUnadj]):
                    # 先查出现有记录
                    stock_code_list = [rec.stock_code for rec in batch]
                    date_list = [rec.date for rec in batch]

                    # 查询数据库已有记录
                    existing_records = db.query(StockHistUnadj).filter(
                        StockHistUnadj.stock_code.in_(stock_code_list),
                        StockHistUnadj.date.in_(date_list)
                    ).all()

                    # 建立快速索引
                    existing_dict = {
                        (rec.stock_code, rec.date): rec for rec in existing_records
                    }

                    # 遍历每条新记录，判断是update还是insert
                    for rec in batch:
                        key = (rec.stock_code, rec.date)
                        if key in existing_dict:
                            # 已存在，更新字段
                            existing = existing_dict[key]
                            existing.open = rec.open
                            existing.close = rec.close
                            existing.high = rec.high
                            existing.low = rec.low
                            existing.volume = rec.volume
                            existing.amount = rec.amount
                            existing.pre_close = rec.pre_close
                            existing.change_percent = rec.change_percent
                            existing.change = rec.change
                            existing.turnover_rate = rec.turnover_rate
                            existing.turnover_rate_f = rec.turnover_rate_f
                            existing.volume_ratio = rec.volume_ratio
                            existing.pe = rec.pe
                            existing.pe_ttm = rec.pe_ttm
                            existing.pb = rec.pb
                            existing.ps = rec.ps
                            existing.ps_ttm = rec.ps_ttm
                            existing.dv_ratio = rec.dv_ratio
                            existing.dv_ttm = rec.dv_ttm
                            existing.total_shares = rec.total_shares
                            existing.float_shares = rec.float_shares
                            existing.free_shares = rec.free_shares
                            existing.mkt_cap = rec.mkt_cap
                            existing.circ_mv = rec.circ_mv
                        else:
                            # 不存在，新增
                            db.add(rec)

                    db.commit()
                    return batch

                # 批量处理，默认1000条一批
                process_in_batches(records, upsert_one_batch)

                return records

        except Exception as e:
            logger.error("Error during batch upsert: %s", e)
            db.rollback()
            raise e
    
    @staticmethod
    def select_dataframe_by_code(stock_code: str):
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
    
    @staticmethod
    def select_dataframe_by_date_range(stock_code: str, start_date, end_date):
        try:
            with get_db() as db:
                selected_columns = [
                    StockHistUnadj.stock_code,
                    StockHistUnadj.date,
                    StockHistUnadj.close,
                    StockHistUnadj.volume,
                    StockHistUnadj.amount,
                    StockHistUnadj.pre_close,
                    StockHistUnadj.change_percent,
                    StockHistUnadj.change,
                    StockHistUnadj.mkt_cap,
                    StockHistUnadj.circ_mv,
                    StockInfo.exchange
                ]
                # 仅查询沪深上市的股票
                query = db.query(*selected_columns).join(
                    StockInfo, StockHistUnadj.stock_code == StockInfo.stock_code
                ).filter(
                    StockInfo.exchange.in_(["SSE", "SZSE"])
                )

                if stock_code:
                    query = query.filter(StockHistUnadj.stock_code == stock_code)
                if start_date:
                    query = query.filter(StockHistUnadj.date >= start_date)
                if end_date:
                    query = query.filter(StockHistUnadj.date <= end_date)

                query = query.order_by(StockHistUnadj.date.asc())

                return chunked_query_to_dataframe(query, table_name="stock_hist_unadj + stock_info")

        except Exception as e:
            logger.error("Error querying stock_hist_unadj with filters: %s", e)
            return pd.DataFrame()
    
    @staticmethod
    def delete_all():
        try:
            with get_db() as db:
                db.query(StockHistUnadj).delete()
                db.commit()
        except Exception as e:
            logger.error("Error deleting allrecords", e)
            db.rollback()
            raise e

class AdjFactorDao:

    _instance = None  # 用于保存单例对象

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AdjFactorDao, cls).__new__(cls)
        # 始终返回已经创建好的 _instance
        return cls._instance
    
    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def get_latest_date(self, stock_code: str):
        """
        查询指定股票在复权因子表中的最新日期，如果没有数据则返回 None。
        """
        try:
            with get_db() as db:
                logger.info(f"Using database bind: {db.bind}")
                result = db.query(AdjFactor.date).filter(AdjFactor.stock_code == stock_code).order_by(AdjFactor.date.desc()).first()
                if result:
                    return result[0]
                else:
                    return None
        except Exception as e:
            logger.error("Error querying latest adj_factor date for %s: %s", stock_code, e)
            return None

    def batch_insert(self, records: List[AdjFactor]):
        """
        批量插入复权因子记录到数据库。
        """
        try:
            with get_db() as db:
                logger.info(f"Using database bind: {db.bind}")
                def insert_one_batch(batch: List[AdjFactor]):
                    db.add_all(batch)
                    db.commit()
                    return batch
                process_in_batches(records, insert_one_batch)
                return records
        except Exception as e:
            logger.error("Error during adj_factor batch insert: %s", e)
            db.rollback()
            raise e

    def batch_upsert(self, records: List[AdjFactor]):
        if not records:
            return []

        try:
            with get_db() as db:
                logger.info(f"Using database bind: {db.bind}")

                def upsert_one_batch(batch: List[AdjFactor]):
                    # 先提取批次中的股票代码和日期
                    stock_code_list = [rec.stock_code for rec in batch]
                    date_list = [rec.date for rec in batch]

                    # 查询数据库中已存在的记录
                    existing_records = db.query(AdjFactor).filter(
                        AdjFactor.stock_code.in_(stock_code_list),
                        AdjFactor.date.in_(date_list)
                    ).all()

                    # 生成字典快速索引
                    existing_dict = {
                        (rec.stock_code, rec.date): rec for rec in existing_records
                    }

                    # 遍历新记录，做upsert
                    for rec in batch:
                        key = (rec.stock_code, rec.date)
                        if key in existing_dict:
                            # 已存在，更新 adj_factor
                            existing = existing_dict[key]
                            existing.adj_factor = rec.adj_factor
                        else:
                            # 不存在，新增
                            db.add(rec)

                    db.commit()
                    return batch

                # 分批处理
                process_in_batches(records, upsert_one_batch)

                return records

        except Exception as e:
            logger.error("Error during adj_factor batch upsert: %s", e)
            db.rollback()
            raise e
    
    def delete_stock_data(self, stock_code: str):
        """
        删除指定股票的复权因子记录。
        """
        try:
            with get_db() as db:
                db.query(AdjFactor).filter(AdjFactor.stock_code == stock_code).delete()
                db.commit()
                logger.info(f"Deleted all adj_factor records for stock {stock_code}")
        except Exception as e:
            logger.error("Error deleting adj_factor records for %s: %s", stock_code, e)
            db.rollback()
            raise e
    
    def delete_all(self):
        """
        删除复权因子表中的所有记录。
        """
        try:
            with get_db() as db:
                db.query(AdjFactor).delete()
                db.commit()
                logger.info("Deleted all adj_factor records")
        except Exception as e:
            logger.error("Error deleting all adj_factor records: %s", e)
            db.rollback()
            raise e
    
    def get_adj_factor_dataframe(self, stock_code: str, start_date=None, end_date=None):
        try:
            with get_db() as db:
                # 构建联表查询
                query = db.query(AdjFactor).join(
                    StockInfo, AdjFactor.stock_code == StockInfo.stock_code
                ).filter(
                    StockInfo.exchange.in_(["SSE", "SZSE"])
                )

                # 添加条件
                if stock_code:
                    query = query.filter(AdjFactor.stock_code == stock_code)
                if start_date:
                    query = query.filter(AdjFactor.date >= start_date)
                if end_date:
                    query = query.filter(AdjFactor.date <= end_date)

                query = query.order_by(AdjFactor.date.asc())

                # 分块读取
                return chunked_query_to_dataframe(
                    base_query=query,
                    table_name="adj_factor + stock_info",
                    chunksize=10000
                )

        except Exception as e:
            logger.error("Error getting adj_factor dataframe for %s: %s", stock_code, e)
            return pd.DataFrame()

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

    @staticmethod
    def get_latest_report_date(stock_code: str):
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

    @staticmethod
    def batch_upsert(records: List[FundamentalData]) -> List[FundamentalData]:
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

    @staticmethod   
    def select_dataframe_by_code(stock_code: str):
        with get_db() as db:
            # 构造查询条件
            query = db.query(FundamentalData).filter(FundamentalData.stock_code == stock_code)
            # 使用 pd.read_sql 将 SQLAlchemy 查询转换为 DataFrame
            df = pd.read_sql(query.statement, db.bind)
        return df
    
    @staticmethod
    def select_dataframe_all():
        """
        分块读取所有 exchange ∈ {'SSE', 'SZSE'} 的 FundamentalData 数据，返回 DataFrame。
        """
        try:
            with get_db() as db:
                # 构造联表查询，过滤 exchange
                query = db.query(FundamentalData).join(
                    StockInfo, FundamentalData.stock_code == StockInfo.stock_code
                ).filter(
                    StockInfo.exchange.in_(["SSE", "SZSE"])
                ).order_by(FundamentalData.stock_code, FundamentalData.report_date)

                # 使用通用 join 分块读取工具
                return chunked_query_to_dataframe(
                    base_query=query,
                    table_name="fundamental_data + stock_info",
                    chunksize=10000
                )

        except Exception as e:
            logger.error("Error selecting fundamental data with exchange filter: %s", e)
            return pd.DataFrame()
    
    @staticmethod
    def delete_all():
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
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def insert(self, record: SuspendData):
        try:
            with get_db() as db:
                db.merge(record)
                db.commit()
            return record
        except Exception as e:
            logger.error("Error insert suspend data: %s", e)
            db.rollback()
            raise e

    def batch_insert(self, records: list):
        try:
            with get_db() as db:
                db.add_all(records)
                db.commit()
            logger.info("Inserted %d suspend records.", len(records))
            return records
        except Exception as e:
            logger.error("Error during batch insert of suspend data: %s", e)
            db.rollback()
            raise e

    def select_dataframe_all(self):
        with get_db() as db:
            query = db.query(SuspendData)
            return pd.read_sql(query.statement, db.bind)

    def get_suspended_stocks_by_date(self, query_date: datetime.date):
        with get_db() as db:
            query = db.query(SuspendData).filter(
                SuspendData.trade_date <= query_date
            )
            return pd.read_sql(query.statement, db.bind)
    def delete_by_date_range(self, start: datetime.date, end: datetime.date):
        """
        删除指定交易日区间的所有停复牌记录。
        :param start: 开始日期
        :param end: 结束日期
        """
        try:
            with get_db() as db:
                db.query(SuspendData).filter(
                    SuspendData.trade_date >= start, 
                    SuspendData.trade_date <= end
                    ).delete()
                db.commit()
                logger.info("Deleted suspend records for trade_date: %s to %s", start, end)
        except Exception as e:
            logger.error("Error deleting suspend records for trade_date %s to %s: %s", start, end, e)
            db.rollback()
            raise e
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
        
class MarketFactorsDao:
    _instance = None

    def __new__(cls, *args, **kwargs):
        # 始终返回已经创建好的 _instance
        return cls._instance

    def __init__(self):
        # __init__ 可能会被多次调用，因此通过 _initialized 标识确保只初始化一次
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def insert(self, record: MarketFactors):
        try:
            with get_db() as db:
                db.add(record)
                db.commit()
            return record
        except Exception as e:
            logger.error("Error insert market factors: %s", e)
            db.rollback()
            raise e

    def upsert_one(self, record: MarketFactors):
        try:
            with get_db() as db:
                existing = db.query(MarketFactors).filter(MarketFactors.date == record.date).first()
                if existing:
                    existing.MKT = record.MKT
                    existing.SMB = record.SMB
                    existing.HML = record.HML
                    existing.QMJ = record.QMJ
                    existing.VOL = record.VOL
                    existing.LIQ = record.LIQ
                    db.commit()
                    db.refresh(existing)
                    logger.info("Market factors updated successfully.")
                else:
                    db.add(record)
                    db.commit()
                    db.refresh(record)
                    logger.info("Market factors inserted successfully.")
        except Exception as e:
            logger.error("Error upsert market factors: %s", e)
            db.rollback()
            raise e
        
    def delete_all(self):
        try:
            with get_db() as db:
                db.query(MarketFactors).delete()
                db.commit()
        except Exception as e:
            logger.error("Error delete all market factors records: %s", e)
            db.rollback()
            raise e
        
    def select_dataframe_by_date(self, start_date: str, end_date: str):
        start_date_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        try:
            with get_db() as db:
                select_columns = [MarketFactors.date, MarketFactors.MKT, MarketFactors.SMB, MarketFactors.HML, MarketFactors.QMJ]
                query = db.query(*select_columns).filter(MarketFactors.date >= start_date_dt, MarketFactors.date < end_date_dt)
                df = pd.read_sql(query.statement, db.bind)
                return df
        except Exception as e:
            return pd.DataFrame()
        
        

UpdateFlagDao._instance = object.__new__(UpdateFlagDao)
UpdateFlagDao._instance.__init__()
FutureTaskDao._instance = object.__new__(FutureTaskDao)
FutureTaskDao._instance.__init__()
AdjFactorDao._instance = object.__new__(AdjFactorDao)
AdjFactorDao._instance.__init__()
CompanyActionDao._instance = object.__new__(CompanyActionDao)
CompanyActionDao._instance.__init__()
StockHistAdjDao._instance = object.__new__(StockHistAdjDao)
StockHistAdjDao._instance.__init__()
SuspendDataDao._instance = object.__new__(SuspendDataDao)
SuspendDataDao._instance.__init__()
StockShareChangeCNInfoDao._instance = object.__new__(StockShareChangeCNInfoDao)
StockShareChangeCNInfoDao._instance.__init__()
MarketFactorsDao._instance = object.__new__(MarketFactorsDao)
MarketFactorsDao._instance.__init__()