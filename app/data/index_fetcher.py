from typing import List
from bisect import bisect_left
import akshare as ak
import pandas as pd
import datetime
import math
import asyncio
from sqlalchemy.orm import Session
from app.models.index_models import IndexInfo, IndexHist
from app.dao.index_info_dao import IndexInfoDao, IndexHistDao
from app.constants.enums import TaskType, TaskStatus
from typing import Union
import re

import logging

logger = logging.getLogger(__name__)

def safe_value(val):
    return None if pd.isna(val) else val

def safe_get(val):
    return 0.0 if val is None or pd.isna(val) else val

class IndexInfoSynchronizer:
    """
    同步股票信息数据到数据库。
    
    主要流程：
      1. 调用 akshare 接口获取数据（pandas DataFrame）
      2. 查询数据库中已存在的指数代码集合
      3. 筛选出新增记录（基于指数代码）
      4. 将新增记录插入数据库（仅新增，不更新或删除）
    """
    def __init__(self):
        self.index_info_dao = IndexInfoDao._instance

    def fetch_data(self) -> pd.DataFrame:
        """
        调用 akshare 接口获取数据，返回 pandas DataFrame
        
        示例接口：
            import akshare as ak
            df = ak.stock_zh_index_spot_em(symbol="sz399812")
        """
        logger.info("Fetching data from akshare ...")
        sh_list = ak.stock_zh_index_spot_em(symbol="上证系列指数")[['代码', '名称']]
        sh_list = sh_list.rename(columns={'代码': 'code', '名称': 'name'})
        sh_list['market'] = 'sh'

        sz_list = ak.stock_zh_index_spot_em(symbol="深证系列指数")[['代码', '名称']]
        sz_list = sz_list.rename(columns={'代码': 'code', '名称': 'name'})
        sz_list['market'] = 'sz'

        zz_list = ak.stock_zh_index_spot_em(symbol="中证系列指数")[['代码', '名称']]
        zz_list = zz_list.rename(columns={'代码': 'code', '名称': 'name'})
        zz_list['market'] = 'csi'

        df = pd.concat([sh_list, sz_list, zz_list], ignore_index=True).reset_index(drop=True)
        df.drop_duplicates(subset=['code'], keep='last', inplace=True)
        return df

    def sync(self, progress_callback=None):
        """
        同步数据：将接口返回的新增记录插入到数据库中
        """
        logger.info("Starting synchronization for index info.")
        # self.index_info_dao.update_all_industry()
        df = self.fetch_data()
        if df.empty:
            logger.warning("Fetched data is empty.")
            return
        
        # 获取数据库中已有的股票代码集合
        try:
            index_info_lst : List[IndexInfo] = self.index_info_dao.load_index_info()
            existing_codes = {si.index_code for si in index_info_lst}
            logger.debug("Existing index codes in DB: %s", existing_codes)
            
            # 筛选出新增数据（证券代码不在 existing_codes 中）
            new_data = df[~df['code'].isin(existing_codes)]
            logger.info("Found %d new records to insert.", len(new_data))
            
            if new_data.empty:
                logger.info("No new records to insert.")
                return
            
            # 将 DataFrame 中每一行转换为 IndexInfo 对象
            new_records = []
            for idx, row in new_data.iterrows():
                record = IndexInfo(
                    index_code=row['code'],
                    index_name=safe_value(row['name']),
                    market=safe_value(row['market'])
                )
                new_records.append(record)
                if progress_callback:
                    if idx % 100 == 0:
                        progress_callback(idx, len(new_records))
            
            # 批量插入新增记录
            self.index_info_dao.batch_insert(new_records)

            logger.info("Inserted %d new records into the database.", len(new_records))
            if progress_callback:
                progress_callback(len(new_records), len(new_records))
        except Exception as e:
            logger.exception("Error during synchronization: %s", str(e))
            raise e

class IndexHistSynchronizer:
    """
    同步指数历史行情数据到数据库。
    
    主要流程：
      1. 查看当前数据库中的指数列表
      2. 遍历指数列表，依次同步每只指数的历史行情数据
      3. 对于某一支指数，查看其最新数据的日期
      4. 如果最新日期小于等于当前日期，则同步该指数的历史行情数据
      5. 将新增的数据插入数据库
    """
    def __init__(self):
        self.index_info_dao = IndexInfoDao._instance
        self.index_hist_dao = IndexHistDao._instance
        self.loop = asyncio.get_event_loop()
        self.index_list_size = 0
        self.completed_num = 0
        self.is_running = False
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(6)

    def initialize(self):
        self.index_list_size = 0
        self.completed_num = 0
        self.is_running = True

    def terminate(self):
        self.index_list_size = 0
        self.completed_num = 0
        self.is_running = False

    async def fetch_index_data(self, index_code, start_date, current_date):
        """异步获取历史行情数据"""
        try:
            async with self.semaphore:
                df = await asyncio.to_thread(ak.index_zh_a_hist, 
                                            symbol=index_code, 
                                            start_date=start_date, 
                                            end_date=current_date)
                return df
        except Exception as e:
            logger.error("Error fetching index data for %s: %s", index_code, e)
            return pd.DataFrame()
        
    pd.DataFrame()
        
    async def batch_insert(self, index_code, new_records):
        def _syn_batch_insert(index_code, new_records):
            # 5. 批量插入新增记录到数据库
            if new_records:
                try:
                    result_records = self.index_hist_dao.batch_insert(new_records)
                    logger.info("Inserted %d new historical records for index %s.", len(result_records), index_code)
                except Exception as e:
                    logger.error("Error inserting new records for index %s: %s", index_code, e)
                    raise e
            else:
                logger.info("No new records to insert for index %s.", index_code)
        async with self.semaphore:
            return await asyncio.to_thread(_syn_batch_insert, index_code, new_records)
    
    async def process_data(self, index_code, start_date, current_date, progress_callback=None):
        try:
            # 3. 调用 akshare 接口获取该股票从 start_date 到 current_date 的历史行情数据（不复权）
            logger.info("Fetching historical data for index %s from %s to %s", index_code, start_date, current_date)
            df = await self.loop.create_task(self.fetch_index_data(index_code, start_date, current_date))

            if df.empty:
                logger.info("No new historical data for index %s.", index_code)
                return

            # 将日期列转换为日期类型
            if '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'], errors='coerce')

            # 处理 DataFrame 数据，构造 IndexHist 对象列表
            new_records = []
            for _, row in df.iterrows():
                record = IndexHist(
                    date=row["日期"].date(),
                    index_code=index_code,
                    open=safe_value(row["开盘"]),
                    close=safe_value(row["收盘"]),
                    high=safe_value(row["最高"]),
                    low=safe_value(row["最低"]),
                    volume=safe_value(row["成交量"]) * 100,
                    amount=safe_value(row["成交额"]),
                    amplitude=safe_value(row["振幅"]),
                    change_percent=safe_value(row["涨跌幅"]),
                    change=safe_value(row["涨跌额"]),
                    turnover_rate=safe_value(row["换手率"])
                )
                new_records.append(record)

            # 5. 批量插入新增记录到数据库
            await self.batch_insert(index_code, new_records)
        except Exception as e:
            logger.error("Error processing index %s: %s", index_code, e)
        finally:
            async with self.lock:
                self.completed_num = self.completed_num + 1
                if progress_callback:
                    progress_callback(self.completed_num, self.index_list_size)

    def sync(self, progress_callback=None):
        self.initialize()
        try:
            logger.info("Starting index historical data synchronization.")
            # 1. 获取当前数据库中的股票列表
            index_list = self.index_info_dao.load_index_info()
            self.index_list_size = len(index_list)
            logger.info("Found %d index to synchronize.", len(index_list))

            batch_size = 20
            for i in range(0, len(index_list), batch_size):
                batch = index_list[i : i + batch_size]
                tasks = []
                for index in batch:
                    index_code = index.index_code
                    logger.info("Synchronizing index %s", index_code)
                    # 2. 查询该股票在历史行情数据表中的最新日期
                    latest_date = self.index_hist_dao.get_latest_date(index_code)
                    if latest_date is None:
                        # 如果没有数据，则从一个默认的开始日期开始，例如 19901126
                        start_date = "19901126"
                    else:
                        # 否则从最新日期的下一天开始同步
                        next_date = latest_date + datetime.timedelta(days=1)
                        start_date = next_date.strftime("%Y%m%d")
                    
                    # 当前日期，格式化为 YYYYMMDD
                    current_date = datetime.datetime.now().strftime("%Y%m%d")
                    
                    # 如果开始日期大于当前日期，说明数据已更新完毕
                    if start_date > current_date:
                        logger.info("Index %s is up to date. (start_date: %s, current_date: %s)", index_code, start_date, current_date)
                        continue

                    tasks.append(self.loop.create_task(self.process_data(index_code, start_date, current_date, progress_callback)))
                if tasks:
                    self.loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            self.terminate()
    
index_info_synchronizer = IndexInfoSynchronizer()
index_hist_synchronizer = IndexHistSynchronizer()
