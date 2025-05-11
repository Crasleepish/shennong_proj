from typing import List
from bisect import bisect_left
import akshare as ak
import tushare as ts
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
import time

import logging

logger = logging.getLogger(__name__)

tspro = ts.pro_api()

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

        csi_list = tspro.index_basic(market='CSI')[['ts_code', 'name', 'market']]
        csi_list = csi_list.rename(columns={'ts_code': 'code', 'name': 'name', 'market': 'market'})

        sse_list = tspro.index_basic(market='SSE')[['ts_code', 'name', 'market']]
        sse_list = sse_list.rename(columns={'ts_code': 'code', 'name': 'name', 'market': 'market'})

        szse_list = tspro.index_basic(market='SZSE')[['ts_code', 'name', 'market']]
        szse_list = szse_list.rename(columns={'ts_code': 'code', 'name': 'name', 'market': 'market'})

        df = pd.concat([csi_list, sse_list, szse_list], ignore_index=True).reset_index(drop=True)
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
      6. 如果从开始日期到结束日期的范围长度超过了20年，则结束日期定为开始日期+20年
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
        self.api_call_reset_time = time.time() + 10
        self.api_call_counter = 0
        self.failed_indexes = []

    def initialize(self):
        self.index_list_size = 0
        self.completed_num = 0
        self.is_running = True
        self.api_call_reset_time = time.time() + 10
        self.api_call_counter = 0
        self.failed_indexes = []

    def terminate(self):
        self.index_list_size = 0
        self.completed_num = 0
        self.is_running = False

    async def fetch_index_data(self, index_code, start_date, current_date):
        """异步获取历史行情数据"""
        try:
            await self._check_api_rate_limit()
            async with self.semaphore:
                df = await asyncio.to_thread(tspro.index_daily, 
                                            ts_code=index_code, 
                                            start_date=start_date, 
                                            end_date=current_date)
                # 去重：保留同一 ts_code + nav_date 中最后一条记录
                df = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
                return df
        except Exception as e:
            logger.error("Error fetching index data for %s: %s", index_code, e)
            return pd.DataFrame()
        
    pd.DataFrame()

    async def _check_api_rate_limit(self):
        """检查并控制API调用频率，确保每分钟不超过200次"""
        current_time = time.time()
        
        if current_time > self.api_call_reset_time:
            # Reset counter if a minute has passed
            self.api_call_counter = 0
            self.api_call_reset_time = current_time + 10
            logger.info("重置计数器，下一次重置%s", self.api_call_reset_time)
            
        if self.api_call_counter >= 40:
            # Wait until the next minute if we've reached the limit
            wait_time = self.api_call_reset_time - current_time
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        else:
            self.api_call_counter += 1
            logger.info("已调用%s次" % self.api_call_counter)
        
    async def batch_insert(self, index_code, new_records):
        def _syn_batch_insert(index_code, new_records):
            # 5. 批量插入新增记录到数据库，如果存在则更新
            if new_records:
                try:
                    result_records = self.index_hist_dao.batch_upsert(new_records)
                    logger.info("Inserted %d new historical records for index %s.", len(result_records), index_code)
                except Exception as e:
                    logger.error("Error inserting new records for index %s: %s", index_code, e)
                    raise e
            else:
                logger.info("No new records to insert for index %s.", index_code)
        async with self.semaphore:
            return await asyncio.to_thread(_syn_batch_insert, index_code, new_records)
    
    async def process_data(self, index_code, start_date, current_date):
        try:
            # 3. 调用 akshare 接口获取该股票从 start_date 到 current_date 的历史行情数据（不复权）
            logger.info("Fetching historical data for index %s from %s to %s", index_code, start_date, current_date)
            df = await self.loop.create_task(self.fetch_index_data(index_code, start_date, current_date))

            if df.empty:
                logger.info("No new historical data for index %s.", index_code)
                return

            # 将日期列转换为日期类型
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')

            # 处理 DataFrame 数据，构造 IndexHist 对象列表
            new_records = []
            for _, row in df.iterrows():
                record = IndexHist(
                    date=row["trade_date"].date(),
                    index_code=index_code,
                    open=safe_value(row["open"]),
                    close=safe_value(row["close"]),
                    high=safe_value(row["high"]),
                    low=safe_value(row["low"]),
                    volume=safe_value(row["vol"]) * 100,
                    amount=safe_value(row["amount"]) * 1000,
                    change_percent=safe_value(row["pct_chg"]),
                    change=safe_value(row["change"])
                )
                new_records.append(record)

            # 5. 批量插入新增记录到数据库
            await self.batch_insert(index_code, new_records)
        except Exception as e:
            logger.error("Error processing index %s: %s", index_code, e)

    def sync(self, progress_callback=None):
        self.initialize()
        try:
            logger.info("Starting index historical data synchronization.")
            index_list = self.index_info_dao.load_index_info()
            self.index_list_size = len(index_list)
            logger.info("Found %d index to synchronize.", len(index_list))

            batch_size = 20
            for i in range(0, len(index_list), batch_size):
                batch = index_list[i : i + batch_size]
                for index in batch:
                    index_code = index.index_code
                    logger.info("Synchronizing index %s", index_code)

                    latest_date = self.index_hist_dao.get_latest_date(index_code)
                    if latest_date is None:
                        start_dt = datetime.date(1990, 1, 1)
                    else:
                        start_dt = latest_date + datetime.timedelta(days=1)

                    current_dt = datetime.date.today()
                    if start_dt > current_dt:
                        logger.info("Index %s is up to date. (start_date: %s, current_date: %s)", index_code, start_dt, current_dt)
                        continue

                    segment_days = 365 * 20
                    segment_start = start_dt
                    while segment_start <= current_dt:
                        segment_end = min(segment_start + datetime.timedelta(days=segment_days), current_dt)
                        try:
                            self.loop.create_task(
                                self.process_data(
                                    index_code,
                                    segment_start.strftime("%Y%m%d"),
                                    segment_end.strftime("%Y%m%d")
                                )
                            )
                        except Exception as e:
                            self.failed_indexes.append(index)
                            break
                        segment_start = segment_end + datetime.timedelta(days=1)

                    self.completed_num = self.completed_num + 1
                    if progress_callback:
                        progress_callback(self.completed_num, self.index_list_size)
        finally:
            if len(self.failed_indexes) > 0:
                failed_index_code = [index.index_code for index in self.failed_indexes]
                unique_failed_index_code = list(set(failed_index_code))
                logger.error(">>>>>>>>>Failed to process stocks<<<<<<<<<: %s", unique_failed_index_code)
            self.terminate()

    def sync_by_trade_date(self, start_date: str, end_date: str, progress_callback=None):
        """
        同步指定交易日的所有指数历史行情数据。
        :param start_date: 交易日期，格式为 'YYYYMMDD'
        :param end_date: 交易日期，格式为 'YYYYMMDD'
        """
        self.initialize()
        try:
            logger.info("Starting index historical data synchronization for %s to %s", start_date, end_date)
            index_list = self.index_info_dao.load_index_info()
            self.index_list_size = len(index_list)
            logger.info("Found %d indexes to process.", len(index_list))

            for index in index_list:
                index_code = index.index_code
                logger.info("Synchronizing index %s for %s to %s", index_code, start_date, end_date)
                try:
                    self.loop.run_until_complete(
                        self.process_data(
                            index_code,
                            start_date,
                            end_date
                        )
                    )
                except Exception as e:
                    self.failed_indexes.append(index)
                    break

                self.completed_num = self.completed_num + 1
                if progress_callback:
                    progress_callback(self.completed_num, self.index_list_size)

        finally:
            if self.failed_indexes:
                unique_failed_index_code = list(set(self.failed_indexes))
                logger.error(">>>>>>>>>Failed to process indexes<<<<<<<<<: %s", unique_failed_index_code)
            self.terminate()
        
index_info_synchronizer = IndexInfoSynchronizer()
index_hist_synchronizer = IndexHistSynchronizer()
