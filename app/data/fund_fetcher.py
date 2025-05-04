from typing import List
from bisect import bisect_left
import akshare as ak
import tushare as ts
import pandas as pd
import datetime
import time
import math
import asyncio
from sqlalchemy.orm import Session
from app.models.fund_models import FundInfo, FundHist
from app.dao.fund_info_dao import FundInfoDao, FundHistDao
from app.constants.enums import TaskType, TaskStatus
from typing import Union
import re
from datetime import timedelta

import logging

logger = logging.getLogger(__name__)

tspro = ts.pro_api()

def safe_value(val):
    return None if pd.isna(val) else val

def safe_get(val):
    return 0.0 if val is None or pd.isna(val) else val

def percent_to_float(value):
    if isinstance(value, str) and '%' in value:
        # 去除百分号并转换为浮点数，然后除以100
        return float(value.strip('%')) / 100
    try:
        # 尝试直接转换为浮点数
        return float(value)
    except ValueError:
        # 如果转换失败，返回原值或根据需求处理
        return None

class FundInfoSynchronizer:
    """
    同步基金信息数据到数据库。
    
    主要流程：
      1. 调用 akshare 接口获取数据（pandas DataFrame）
      2. 查询数据库中已存在的指数代码集合
      3. 筛选出新增记录（基于指数代码）
      4. 将新增记录插入数据库（仅新增，不更新或删除）
    """
    def __init__(self):
        self.fund_info_dao = FundInfoDao._instance

    def fetch_data(self) -> pd.DataFrame:
        """
        调用 akshare 接口获取数据，返回 pandas DataFrame
        
        示例接口：
            import akshare as ak
            df = ak.fund_open_fund_daily_em()
        """
        logger.info("Fetching data from akshare ...")
        fund_list = ak.fund_info_index_em(symbol="全部", indicator="全部")[['基金代码', '基金名称', '手续费']]
        df = fund_list.rename(columns={'基金代码': 'fund_code', '基金名称': 'fund_name', '手续费': 'fee_rate'})
        df['fund_code'] = df['fund_code'] + '.OF'

        etf_list = tspro.fund_basic(market='E')[['ts_code', 'name']]
        etf_list = etf_list.rename(columns={'ts_code': 'fund_code', 'name': 'fund_name'})
        etf_list['fee_rate'] = 0.0

        df = pd.concat([df, etf_list], ignore_index=True)

        df.drop_duplicates(subset=['fund_code'], keep='last', inplace=True)
        return df

    def sync(self, progress_callback=None):
        """
        同步数据：将接口返回的新增记录插入到数据库中
        """
        logger.info("Starting synchronization for fund info.")
        df = self.fetch_data()
        if df.empty:
            logger.warning("Fetched data is empty.")
            return
        
        # 获取数据库中已有的股票代码集合
        try:
            fund_info_lst : List[FundInfo] = self.fund_info_dao.load_fund_info()
            existing_codes = {si.fund_code for si in fund_info_lst}
            logger.debug("Existing fund codes in DB: %s", existing_codes)
            
            # 筛选出新增数据（证券代码不在 existing_codes 中）
            new_data = df[~df['fund_code'].isin(existing_codes)]
            logger.info("Found %d new records to insert.", len(new_data))
            
            if new_data.empty:
                logger.info("No new records to insert.")
                return
            
            # 将 DataFrame 中每一行转换为 FundInfo 对象
            new_records = []
            for idx, row in new_data.iterrows():
                record = FundInfo(
                    fund_code=row['fund_code'],
                    fund_name=safe_value(row['fund_name']),
                    fee_rate=percent_to_float(safe_value(row['fee_rate']))
                )
                new_records.append(record)
                if progress_callback:
                    if idx % 100 == 0:
                        progress_callback(idx, len(new_records))
            
            # 批量插入新增记录
            self.fund_info_dao.batch_insert(new_records)

            logger.info("Inserted %d new records into the database.", len(new_records))
            if progress_callback:
                progress_callback(len(new_records), len(new_records))
        except Exception as e:
            logger.exception("Error during synchronization: %s", str(e))
            raise e

class FundHistSynchronizer:
    def __init__(self):
        self.fund_info_dao = FundInfoDao._instance
        self.fund_hist_dao = FundHistDao._instance
        self.loop = asyncio.get_event_loop()
        self.fund_list_size = 0
        self.completed_num = 0
        self.is_running = False
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(6)
        self.api_call_counter = 0
        self.api_call_reset_time = time.time() + 60

    def initialize(self):
        self.fund_list_size = 0
        self.completed_num = 0
        self.is_running = True
        self.api_call_counter = 0
        self.api_call_reset_time = time.time() + 60

    def terminate(self):
        self.fund_list_size = 0
        self.completed_num = 0
        self.is_running = False

    async def _check_api_rate_limit(self):
        """检查并控制API调用频率，确保每分钟不超过400次"""
        current_time = time.time()
        
        if current_time > self.api_call_reset_time:
            # Reset counter if a minute has passed
            self.api_call_counter = 0
            self.api_call_reset_time = current_time + 60

        if self.api_call_counter >= 400:
            wait_time = self.api_call_reset_time - current_time
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.api_call_counter = 0
                self.api_call_reset_time = time.time() + 60

        self.api_call_counter += 1

    async def fetch_fund_data(self, fund_code, start_date, end_date):
        try:
            async with self.semaphore:
                await self._check_api_rate_limit()
                df = await asyncio.to_thread(
                    tspro.fund_nav,
                    ts_code=fund_code,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d")
                )
                if not df.empty:
                    df['nav_date'] = pd.to_datetime(df['nav_date'], errors='coerce')
                    # 去重：保留同一 ts_code + nav_date 中最后一条记录
                    df = df.drop_duplicates(subset=['ts_code', 'nav_date'], keep='last')
                    df = df.sort_values('nav_date').reset_index(drop=True)
                    df['pct_change'] = df['unit_nav'].pct_change()
                    return df[['nav_date', 'unit_nav', 'adj_nav', 'pct_change']]
                return pd.DataFrame()
        except Exception as e:
            logger.error("Error fetching fund data for %s: %s", fund_code, e)
            return pd.DataFrame()

    async def batch_insert(self, fund_code, new_records):
        async with self.semaphore:
            return await asyncio.to_thread(self.fund_hist_dao.batch_upsert, new_records)

    async def process_data(self, fund_code, start_date, end_date):
        try:
            logger.info("Fetching historical data for fund %s from %s to %s", fund_code, start_date, end_date)
            df = await self.loop.create_task(self.fetch_fund_data(fund_code, start_date, end_date))

            if df.empty:
                logger.info("No new historical data for fund %s.", fund_code)
                return

            new_records = []
            for _, row in df.iterrows():
                record = FundHist(
                    fund_code=fund_code,
                    date=row["nav_date"].date(),
                    value=safe_value(row['unit_nav']),
                    net_value=safe_value(row['adj_nav']),
                    change_percent=safe_value(row['pct_change'])
                )
                new_records.append(record)

            await self.batch_insert(fund_code, new_records)
        except Exception as e:
            logger.error("Error processing fund %s: %s", fund_code, e)

    def sync(self, progress_callback=None):
        self.initialize()
        try:
            logger.info("Starting fund historical data synchronization.")
            fund_list = self.fund_info_dao.load_fund_info()
            self.fund_list_size = len(fund_list)
            logger.info("Found %d fund to synchronize.", len(fund_list))

            batch_size = 20
            for i in range(0, len(fund_list), batch_size):
                batch = fund_list[i: i + batch_size]
                for fund in batch:
                    fund_code = fund.fund_code
                    logger.info("Synchronizing fund %s", fund_code)
                    latest_date = self.fund_hist_dao.get_latest_date(fund_code)

                    if latest_date is None:
                        start_date_dt = datetime.date(1990, 1, 1)
                    else:
                        start_date_dt = latest_date + datetime.timedelta(days=1)

                    current_date_dt = datetime.date.today()
                    if start_date_dt > current_date_dt:
                        logger.info("Fund %s is up to date. (start_date: %s, current_date: %s)", fund_code, start_date_dt, current_date_dt)
                        continue

                    # 每段20年，串行处理每一段时间
                    segment_days = 365 * 20
                    segment_start = start_date_dt
                    while segment_start <= current_date_dt:
                        segment_end = min(segment_start + datetime.timedelta(days=segment_days), current_date_dt)
                        self.loop.run_until_complete(
                            self.process_data(fund_code, segment_start, segment_end)
                        )
                        segment_start = segment_end + datetime.timedelta(days=1)

                    self.completed_num += 1
                    if progress_callback:
                        progress_callback(self.completed_num, self.fund_list_size)
        finally:
            self.terminate()

    
fund_info_synchronizer = FundInfoSynchronizer()
fund_hist_synchronizer = FundHistSynchronizer()
