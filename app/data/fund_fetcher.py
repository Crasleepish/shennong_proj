from typing import List
from bisect import bisect_left
import akshare as ak
import pandas as pd
import datetime
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
        df = fund_list.rename(columns={'基金代码': 'fund_code', '基金名称': 'fund_name', '手续费': 'fee_rete'})

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
                    fee_rete=percent_to_float(safe_value(row['fee_rete']))
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
        self.fund_info_dao = FundInfoDao._instance
        self.fund_hist_dao = FundHistDao._instance
        self.loop = asyncio.get_event_loop()
        self.fund_list_size = 0
        self.completed_num = 0
        self.is_running = False
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(6)

    def initialize(self):
        self.fund_list_size = 0
        self.completed_num = 0
        self.is_running = True

    def terminate(self):
        self.fund_list_size = 0
        self.completed_num = 0
        self.is_running = False

    async def fetch_fund_data(self, fund_code):
        """异步获取历史行情数据"""
        try:
            async with self.semaphore:
                df_nav = await asyncio.to_thread(ak.fund_open_fund_info_em, 
                                            symbol=fund_code, 
                                            indicator="单位净值走势", 
                                            period="成立来")
                if not df_nav.empty:
                    df_nav = df_nav[['净值日期', '单位净值']]
                    df_nav['净值日期'] = pd.to_datetime(df_nav['净值日期'], errors='coerce')

                df_div = ak.fund_open_fund_info_em(symbol=fund_code, indicator="分红送配详情")
                if not df_div.empty:
                    df_div = df_div[['除息日', '每份分红']]
                    df_div['除息日'] = pd.to_datetime(df_div['除息日'], errors='coerce')

                df_split = ak.fund_open_fund_info_em(symbol=fund_code, indicator="拆分详情")
                if not df_split.empty:
                    df_split = df_split[['拆分折算日', '拆分折算比例']]
                    df_split['拆分折算日'] = pd.to_datetime(df_split['拆分折算日'], errors='coerce')

                # 按日期排序
                df_nav = df_nav.sort_values('净值日期').reset_index(drop=True)
                
                # 初始化复权因子列
                df_nav['复权因子'] = 1.0

                # 处理分红事件
                for idx, row in df_div.iterrows():
                    ex_div_date = row['除息日']
                    # 找出除息日前一日的最后一个交易日记录
                    df_before = df_nav[df_nav['净值日期'] < ex_div_date]
                    if df_before.empty:
                        continue  # 若没有前一交易日数据则跳过
                    last_record = df_before.iloc[-1]
                    nav_prev = last_record['单位净值']
                    
                    # 从分红描述中提取分红金额，例如 "每份派现金0.0650元"
                    dividend_match = re.findall(r"[\d\.]+", row['每份分红'])
                    if not dividend_match:
                        continue
                    dividend = float(dividend_match[0])
                    
                    # 计算分红复权因子
                    factor = nav_prev / (nav_prev - dividend)
                    
                    # 对除息日及之后的每个交易日净值调整复权因子
                    df_nav.loc[df_nav['净值日期'] >= ex_div_date, '复权因子'] *= factor

                # 处理拆分事件
                for idx, row in df_split.iterrows():
                    split_date = row['拆分折算日']
                    # 提取拆分比例，例如 "1:1.0238" 提取因子 1.0238
                    ratio_parts = row['拆分折算比例'].split(':')
                    if len(ratio_parts) != 2:
                        continue
                    factor = float(ratio_parts[1])
                    
                    # 对拆分折算日及之后的净值进行调整
                    df_nav.loc[df_nav['净值日期'] >= split_date, '复权因子'] *= factor

                # 计算复权净值
                df_nav['复权净值'] = df_nav['单位净值'] * df_nav['复权因子']


                return df_nav[['净值日期', '单位净值', '复权因子', '复权净值']]
        except Exception as e:
            logger.error("Error fetching fund data for %s: %s", fund_code, e)
            return pd.DataFrame()
        
    async def batch_insert(self, fund_code, new_records):
        def _syn_batch_insert(fund_code, new_records):
            # 5. 批量插入新增记录到数据库
            if new_records:
                try:
                    result_records = self.fund_hist_dao.batch_insert(new_records)
                    logger.info("Inserted %d new historical records for fund %s.", len(result_records), fund_code)
                except Exception as e:
                    logger.error("Error inserting new records for fund %s: %s", fund_code, e)
                    raise e
            else:
                logger.info("No new records to insert for fund %s.", fund_code)
        async with self.semaphore:
            return await asyncio.to_thread(_syn_batch_insert, fund_code, new_records)
    
    async def process_data(self, fund_code, start_date, current_date, progress_callback=None):
        try:
            # 3. 调用 akshare 接口获取该股票从 start_date 到 current_date 的历史行情数据（不复权）
            logger.info("Fetching historical data for fund %s from %s to %s", fund_code, start_date, current_date)
            df = await self.loop.create_task(self.fetch_fund_data(fund_code))

            if df.empty:
                logger.info("No new historical data for fund %s.", fund_code)
                return

            # 将日期列转换为日期类型
            if '净值日期' in df.columns:
                df['净值日期'] = pd.to_datetime(df['净值日期'], errors='coerce')
            
            df['收益率'] = df['单位净值'].pct_change()

            df = df[(df['净值日期'] >= start_date) & (df['净值日期'] <= current_date)]

            # 处理 DataFrame 数据，构造 FundHist 对象列表
            new_records = []
            for _, row in df.iterrows():
                record = FundHist(
                    fund_code=fund_code,
                    date=row["净值日期"].date(),
                    value=safe_value(row['单位净值']),
                    adjust_factor=safe_value(row['复权因子']),
                    net_value=safe_value(row["复权净值"]),
                    change_percent=safe_value(row["收益率"])
                )
                new_records.append(record)

            # 5. 批量插入新增记录到数据库
            await self.batch_insert(fund_code, new_records)
        except Exception as e:
            logger.error("Error processing fund %s: %s", fund_code, e)
        finally:
            async with self.lock:
                self.completed_num = self.completed_num + 1
                if progress_callback:
                    progress_callback(self.completed_num, self.fund_list_size)

    def sync(self, progress_callback=None):
        self.initialize()
        try:
            logger.info("Starting fund historical data synchronization.")
            # 1. 获取当前数据库中的股票列表
            fund_list = self.fund_info_dao.load_fund_info()
            self.fund_list_size = len(fund_list)
            logger.info("Found %d fund to synchronize.", len(fund_list))

            batch_size = 20
            for i in range(0, len(fund_list), batch_size):
                batch = fund_list[i : i + batch_size]
                tasks = []
                for fund in batch:
                    fund_code = fund.fund_code
                    logger.info("Synchronizing fund %s", fund_code)
                    # 2. 查询该股票在历史行情数据表中的最新日期
                    latest_date = self.fund_hist_dao.get_latest_date(fund_code)
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
                        logger.info("Fund %s is up to date. (start_date: %s, current_date: %s)", fund_code, start_date, current_date)
                        continue

                    tasks.append(self.loop.create_task(self.process_data(fund_code, start_date, current_date, progress_callback)))
                if tasks:
                    self.loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            self.terminate()
    
fund_info_synchronizer = FundInfoSynchronizer()
fund_hist_synchronizer = FundHistSynchronizer()
