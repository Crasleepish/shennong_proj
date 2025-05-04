from typing import List
from bisect import bisect_left
import tushare as ts
import akshare as ak
import pandas as pd
import datetime
import math
import asyncio
from sqlalchemy.orm import Session
from app.models.stock_models import StockInfo, StockHistUnadj, UpdateFlag, CompanyAction, AdjFactor, FutureTask, StockHistAdj, FundamentalData, SuspendData
from app.dao.stock_info_dao import StockInfoDao, StockHistUnadjDao, UpdateFlagDao, CompanyActionDao, AdjFactorDao, FutureTaskDao, StockHistAdjDao, FundamentalDataDao, SuspendDataDao, StockShareChangeCNInfoDao
from app.constants.enums import TaskType, TaskStatus
from typing import Union
from .custom_api import fetch_stock_equity_changes
import re
import time

import logging

logger = logging.getLogger(__name__)

tspro = ts.pro_api()

def safe_value(val):
    return None if pd.isna(val) else val

def safe_get(val):
    return 0.0 if val is None or pd.isna(val) else val

class StockInfoSynchronizer:
    """
    同步股票信息数据到数据库。
    
    主要流程：
      1. 调用 akshare 接口获取数据（pandas DataFrame）
      2. 查询数据库中已存在的股票代码集合
      3. 筛选出新增记录（基于证券代码）
      4. 将新增记录插入数据库（仅新增，不更新或删除）
    """
    def __init__(self):
        self.stock_info_dao = StockInfoDao._instance
        self.update_flag_dao = UpdateFlagDao._instance
        self.future_task_dao = FutureTaskDao._instance

    def fetch_data(self) -> pd.DataFrame:
        """
        调用 akshare 接口获取数据，返回 pandas DataFrame
        
        示例接口：
            import akshare as ak
            df = ak.stock_info_sh_name_code(symbol="主板A股")
        """
        logger.info("Fetching data from akshare ...")
        stock_list = tspro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,market,exchange,list_date,list_status')
        stock_list = stock_list.rename(columns={'ts_code': 'stock_code', 'name': 'stock_name', 'market': 'market', 'exchange': 'exchange', 'list_date': 'listing_date', 'list_status': 'list_status'})
        # 将上市日期转换为日期类型（若转换失败则置为 NaT）
        stock_list['listing_date'] = pd.to_datetime(stock_list['listing_date'], errors='coerce').dt.date

        # 退市股票列表
        stock_ts_list = tspro.stock_basic(exchange='', list_status='D', fields='ts_code,symbol,name,market,exchange,list_date,list_status')
        stock_ts_list = stock_ts_list.rename(columns={'ts_code': 'stock_code', 'name': 'stock_name', 'market': 'market', 'exchange': 'exchange', 'list_date': 'listing_date', 'list_status': 'list_status'})
        stock_ts_list['listing_date'] = pd.to_datetime(stock_ts_list['listing_date'], errors='coerce').dt.date

        df = pd.concat([stock_list, stock_ts_list], ignore_index=True).sort_values('listing_date').reset_index(drop=True)
        df.drop_duplicates(subset=['stock_code'], keep='last', inplace=True)
        return df
    
    def update_industry(self):
        """
        基于申万行业分类，将行业信息写入 stock_info 表
        """
        logger.info("开始更新行业信息（申万2021）")
        try:
            industry_df = tspro.index_classify(level='L1', src='SW2021')
            logger.info(f"获取到 {len(industry_df)} 个一级行业")

            for _, row in industry_df.iterrows():
                index_code = row['index_code']
                industry_name = row['industry_name']
                try:
                    members_df = tspro.index_member_all(l1_code=index_code, fields='ts_code')
                    stock_codes = members_df['ts_code'].dropna().tolist()
                    self.stock_info_dao.update_industry_by_mapping(stock_codes, industry_name)
                except Exception as e:
                    logger.warning(f"[跳过] 获取行业 {industry_name} 成分股失败: {e}")
            
            logger.info("行业更新完成")
        except Exception as e:
            logger.error("行业信息更新失败: %s", e)
            raise e

    def sync(self, progress_callback=None):
        """
        同步数据：将接口返回的新增记录插入到数据库中
        """
        logger.info("Starting synchronization for stock info.")
        new_data = self.fetch_data()
        if new_data.empty:
            logger.warning("Fetched data is empty.")
            return
        
        try:
            logger.info("Found %d new records to insert/update.", len(new_data))
            
            # 将 DataFrame 中每一行转换为 StockInfo 对象
            new_records = []
            for idx, row in new_data.iterrows():
                record = StockInfo(
                    stock_code=safe_value(row['stock_code']),
                    stock_name=safe_value(row['stock_name']),
                    market=safe_value(row['market']),
                    exchange=safe_value(row['exchange']),
                    listing_date=safe_value(row['listing_date']),
                    industry=None,
                    list_status=safe_value(row['list_status']),
                )
                new_records.append(record)
                if progress_callback:
                    if idx % 100 == 0:
                        progress_callback(idx, len(new_records))
            
            # 批量插入新增记录
            self.stock_info_dao.batch_upsert(new_records)

            # 更新行业信息
            self.update_industry()

            # 将新数据插入到update_flag表中
            for idx, record in enumerate(new_records, start=1):
                self.update_flag_dao.upsert_one(UpdateFlag(stock_code=record.stock_code, action_update_flag='1', fundamental_update_flag='1'))
                
            logger.info("Inserted %d new records into the database.", len(new_records))
            if progress_callback:
                progress_callback(len(new_records), len(new_records))
        except Exception as e:
            logger.exception("Error during synchronization: %s", str(e))
            raise e

class StockHistSynchronizer:
    """
    同步股票历史行情数据到数据库。
    
    主要流程：
      1. 查看当前数据库中的股票列表
      2. 遍历股票列表，依次同步每只股票的历史行情数据
      3. 对于某一支股票，查看其最新数据的日期
      4. 如果最新日期小于等于当前日期，则同步该支股票的历史行情数据（不复权），调用tushare接口（tspro.daily）获取最新数据次日到当前日期的历史行情数据
      5. 如果时间跨度大于20年，则分割为多个请求，每次请求20年数据
      6. 将新增的数据插入数据库
      7. 限制接口调用次数每分钟不超过400次
    """
    def __init__(self):
        self.tspro = tspro  # Tushare API client
        self.stock_info_dao = StockInfoDao._instance
        self.stock_hist_unadj_dao = StockHistUnadjDao._instance
        self.update_flag_dao = UpdateFlagDao._instance
        self.loop = asyncio.get_event_loop()
        self.stock_list_size = 0
        self.completed_num = 0
        self.is_running = False
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(6)
        # Track API calls to comply with Tushare's rate limiting (400 calls per minute)
        self.api_call_counter = 0
        self.api_call_reset_time = time.time() + 60

    def initialize(self):
        self.stock_list_size = 0
        self.completed_num = 0
        self.is_running = True
        self.api_call_counter = 0
        self.api_call_reset_time = time.time() + 60

    def terminate(self):
        self.stock_list_size = 0
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
            # Wait until the next minute if we've reached the limit
            wait_time = self.api_call_reset_time - current_time
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                self.api_call_counter = 0
                self.api_call_reset_time = time.time() + 60
        
        self.api_call_counter += 1

    async def fetch_stock_data(self, stock_code, market, start_date, end_date):
        """异步获取股票历史行情数据"""
        try:
            async with self.semaphore:
                # Check API rate limit before making the call
                await self._check_api_rate_limit()
                
                # Convert stock_code to ts_code format (code.MARKET)
                ts_code = stock_code
                
                # Call tushare daily API
                df = await asyncio.to_thread(
                    self.tspro.daily,
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Transform tushare data format to match the expected format
                if not df.empty:
                    # Rename columns to match the original format
                    df = df.rename(columns={
                        'ts_code': 'ts_code',
                        'trade_date': '日期',
                        'open': '开盘',
                        'high': '最高',
                        'low': '最低',
                        'close': '收盘',
                        'pre_close': '前收盘',
                        'vol': '成交量',
                        'amount': '成交额',
                        'pct_chg': '涨跌幅',
                        'change': '涨跌额'
                    })
                    
                    # Convert trade_date from string to datetime
                    df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d')
                    
                    # Add stock code column
                    df['股票代码'] = stock_code
                    
                return df
        except Exception as e:
            logger.error("Error fetching stock data for %s: %s", stock_code, e)
            return pd.DataFrame()
        
    async def fetch_daily_basic_data(self, stock_code, market, start_date, end_date):
        """异步获取股票每日基本面数据"""
        try:
            async with self.semaphore:
                # Check API rate limit before making the call
                await self._check_api_rate_limit()
                
                # Convert stock_code to ts_code format (code.MARKET)
                ts_code = f"{stock_code}"
                
                # Define fields to fetch
                fields = 'ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv'
                
                # Call tushare daily_basic API
                df = await asyncio.to_thread(
                    self.tspro.daily_basic,
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields=fields
                )
                
                if not df.empty:
                    # Convert trade_date from string to datetime
                    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                    
                return df
        except Exception as e:
            logger.error("Error fetching daily basic data for %s: %s", stock_code, e)
            return pd.DataFrame()
        
    def merge_data(self, daily_df, daily_basic_df):
        """合并股票日线行情和基本面数据"""
        if daily_df.empty:
            return pd.DataFrame()
            
        if not daily_basic_df.empty:
            # Rename columns for merging
            daily_basic_df = daily_basic_df.rename(columns={'trade_date': '日期'})
            
            # Merge dataframes on date and ts_code
            merged_df = pd.merge(
                daily_df,
                daily_basic_df,
                how='left',
                left_on=['日期', 'ts_code'],
                right_on=['日期', 'ts_code']
            )
            
            # Add market cap calculation if not available from daily_basic
            if 'total_mv' in merged_df.columns:
                merged_df['mkt_cap'] = merged_df['total_mv'] * 10000  # Convert from 万元 to 元
            elif 'total_share' in merged_df.columns and '收盘' in merged_df.columns:
                merged_df['mkt_cap'] = merged_df['收盘'] * merged_df['total_share'] * 10000  # Convert from 万股 to 股
                
            # Convert share values from 万股 to 股
            if 'total_share' in merged_df.columns:
                merged_df['total_shares'] = merged_df['total_share'] * 10000
            if 'float_share' in merged_df.columns:
                merged_df['float_shares'] = merged_df['float_share'] * 10000
            if 'free_share' in merged_df.columns:
                merged_df['free_shares'] = merged_df['free_share'] * 10000
                
            return merged_df
        else:
            # If no daily_basic data available, just add empty columns
            daily_df['turnover_rate'] = None
            daily_df['turnover_rate_f'] = None
            daily_df['volume_ratio'] = None
            daily_df['pe'] = None
            daily_df['pe_ttm'] = None
            daily_df['pb'] = None
            daily_df['ps'] = None
            daily_df['ps_ttm'] = None
            daily_df['dv_ratio'] = None
            daily_df['dv_ttm'] = None
            daily_df['total_shares'] = None
            daily_df['float_shares'] = None
            daily_df['free_shares'] = None
            daily_df['mkt_cap'] = None
            daily_df['circ_mv'] = None
            
            return daily_df
        
    async def batch_insert_unadj(self, stock_code, new_records):
        def _syn_batch_insert_unadj(stock_code, new_records):
            # 5. 批量插入新增记录到数据库
            if new_records:
                try:
                    result_records = self.stock_hist_unadj_dao.batch_insert(new_records)
                    logger.info("Inserted %d new historical records for stock %s.", len(result_records), stock_code)
                    self.update_flag_dao.update_action_flag(stock_code, "1")
                except Exception as e:
                    logger.error("Error inserting new records for stock %s: %s", stock_code, e)
                    raise e
            else:
                logger.info("No new records to insert for stock %s.", stock_code)
        async with self.semaphore:
            return await asyncio.to_thread(_syn_batch_insert_unadj, stock_code, new_records)
        
    async def process_data_unadj(self, stock_code, market, start_date, current_date, progress_callback=None):
        try:
            # 3. 调用 tushare 接口获取该股票从 start_date 到 current_date 的历史行情数据（不复权）和基本面数据
            logger.info("Fetching historical data for stock %s from %s to %s", stock_code, start_date, current_date)
            daily_task = self.loop.create_task(self.fetch_stock_data(stock_code, market, start_date, current_date))
            daily_basic_task = self.loop.create_task(self.fetch_daily_basic_data(stock_code, market, start_date, current_date))
            daily_df, daily_basic_df = await asyncio.gather(daily_task, daily_basic_task)

            if daily_df.empty:
                logger.info("No new historical data for stock %s.", stock_code)
                return

            # 合并日线数据和基本面数据
            df = self.merge_data(daily_df, daily_basic_df)

            # 4. 处理 DataFrame 数据，构造 StockHist 对象列表
            new_records = []
            for _, row in df.iterrows():
                record = StockHistUnadj(
                    date=row["日期"].date(),
                    stock_code=row["股票代码"],
                    open=safe_value(row["开盘"]),
                    close=safe_value(row["收盘"]),
                    high=safe_value(row["最高"]),
                    low=safe_value(row["最低"]),
                    volume=safe_value(row["成交量"]) * 100,  # Tushare volume is in 100s of shares
                    amount=safe_value(row["成交额"]),
                    change_percent=safe_value(row["涨跌幅"]),
                    change=safe_value(row["涨跌额"]),
                    turnover_rate=safe_value(row.get("turnover_rate")),
                    # Add additional fields from daily_basic
                    turnover_rate_f=safe_value(row.get("turnover_rate_f")),
                    volume_ratio=safe_value(row.get("volume_ratio")),
                    pe=safe_value(row.get("pe")),
                    pe_ttm=safe_value(row.get("pe_ttm")),
                    pb=safe_value(row.get("pb")),
                    ps=safe_value(row.get("ps")),
                    ps_ttm=safe_value(row.get("ps_ttm")),
                    dv_ratio=safe_value(row.get("dv_ratio")),
                    dv_ttm=safe_value(row.get("dv_ttm")),
                    total_shares=safe_value(row.get("total_shares")),
                    float_shares=safe_value(row.get("float_shares")),
                    free_shares=safe_value(row.get("free_shares")),
                    mkt_cap=safe_value(row.get("mkt_cap")),
                    circ_mv=safe_value(row.get("circ_mv")) * 10000 if row.get("circ_mv") is not None else None,  # Convert from 万元 to 元
                    # Include pre_close if available
                    pre_close=safe_value(row.get("前收盘"))
                )
                new_records.append(record)

            # 5. 批量插入新增记录到数据库
            await self.batch_insert_unadj(stock_code, new_records)
        except Exception as e:
            logger.error("Error processing stock %s: %s", stock_code, e)
        finally:
            async with self.lock:
                self.completed_num = self.completed_num + 1
                if progress_callback:
                    progress_callback(self.completed_num, self.stock_list_size)

    def sync(self, progress_callback=None):
        self.initialize()
        try:
            logger.info("Starting stock historical data synchronization.")
            stock_list = self.stock_info_dao.load_stock_info()
            self.stock_list_size = len(stock_list)
            logger.info("Found %d stocks to synchronize.", len(stock_list))

            batch_size = 20
            for i in range(0, len(stock_list), batch_size):
                batch = stock_list[i : i + batch_size]
                tasks = []
                for stock in batch:
                    stock_code = stock.stock_code
                    market = stock.market
                    logger.info("Synchronizing stock %s", stock_code)

                    latest_date = self.stock_hist_unadj_dao.get_latest_date(stock_code)
                    if latest_date is None:
                        start_date_dt = datetime.date(1990, 1, 1)
                    else:
                        start_date_dt = latest_date + datetime.timedelta(days=1)

                    current_date_dt = datetime.date.today()
                    if start_date_dt > current_date_dt:
                        logger.info("Stock %s is up to date. (start_date: %s, current_date: %s)", stock_code, start_date_dt, current_date_dt)
                        continue

                    # 每段最多20年（约7300天）
                    segment_days = 365 * 20
                    segment_start = start_date_dt
                    while segment_start < current_date_dt:
                        segment_end = min(segment_start + datetime.timedelta(days=segment_days), current_date_dt)
                        tasks.append(
                            self.loop.create_task(
                                self.process_data_unadj(
                                    stock_code, market,
                                    segment_start.strftime("%Y%m%d"),
                                    segment_end.strftime("%Y%m%d"),
                                    progress_callback
                                )
                            )
                        )
                        segment_start = segment_end + datetime.timedelta(days=1)

                if tasks:
                    self.loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            self.terminate()

    def sync_by_trade_date(self, trade_date: str):
        """
        同步指定交易日 trade_date 当天所有股票的日行情数据 + 每日指标数据到数据库。
        trade_date: 格式 'YYYYMMDD' （例如 '20240830'）
        """
        try:
            logger.info(f"Fetching daily + daily_basic data for all stocks on {trade_date}")
            
            # 拉取daily行情数据
            daily_df = tspro.daily(trade_date=trade_date)
            
            if daily_df.empty:
                logger.warning(f"No daily data found for trade_date {trade_date}")
                return
            
            # 拉取daily_basic指标数据
            fields = 'ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv'
            daily_basic_df = tspro.daily_basic(ts_code='', trade_date=trade_date, fields=fields)
            
            if daily_basic_df.empty:
                logger.warning(f"No daily_basic data found for trade_date {trade_date}")

            # 合并行情和指标数据
            merged_df = pd.merge(
                daily_df,
                daily_basic_df,
                on=['ts_code', 'trade_date'],
                how='left'
            )

            new_records = []

            # 遍历DataFrame生成 StockHistUnadj 对象
            for _, row in merged_df.iterrows():
                try:
                    record = StockHistUnadj(
                        stock_code=safe_value(row['ts_code']),
                        date=pd.to_datetime(row['trade_date']).date(),
                        open=safe_value(row.get('open')),
                        close=safe_value(row.get('close')),
                        high=safe_value(row.get('high')),
                        low=safe_value(row.get('low')),
                        volume=safe_value(row.get('vol')) * 100 if row.get('vol') is not None else None,
                        amount=safe_value(row.get('amount')),
                        pre_close=safe_value(row.get('pre_close')),
                        change_percent=safe_value(row.get('pct_chg')),
                        change=safe_value(row.get('change')),
                        turnover_rate=safe_value(row.get('turnover_rate')),
                        turnover_rate_f=safe_value(row.get('turnover_rate_f')),
                        volume_ratio=safe_value(row.get('volume_ratio')),
                        pe=safe_value(row.get('pe')),
                        pe_ttm=safe_value(row.get('pe_ttm')),
                        pb=safe_value(row.get('pb')),
                        ps=safe_value(row.get('ps')),
                        ps_ttm=safe_value(row.get('ps_ttm')),
                        dv_ratio=safe_value(row.get('dv_ratio')),
                        dv_ttm=safe_value(row.get('dv_ttm')),
                        total_shares=safe_value(row.get('total_share')) * 10000 if row.get('total_share') is not None else None,  # 单位万股转股
                        float_shares=safe_value(row.get('float_share')) * 10000 if row.get('float_share') is not None else None,
                        free_shares=safe_value(row.get('free_share')) * 10000 if row.get('free_share') is not None else None,
                        mkt_cap=safe_value(row.get('total_mv')) * 10000 if row.get('total_mv') is not None else None,  # 单位万元转元
                        circ_mv=safe_value(row.get('circ_mv')) * 10000 if row.get('circ_mv') is not None else None
                    )
                    new_records.append(record)
                except Exception as e:
                    logger.error(f"Error processing row {row.to_dict()}: {e}")

            if new_records:
                self.stock_hist_unadj_dao.batch_upsert(new_records)
                logger.info(f"Upserted {len(new_records)} records for trade_date {trade_date}")
            else:
                logger.info(f"No valid records to upsert for trade_date {trade_date}")

        except Exception as e:
            logger.error(f"Error syncing daily + daily_basic data for {trade_date}: {e}")
            
class StockAdjHistSynchronizer:
    """
        同步前复权历史行情数据。
        
        主要流程：
        1. 判断是否存在 UPDATE_ADJ 任务：通过 FutureTaskDao.select_by_code_date_type(stock_code, today, "UPDATE_ADJ")
            如果存在且状态为 INIT，则更新该股票的所有前复权数据（先删除，再重新计算）；否则，直接复制最新的无复权数据到前复权数据表。
        2. 对于更新情形，更新步骤：
            a. 获取该股票所有的无复权历史数据（stock_hist_unadj），按日期升序排列。
            b. 获取该股票所有的公司行动记录（company_action），按 ex_dividend_date 升序排列。
            c. 对于无复权数据中每个交易日 i，查找所有 ex_date 大于 i 的公司行动记录，按顺序计算复权因子 F_j，
                其中每个 F_j 的计算依赖于事件前一交易日的无复权收盘价 P_ref：
                    F_j = (P_ref * (1 + bonus + conversion + rights_issue)) / (P_ref - dividend + (rights_issue * rights_issue_price))
                注意：bonus, conversion, rights_issue, dividend 均为“每股”数据（原接口数据除以10）。
            d. 将无复权数据中的开盘、收盘、最高、最低乘以累计复权因子，生成前复权数据记录。
            e. 批量插入前复权数据到 stock_hist_adj 表，并更新 FutureTask 状态为 DONE。
        3. 如果没有 UPDATE_ADJ 任务，则直接复制最新的无复权数据到 stock_hist_adj 表。
        """

    def __init__(self):
        self.stock_info_dao = StockInfoDao._instance
        self.stock_hist_unadj_dao = StockHistUnadjDao._instance
        self.update_flag_dao = UpdateFlagDao._instance
        self.company_action_dao = CompanyActionDao._instance
        self.stock_hist_adj_dao = StockHistAdjDao()
        self.future_task_dao = FutureTaskDao._instance
        self.stock_share_change_dao = StockShareChangeCNInfoDao._instance
        self.loop = asyncio.get_event_loop()
        self.stock_list_size = 0
        self.completed_num = 0
        self.is_running = False
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(6)

    def initialize(self):
        self.stock_list_size = 0
        self.completed_num = 0
        self.is_running = True

    def terminate(self):
        self.stock_list_size = 0
        self.completed_num = 0
        self.is_running = False

    async def process_data_adj(self, stock_code: str, progress_callback=None):
        try:
            async with self.semaphore:
                today = datetime.date.today()
                # 检查未来任务是否存在 UPDATE_ADJ 任务，状态为 INIT
                future_tasks = await asyncio.to_thread(self.future_task_dao.select_by_code_date_type, stock_code, today, TaskType.UPDATE_ADJ.name, "INIT")
                if future_tasks is not None and len(future_tasks) != 0:
                    logger.info("Found UPDATE_ADJ task for %s, updating forward-adjusted history.", stock_code)
                    
                    # 获取所有无复权数据，按日期升序排列
                    df_unadj = await asyncio.to_thread(self.stock_hist_unadj_dao.select_dataframe_by_code, stock_code)
                    if df_unadj is None or df_unadj.empty:
                        logger.info("No non-adjusted data for stock %s, skipping.", stock_code)
                        return
                    # 遍历无复权记录，确保按交易日期升序排列
                    df_unadj = df_unadj.sort_values(by="date")
                    earliest_date = df_unadj.iloc[0]['date']
                    # 获取所有公司行动记录（已转换为每股数据），按 ex_dividend_date 升序排列
                    df_action = await asyncio.to_thread(self.company_action_dao.select_dataframe_by_code, stock_code)
                    if df_action is None or df_action.empty:
                        logger.info("No company action data for stock %s.", stock_code)
                        await self.fetch_data_from_unadj_data(stock_code)
                        return
                    #仅过滤有意义的行动数据
                    df_action = df_action[df_action['ex_dividend_date'] >= earliest_date]
                    # 获取交易日期列表和对应的收盘价列表
                    trade_dates = df_unadj["date"].tolist()
                    close_prices = df_unadj["close"].tolist()
                    # 预处理公司行动数据
                    df_action = df_action.sort_values(by="ex_dividend_date")
                    # 转换为字典列表，方便后续处理
                    actions = df_action.to_dict(orient="records")
                    # 预先计算参考价格字典：对于每个 action，参考价格为无复权数据中最后一个交易日期 < action.ex_dividend_date 的收盘价
                    ref_prices = {}
                    for action in actions:
                        ex_date = action["ex_dividend_date"]
                        idx = bisect_left(trade_dates, ex_date)
                        if idx == 0:
                            ref_prices[ex_date] = None
                        else:
                            ref_prices[ex_date] = close_prices[idx - 1]

                    # 预先计算每个 action 的单独复权因子 F_j
                    action_factors = {}
                    for action in actions:
                        ex_date = action["ex_dividend_date"]
                        P_ref = ref_prices.get(ex_date)
                        if not P_ref or P_ref == 0:
                            F = 1.0
                        else:
                            bonus = safe_get(action.get("bonus_ratio")) or 0.0
                            conversion = safe_get(action.get("conversion_ratio")) or 0.0
                            rights_issue = safe_get(action.get("rights_issue_ratio")) or 0.0
                            dividend = safe_get(action.get("dividend_per_share")) or 0.0
                            rights_issue_price = safe_get(action.get("rights_issue_price")) or 0.0 
                            try:
                                F = (P_ref - dividend + (rights_issue * rights_issue_price)) / (P_ref * (1 + bonus + conversion + rights_issue))
                            except Exception as e:
                                logger.error("Error computing factor for stock %s on action date %s: %s", stock_code, ex_date, e)
                                F = 1.0
                        action_factors[ex_date] = F

                    # 计算累计复权因子。先取出所有 action 的 ex_dividend_date 并排序（升序）
                    sorted_action_dates = sorted(action_factors.keys())
                    cum_factors = {}
                    cum = 1.0
                    # 从最晚的日期开始反向累乘
                    for ex_date in reversed(sorted_action_dates):
                        cum *= action_factors[ex_date]
                        cum_factors[ex_date] = cum

                    # 定义一个辅助函数，根据交易日查找累计复权因子
                    def get_cum_factor_for_trade(trade_date):
                        """
                        返回所有 ex_dividend_date 大于 trade_date 的累计复权因子，
                        如果没有，则返回 1.0。
                        """
                        # 在 sorted_action_dates 中查找第一个严格大于 trade_date 的日期
                        idx = bisect_left(sorted_action_dates, trade_date)
                        # 如果找到的日期不大于 trade_date，则继续移动指针
                        while idx < len(sorted_action_dates) and sorted_action_dates[idx] <= trade_date:
                            idx += 1
                        if idx < len(sorted_action_dates):
                            return cum_factors[sorted_action_dates[idx]]
                        else:
                            return 1.0

                    # 计算前复权数据：对每一条无复权记录，累计后续所有复权因子
                    adj_records = []
                    # 用于保存前一交易日的前复权收盘价，用于计算涨跌相关指标
                    previous_adj_close = None
                    # 对每条记录
                    for _, row in df_unadj.iterrows():
                        trade_date = row["date"]
                        # 查找累计复权因子
                        cumulative_factor = get_cum_factor_for_trade(trade_date)
                        # 计算前复权价格
                        row_open = safe_value(row["open"])
                        row_close = safe_value(row["close"])
                        row_high = safe_value(row["high"])
                        row_low = safe_value(row["low"])
                        row_volume = safe_value(row["volume"])
                        adj_open = row_open * cumulative_factor if row_open is not None else None
                        adj_close = row_close * cumulative_factor if row_close is not None else None
                        adj_high = row_high * cumulative_factor if row_high is not None else None
                        adj_low = row_low * cumulative_factor if row_low is not None else None
                        adj_volume = row_volume / cumulative_factor if row_volume is not None else None
                        adj_amount = adj_volume * adj_close if adj_volume is not None and adj_close is not None else None
                        # 重新计算涨跌指标：若存在前一交易日的前复权收盘价，则计算差值和百分比变化
                        if previous_adj_close is None:
                            adj_change = None
                            adj_change_percent = None
                            adj_amplitude = None
                        else:
                            adj_change = adj_close - previous_adj_close
                            # 避免除零错误
                            if previous_adj_close != 0:
                                adj_change_percent = (adj_change / previous_adj_close) * 100
                                adj_amplitude = ((adj_high - adj_low) / previous_adj_close) * 100
                            else:
                                adj_change_percent = None
                                adj_amplitude = None

                        # 更新 previous_adj_close 为当前前复权收盘价，供下一条记录计算使用
                        previous_adj_close = adj_close

                        # 构造前复权数据记录，其他字段直接复制无复权记录
                        record = StockHistAdj(
                            stock_code = stock_code,
                            date = trade_date,
                            open = adj_open,
                            close = adj_close,
                            high = adj_high,
                            low = adj_low,
                            volume = adj_volume,
                            amount = adj_amount,
                            amplitude = adj_amplitude,
                            change_percent = adj_change_percent,
                            change = adj_change,
                            turnover_rate = safe_value(row.get("turnover_rate")),
                            mkt_cap = safe_value(row.get("mkt_cap")),
                            total_shares = safe_value(row.get("total_shares"))
                        )
                        adj_records.append(record)
                    # 批量插入前复权数据
                    if adj_records:
                        # 更新模式：删除该股票所有前复权数据
                        await asyncio.to_thread(self.stock_hist_adj_dao.delete_by_stock_code, stock_code)
                        await asyncio.to_thread(self.stock_hist_adj_dao.batch_insert, adj_records)
                        logger.info("Updated forward-adjusted data for stock %s: %d records inserted.", stock_code, len(adj_records))
                    else:
                        logger.info("No forward-adjusted records computed for stock %s.", stock_code)
                    # 更新未来任务状态为 DONE
                    for future_task in future_tasks:
                        self.future_task_dao.update_status_by_id(future_task.task_id, "DONE")
                    # 一支股票只执行一次UPDATE_ADJ任务，会完成所有日期数据的复权计算
                else:
                    await self.fetch_data_from_unadj_data(stock_code)
        except Exception as e:
            logger.error("Error executing stock hist adj task for %s: %s", stock_code, e)
            return
        finally:
            async with self.lock:
                self.completed_num = self.completed_num + 1
                if progress_callback:
                    progress_callback(self.completed_num, self.stock_list_size)
                
    async def fetch_data_from_unadj_data(self, stock_code):
        # 无更新任务时：直接复制无复权数据表中，日期在前复权数据表最新日期之后的所有数据到前复权数据表
        latest_adj_date = await asyncio.to_thread(self.stock_hist_adj_dao.get_latest_date, stock_code)
        if latest_adj_date is None:
            # 如果前复权数据表无记录，则设置为极早日期
            latest_adj_date = datetime.date(1900, 1, 1)
        # 查询无复权数据中日期大于最新前复权日期的所有记录
        df_new = await asyncio.to_thread(self.stock_hist_unadj_dao.select_after_date_as_dataframe, stock_code, latest_adj_date + datetime.timedelta(days=1))
        if df_new is None or df_new.empty:
            logger.info("No new non-adjusted records to copy for stock %s.", stock_code)
        else:
            new_adj_records = []
            for _, row in df_new.iterrows():
                record = StockHistAdj(
                    stock_code = stock_code,
                    date = row["date"],
                    open = safe_value(row["open"]),
                    close = safe_value(row["close"]),
                    high = safe_value(row["high"]),
                    low = safe_value(row["low"]),
                    volume = safe_value(row["volume"]),
                    amount = safe_value(row["amount"]),
                    amplitude = safe_value(row["amplitude"]),
                    change_percent = safe_value(row["change_percent"]),
                    change = safe_value(row["change"]),
                    turnover_rate = safe_value(row["turnover_rate"]),
                    mkt_cap = safe_value(row.get("mkt_cap")),
                    total_shares = safe_value(row.get("total_shares"))
                )
                new_adj_records.append(record)
            await asyncio.to_thread(self.stock_hist_adj_dao.batch_insert, new_adj_records)
            logger.info("Copied %d new non-adjusted records for stock %s to adjusted table.", len(new_adj_records), stock_code)

    def sync(self, progress_callback=None):
        self.initialize()
        try:
            # 获取股票列表（此处采用 stock_info 表中的股票）
            stock_list = self.stock_info_dao.load_stock_info()
            self.stock_list_size = len(stock_list)
            logger.info("Processing forward-adjusted data for %d stocks.", len(stock_list))
            
            batch_size = 20
            for i in range(0, len(stock_list), batch_size):
                batch = stock_list[i : i + batch_size]
                tasks = []
                for stock in batch:
                    stock_code = stock.stock_code
                    logger.info("Processing stock %s", stock_code)
                    tasks.append(self.loop.create_task(self.process_data_adj(stock_code, progress_callback)))
                if tasks:
                    self.loop.run_until_complete(asyncio.gather(*tasks))

            if progress_callback:
                progress_callback(len(stock_list), len(stock_list))
        finally:
            self.terminate()


class AdjFactorSynchronizer:
    """
    同步复权因子到数据库。

    主要流程：
      1. 获取当前数据库中的股票列表
      2. 遍历股票列表，依次同步每只股票的复权因子数据
      3. 对于某一支股票，查看其最新数据的日期
      4. 如果最新日期小于等于当前日期，则同步该支股票的复权因子数据，调用 tushare 接口(tspro.adj_factor)
      5. 如果时间跨度大于20年，则结束日期截取至开始日期+20年
      6. 将新增的数据插入数据库
      7. 限制接口调用次数每分钟不超过400次
    """

    def __init__(self):
        self.tspro = tspro
        self.stock_info_dao = StockInfoDao._instance
        self.adj_factor_dao = AdjFactorDao._instance
        self.loop = asyncio.get_event_loop()
        self.semaphore = asyncio.Semaphore(6)  # 限制同时6个任务
        self.api_call_counter = 0
        self.api_call_reset_time = time.time() + 60
        self.stock_list_size = 0
        self.completed_num = 0
        self.lock = asyncio.Lock()

    def initialize(self):
        self.stock_list_size = 0
        self.completed_num = 0
        self.api_call_counter = 0
        self.api_call_reset_time = time.time() + 60

    def terminate(self):
        self.stock_list_size = 0
        self.completed_num = 0

    async def _check_api_rate_limit(self):
        current_time = time.time()
        if current_time > self.api_call_reset_time:
            self.api_call_counter = 0
            self.api_call_reset_time = current_time + 60
        if self.api_call_counter >= 400:
            wait_time = self.api_call_reset_time - current_time
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
            self.api_call_counter = 0
            self.api_call_reset_time = time.time() + 60
        self.api_call_counter += 1

    async def fetch_adj_factor(self, stock_code, start_date, end_date):
        try:
            async with self.semaphore:
                await self._check_api_rate_limit()

                ts_code = stock_code
                df = await asyncio.to_thread(
                    self.tspro.adj_factor,
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )

                if not df.empty:
                    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                return df

        except Exception as e:
            logger.error(f"Error fetching adj factor for {stock_code}: {e}")
            return pd.DataFrame()

    async def batch_insert_adj_factors(self, stock_code, df):
        """批量插入 adj_factor 表"""
        def _syn_batch_insert(records):
            if records:
                try:
                    self.adj_factor_dao.batch_insert(records)
                    logger.info(f"Inserted {len(records)} adj_factor records for stock {stock_code}")
                except Exception as e:
                    logger.error(f"Error inserting adj_factors for {stock_code}: {e}")

        records = []
        for _, row in df.iterrows():
            record = AdjFactor(
                stock_code=stock_code,
                date=row["trade_date"].date(),
                adj_factor=safe_value(row["adj_factor"])
            )
            records.append(record)

        if records:
            await asyncio.to_thread(_syn_batch_insert, records)

    async def process_stock(self, stock_code, market, start_date, end_date, progress_callback=None):
        try:
            logger.info(f"Fetching adj factor for stock {stock_code} from {start_date} to {end_date}")

            df = await self.fetch_adj_factor(stock_code, start_date, end_date)
            if df.empty:
                logger.info(f"No new adj factor data for stock {stock_code}.")
                return

            await self.batch_insert_adj_factors(stock_code, df)

        except Exception as e:
            logger.error(f"Error processing stock {stock_code}: {e}")
        finally:
            async with self.lock:
                self.completed_num += 1
                if progress_callback:
                    progress_callback(self.completed_num, self.stock_list_size)

    def sync(self, progress_callback=None):
        self.initialize()
        try:
            logger.info("Starting adj factor synchronization.")
            stock_list = self.stock_info_dao.load_stock_info()
            self.stock_list_size = len(stock_list)
            logger.info(f"Found {len(stock_list)} stocks to synchronize.")

            batch_size = 20
            for i in range(0, len(stock_list), batch_size):
                batch = stock_list[i:i+batch_size]
                tasks = []

                for stock in batch:
                    stock_code = stock.stock_code
                    market = stock.market

                    latest_date = self.adj_factor_dao.get_latest_date(stock_code)
                    if latest_date is None:
                        start_date_dt = datetime.date(1990, 1, 1)
                    else:
                        start_date_dt = latest_date + datetime.timedelta(days=1)

                    current_date_dt = datetime.date.today()
                    if start_date_dt > current_date_dt:
                        logger.info(f"Stock {stock_code} is up to date.")
                        continue

                    segment_days = 365 * 20
                    segment_start = start_date_dt
                    while segment_start <= current_date_dt:
                        segment_end = min(segment_start + datetime.timedelta(days=segment_days), current_date_dt)
                        tasks.append(self.loop.create_task(
                            self.process_stock(
                                stock_code,
                                market,
                                segment_start.strftime("%Y%m%d"),
                                segment_end.strftime("%Y%m%d"),
                                progress_callback
                            )
                        ))
                        segment_start = segment_end + datetime.timedelta(days=1)

                if tasks:
                    self.loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            self.terminate()

    def sync_by_trade_date(self, trade_date: str):
        """
        同步指定交易日 trade_date 的全市场复权因子数据到数据库。
        trade_date: 格式 'YYYYMMDD' （如 '20240726'）
        """
        try:
            logger.info(f"Fetching adj_factor for all stocks on {trade_date}")
            
            # 从tushare拉取指定交易日的复权因子数据
            df = tspro.adj_factor(ts_code='', trade_date=trade_date)
            
            if df.empty:
                logger.warning(f"No adj_factor data found for trade_date {trade_date}")
                return
            
            new_records = []

            for _, row in df.iterrows():
                try:
                    record = AdjFactor(
                        stock_code=safe_value(row['ts_code']),
                        date=pd.to_datetime(row['trade_date']).date(),
                        adj_factor=safe_value(row.get('adj_factor'))
                    )
                    new_records.append(record)
                except Exception as e:
                    logger.error(f"Error processing adj_factor row {row.to_dict()}: {e}")

            if new_records:
                self.adj_factor_dao.batch_upsert(new_records)
                logger.info(f"Inserted {len(new_records)} adj_factor records for {trade_date}")
            else:
                logger.info(f"No valid adj_factor records to insert for {trade_date}")

        except Exception as e:
            logger.error(f"Error syncing adj_factor for {trade_date}: {e}")

class CompanyActionSynchronizer:
    '''
    同步公司行动数据到数据库。
    
    主要流程：
      1. 获取当前数据库中的股票列表
      2. 遍历股票列表，依次同步每只股票的company_action数据
      3. 对于某一支股票，查看其最新数据的日期
      4. 如果最新日期小于等于当前日期，则查询该支股票的公司行动数据，使用akshare接口，
        分别调用ak.stock_history_dividend_detail(symbol="xxxxxx", indicator="分红")，和ak.stock_history_dividend_detail(symbol="xxxxxx", indicator="配股")，
        将每10股的送股、转增、配股、分红比例转换为每股的值，将数据插入数据库
      5. 将合并后的数据转换为统一格式的 CompanyAction 对象，并调用 DAO 插入数据库。
    '''
    def __init__(self):
        self.stock_info_dao = StockInfoDao._instance
        self.company_action_dao = CompanyActionDao._instance
        self.update_flag_dao = UpdateFlagDao._instance
        self.future_task_dao = FutureTaskDao._instance
        self.df_rights_all = None
        self.loop = asyncio.get_event_loop()
        self.stock_list_size = 0
        self.completed_num = 0
        self.is_running = False
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(6)

    def initialize(self):
        self.df_rights_all = None
        self.stock_list_size = 0
        self.completed_num = 0
        self.is_running = True

    def terminate(self):
        self.df_rights_all = None
        self.stock_list_size = 0
        self.completed_num = 0
        self.is_running = False

    def parse_ths_dividend(self, df):
        # 筛选实施方案的行
        df = df[df['方案进度'] == '实施方案'].copy()
        
        # 正则表达式匹配分红方案的每一段
        pattern = re.compile(r'(送(\d+\.?\d*)股)|(转(\d+\.?\d*)股)|(派(\d+\.?\d*)元)')
        
        rows = []
        for _, row in df.iterrows():
            方案说明 = row['分红方案说明']
            
            # 初始化默认值
            song, zhuan, pai = 0.0, 0.0, 0.0
            
            # 逐段匹配
            remaining_text = 方案说明
            while remaining_text:
                match = pattern.search(remaining_text)
                if not match:
                    break  # 没有匹配项，退出循环
                
                # 根据匹配结果更新值
                if match.group(1):  # 送股
                    song = float(match.group(2))
                elif match.group(3):  # 转股
                    zhuan = float(match.group(4))
                elif match.group(5):  # 派息
                    pai = float(match.group(6))
                
                # 删除已匹配的部分，继续匹配剩余文本
                remaining_text = remaining_text[match.end():]
            
            # 构造新行数据
            new_row = {
                '除权除息日': row['A股除权除息日'],
                '最新公告日期': row['实施公告日'],
                '送转股份-送股比例': song,
                '送转股份-转股比例': zhuan,
                '现金分红-现金分红比例': pai,
                '股权登记日': row['A股股权登记日']
            }
            rows.append(new_row)
        
        # 创建新DataFrame
        return pd.DataFrame(rows, columns=['除权除息日', '最新公告日期', '送转股份-送股比例', '送转股份-转股比例', '现金分红-现金分红比例', '股权登记日'])

    async def process_data(self, stock_code: str, progress_callback=None):
        try:
            async with self.semaphore:
                # 3. 分别获取分红和配股数据
                logger.info("Processing stock %s", stock_code)
                try:
                    df_dividend = await asyncio.to_thread(ak.stock_fhps_detail_em, symbol=stock_code)
                except Exception as e:
                    logger.error("Error fetching em dividend data for %s: %s", stock_code, e)
                    df_dividend = pd.DataFrame()
                # 如果没取到东财数据，尝试获取同花顺数据
                if df_dividend.empty:
                    try:
                        df_ths_dividend = await asyncio.to_thread(ak.stock_fhps_detail_ths, symbol=stock_code)
                        df_dividend = self.parse_ths_dividend(df_ths_dividend)
                        df_dividend.dropna(subset=['除权除息日', '最新公告日期', '股权登记日'], inplace=True)
                    except Exception as e:
                        logger.error("Error fetching ths dividend data for %s: %s", stock_code, e)
                        df_dividend = pd.DataFrame()
                try:
                    df_rights = self.df_rights_all[self.df_rights_all["股票代码"]==stock_code]
                except Exception as e:
                    logger.error("Error fetching rights data for %s: %s", stock_code, e)
                    df_rights = pd.DataFrame()

                # 4. 数据预处理
                # 对分红数据：重命名“除权除息日”为 ex_date，并过滤出 ex_date 不为空 的记录
                if not df_dividend.empty and "除权除息日" in df_dividend.columns:
                    df_dividend["ex_date"] = pd.to_datetime(df_dividend["除权除息日"], errors="coerce").dt.date
                    df_dividend = df_dividend[df_dividend["ex_date"].notnull()]
                    # 按 ex_date 分组，聚合各字段：对于比例字段采用求和（假设同一交易日内多个记录需要合并），对于日期字段采用首个非空值
                    df_dividend = df_dividend.groupby("ex_date").agg({
                        "最新公告日期": "first",
                        "送转股份-送股比例": "sum",
                        "送转股份-转股比例": "sum",
                        "现金分红-现金分红比例": "sum",
                        "股权登记日": "first"
                    }).reset_index()
                else:
                    df_dividend = pd.DataFrame(columns=["ex_date", "最新公告日期", "送转股份-送股比例", "送转股份-转股比例", "现金分红-现金分红比例", "股权登记日"])
                    
                # 对配股数据：重命名“除权日”为 ex_date，并过滤出 ex_date 不为空 的记录
                if not df_rights.empty and "除权日" in df_rights.columns:
                    df_rights["ex_date"] = pd.to_datetime(df_rights["除权日"], errors="coerce").dt.date
                    df_rights = df_rights[df_rights["ex_date"].notnull()]
                else:
                    df_rights = pd.DataFrame(columns=["ex_date", "公告日期", "配股比例", "配股价", "股权登记日"])

                if df_dividend.empty or df_rights.empty:
                    df_dividend["ex_date"] = df_dividend["ex_date"].astype(object)
                    df_rights["ex_date"] = df_rights["ex_date"].astype(object)

                # 5. 合并分红和配股数据：以 ex_date 为键，外连接
                df_merged = pd.merge(df_dividend, df_rights, on="ex_date", how="outer", suffixes=('_div', '_right'))
                logger.debug("Merged company action data for %s:\n%s", stock_code, df_merged)

                # 6. 根据合并结果构造统一的 CompanyAction 对象列表
                new_records = []
                for _, row in df_merged.iterrows():
                    try:
                        # 对比例字段除以10转换为每股数据
                        bonus_ratio = safe_value(row.get("送转股份-送股比例"))
                        if pd.notnull(bonus_ratio):
                            bonus_ratio = bonus_ratio / 10
                        conversion_ratio = safe_value(row.get("送转股份-转股比例"))
                        if pd.notnull(conversion_ratio):
                            conversion_ratio = conversion_ratio / 10
                        dividend_per_share = safe_value(row.get("现金分红-现金分红比例"))
                        if pd.notnull(dividend_per_share):
                            dividend_per_share = dividend_per_share / 10
                        rights_issue_ratio = safe_value(row.get("配股比例"))
                        if pd.notnull(rights_issue_ratio):
                            rights_issue_ratio = rights_issue_ratio / 10
                        rights_issue_price = safe_value(row.get("配股价"))

                        # 对公告日期和股权登记日，优先选用分红数据，如无则选用配股数据
                        ann_date = None
                        if pd.notnull(row.get("最新公告日期")):
                            ann_date = pd.to_datetime(row.get("最新公告日期"), errors="coerce").date()
                        elif pd.notnull(row.get("公告日期")):
                            ann_date = pd.to_datetime(row.get("公告日期"), errors="coerce").date()

                        rec_date = None
                        if pd.notnull(row.get("股权登记日_div")):
                            rec_date = pd.to_datetime(row.get("股权登记日_div"), errors="coerce").date()
                        elif pd.notnull(row.get("股权登记日_right")):
                            rec_date = pd.to_datetime(row.get("股权登记日_right"), errors="coerce").date()

                        record = CompanyAction(
                            stock_code = stock_code,
                            ex_dividend_date = safe_value(row["ex_date"]),
                            bonus_ratio = bonus_ratio,
                            conversion_ratio = conversion_ratio,
                            dividend_per_share = dividend_per_share,
                            rights_issue_ratio = rights_issue_ratio,
                            rights_issue_price = rights_issue_price,
                            announcement_date = ann_date,
                            record_date = rec_date
                        )
                        
                        new_records.append(record)
                    except Exception as e:
                        logger.error("Error processing merged row for stock %s: %s", stock_code, e)
                        raise e
                # 7. 插入新记录到数据库
                if new_records:
                    try:
                        # self.company_action_dao.batch_insert_if_not_exists(new_records)
                        inserted_results = await asyncio.to_thread(self.company_action_dao.batch_insert_if_not_exists, new_records)
                        logger.info("Inserted %d company action records for %s.", len(inserted_results), stock_code)
                        # 由于公司行动更新了，因些需要更新前复权数据，更新flag
                        for row in inserted_results:
                            self.future_task_dao.insert_one(FutureTask(stock_code=stock_code, task_date=row.ex_dividend_date, task_type=TaskType.UPDATE_ADJ.name, task_status=TaskStatus.INIT.name))
                    except Exception as e:
                        logger.error("Error inserting records for stock %s: %s", stock_code, e)
                        raise e
                else:
                    logger.info("No new company action records for %s.", stock_code)
                self.update_flag_dao.update_action_flag(stock_code, "0")
        except Exception as e:
            logger.error("Error processing stock %s for company action data: %s", stock_code, e)
            raise e
        finally:
            async with self.lock:
                self.completed_num = self.completed_num + 1
                if progress_callback:
                    progress_callback(self.completed_num, self.stock_list_size)
        

    def sync(self, progress_callback=None):
        self.initialize()
        try:
            logger.info("Starting company action synchronization.")
            # 1. 获取当前股票列表
            stock_list: List[StockInfo] = self.stock_info_dao.load_stock_info()
            self.stock_list_size = len(stock_list)
            logger.info("Found %d stocks to process.", len(stock_list))

            self.df_rights_all = ak.stock_pg_em()
            batch_size = 20
            for i in range(0, len(stock_list), batch_size):
                batch = stock_list[i : i + batch_size]
                tasks=[]
                for stock in batch:
                    stock_code = stock.stock_code
                    update_flags = self.update_flag_dao.select_one_by_code(stock_code)

                    if update_flags["action_update_flag"] == "0":
                        logger.info("Stock %s is up-to-date.", stock_code)
                        continue

                    # process_data
                    tasks.append(self.loop.create_task(self.process_data(stock_code, progress_callback)))
                if tasks:
                    self.loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            self.terminate()
        

class FundamentalDataSynchronizer:
    def __init__(self):
        self.stock_info_dao = StockInfoDao._instance  # 单例
        self.fundamental_dao = FundamentalDataDao._instance
        self.update_flag_dao = UpdateFlagDao._instance
        self.loop = asyncio.get_event_loop()
        self.stock_list_size = 0
        self.completed_num = 0
        self.is_running = False
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(6)
        self.api_call_reset_time = time.time() + 60
        self.api_call_counter = 0
        self.failed_stocks = []

    def initialize(self):
        self.stock_list_size = 0
        self.completed_num = 0
        self.is_running = True
        self.api_call_reset_time = time.time() + 60
        self.api_call_counter = 0
        self.failed_stocks = []

    def terminate(self):
        self.stock_list_size = 0
        self.completed_num = 0
        self.is_running = False

    async def _check_api_rate_limit(self):
        """检查并控制API调用频率，确保每分钟不超过200次"""
        current_time = time.time()
        
        if current_time > self.api_call_reset_time:
            # Reset counter if a minute has passed
            self.api_call_counter = 0
            self.api_call_reset_time = current_time + 60
            
        if self.api_call_counter >= 200:
            # Wait until the next minute if we've reached the limit
            wait_time = self.api_call_reset_time - current_time
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                self.api_call_counter = 0
                self.api_call_reset_time = time.time() + 60
        
        self.api_call_counter += 1

    async def fetch_all_data_by_tushare(self, period: str):
        logger.info(f"Fetching full market data for period {period} asynchronously...")
        df_income = await asyncio.to_thread(
            tspro.income_vip, period=period,
            fields="ts_code,f_ann_date,end_date,n_income_attr_p,operate_profit,total_revenue,total_cogs"
        )
        df_balance = await asyncio.to_thread(
            tspro.balancesheet_vip, period=period,
            fields="ts_code,f_ann_date,end_date,total_hldr_eqy_exc_min_int,total_assets,total_cur_liab,total_ncl"
        )
        df_cashflow = await asyncio.to_thread(
            tspro.cashflow_vip, period=period,
            fields="ts_code,f_ann_date,end_date,n_cashflow_act,c_pay_acq_const_fiolta"
        )
        return df_income, df_balance, df_cashflow

    async def process_one_period_by_tushare(self, period: str):
        try:
            df_income, df_balance, df_cashflow = await self.fetch_all_data_by_tushare(period)

            if df_income.empty and df_balance.empty and df_cashflow.empty:
                logger.warning(f"All dataframes empty for period {period}, skipping.")
                return

            df_merge = df_income.merge(df_balance, on=['ts_code', 'end_date'], how='outer', suffixes=('_income', '_balance'))
            df_merge = df_merge.merge(df_cashflow, on=['ts_code', 'end_date'], how='outer', suffixes=('', '_cashflow'))

            logger.info(f"Merged {len(df_merge)} records for period {period}.")

            fundamental_records = []
            for _, row in df_merge.iterrows():
                try:
                    report_date = pd.to_datetime(row['end_date'], errors="coerce").date()
                    if pd.isnull(report_date):
                        continue
                except Exception as e:
                    logger.error(f"Error parsing report_date for period {period}: {e}")
                    continue

                record = FundamentalData(
                    stock_code=row['ts_code'],
                    report_date=report_date,
                    total_equity=parse_amount(row.get('total_hldr_eqy_exc_min_int')),
                    total_assets=parse_amount(row.get('total_assets')),
                    current_liabilities=parse_amount(row.get('total_cur_liab')),
                    noncurrent_liabilities=parse_amount(row.get('total_ncl')),
                    net_profit=parse_amount(row.get('n_income_attr_p')),
                    operating_profit=parse_amount(row.get('operate_profit')),
                    total_revenue=parse_amount(row.get('total_revenue')),
                    total_cost=parse_amount(row.get('total_cogs')),
                    net_cash_from_operating=parse_amount(row.get('n_cashflow_act')),
                    cash_for_fixed_assets=parse_amount(row.get('c_pay_acq_const_fiolta'))
                )
                fundamental_records.append(record)

            if fundamental_records:
                logger.info(f"Batch upserting {len(fundamental_records)} fundamental records for period {period}...")
                await asyncio.to_thread(self.fundamental_dao.batch_upsert, fundamental_records)
                logger.info(f"Batch upsert successful for period {period}.")
            else:
                logger.warning(f"No fundamental records to insert for period {period}.")

        except Exception as e:
            logger.error(f"Error processing fundamental data for period {period}: {e}")

    async def get_one_stock_data_by_akshare(self, short_stock_code: str):
        await self._check_api_rate_limit()
        df_debt = None
        df_benefit = None
        df_cash = None
        try:
            df_debt = await asyncio.to_thread(ak.stock_financial_debt_ths, symbol=short_stock_code, indicator="按报告期")
        except Exception as e:
            logger.error(f"Error fetching debt data for stock {short_stock_code}: {e}")
            df_debt = pd.DataFrame(columns=['报告期', '报表核心指标', '*所有者权益（或股东权益）合计', '*资产合计', '*负债合计', '*归属于母公司所有者权益合计',
       '报表全部指标', '流动资产', '货币资金', '交易性金融资产', '应收票据及应收账款', '应收账款', '预付款项', '存货',
       '总现金', '流动资产合计', '非流动资产', '无形资产', '非流动资产合计', '资产合计', '流动负债', '短期借款',
       '应付票据及应付账款', '应付账款', '预收款项', '应付职工薪酬', '应交税费', '其他应付款合计', '应付股利',
       '其他应付款', '流动负债合计', '非流动负债', '长期借款', '非流动负债合计', '负债合计', '所有者权益（或股东权益）',
       '实收资本（或股本）', '资本公积', '盈余公积', '未分配利润', '归属于母公司所有者权益合计', '少数股东权益',
       '所有者权益（或股东权益）合计', '负债和所有者权益（或股东权益）合计'])
        try:
            df_benefit = await asyncio.to_thread(ak.stock_financial_benefit_ths, symbol=short_stock_code, indicator="按报告期")
        except Exception as e:
            logger.error(f"Error fetching benefit data for stock {short_stock_code}: {e}")
            df_benefit = pd.DataFrame(columns=['报告期', '报表核心指标', '*净利润', '*营业总收入', '*营业总成本', '*归属于母公司所有者的净利润', '报表全部指标',
       '一、营业总收入', '其中：营业收入', '二、营业总成本', '其中：营业成本', '营业税金及附加', '销售费用', '管理费用',
       '财务费用', '投资收益', '三、营业利润', '加：营业外收入', '减：营业外支出', '四、利润总额', '减：所得税费用',
       '五、净利润', '归属于母公司所有者的净利润', '六、每股收益', '（一）基本每股收益', '七、其他综合收益',
       '八、综合收益总额'])
        try:
            df_cash = await asyncio.to_thread(ak.stock_financial_cash_ths, symbol=short_stock_code, indicator="按报告期")
        except Exception as e:
            logger.error(f"Error fetching cash data for stock {short_stock_code}: {e}")
            df_cash = pd.DataFrame(columns=['报告期', '*经营活动产生的现金流量净额', '购建固定资产、无形资产和其他长期资产支付的现金'])
        return df_debt, df_benefit, df_cash

    async def process_data(self, short_stock_code: str, full_stock_code: str, progress_callback=None):
        # 3. 调用 akshare 接口获取三个数据源，indicator 均采用 "按报告期"
        try:
            df_debt, df_benefit, df_cash = await self.get_one_stock_data_by_akshare(short_stock_code)

            # 4. 将“报告期”转换为 Pandas Timestamp 类型
            for df in [df_debt, df_benefit, df_cash]:
                if "报告期" in df.columns:
                    df["报告期"] = pd.to_datetime(df["报告期"], errors="coerce")
                else:
                    logger.error("DataFrame missing '报告期' column for stock %s", full_stock_code)
                    return

            # 5. 合并三个 DataFrame，按 "报告期" 列内连接（假设各报表数据报告期一致）
            df_merge = pd.merge(df_debt, df_benefit, on="报告期", how="outer", suffixes=("_debt", "_benefit"))
            df_merge = pd.merge(df_merge, df_cash, on="报告期", how="outer")
            # 此时 df_merge 包含 "报告期" 以及各表中关键字段

            # 6. 根据合并结果构造 FundamentalData 对象列表
            fundamental_records = []
            for _, row in df_merge.iterrows():
                try:
                    report_ts = row["报告期"]
                    if pd.isnull(report_ts):
                        continue
                    report_date = report_ts.date()
                except Exception as e:
                    logger.error("Error parsing report_date for stock %s: %s", full_stock_code, e)
                    continue

                total_equity = parse_amount(row.get("*归属于母公司所有者权益合计"))
                total_assets = parse_amount(row.get("*资产合计"))
                current_liabilities = parse_amount(row.get("流动负债合计"))
                noncurrent_liabilities = parse_amount(row.get("非流动负债合计"))
                if current_liabilities is None and noncurrent_liabilities is None:
                    current_liabilities = parse_amount(row.get("*负债合计"))
                net_profit = parse_amount(row.get("*归属于母公司所有者的净利润"))
                if net_profit is None:
                    net_profit = parse_amount(row.get("*净利润"))
                operating_profit = parse_amount(row.get("三、营业利润"))
                total_revenue = parse_amount(row.get("*营业总收入"))
                total_cost = parse_amount(row.get("*营业总成本"))
                if total_cost is None:
                    total_cost = parse_amount(row.get("*营业支出"))
                net_cash_from_operating = parse_amount(row.get("*经营活动产生的现金流量净额"))
                cash_for_fixed_assets = parse_amount(row.get("购建固定资产、无形资产和其他长期资产支付的现金"))
                
                record = FundamentalData(
                    stock_code = full_stock_code,
                    report_date = report_date,
                    total_equity = total_equity,
                    total_assets = total_assets,
                    current_liabilities = current_liabilities,
                    noncurrent_liabilities = noncurrent_liabilities,
                    net_profit = net_profit,
                    operating_profit = operating_profit,
                    total_revenue = total_revenue,
                    total_cost = total_cost,
                    net_cash_from_operating = net_cash_from_operating,
                    cash_for_fixed_assets = cash_for_fixed_assets
                )
                fundamental_records.append(record)
            
            # 7. 批量插入 fundamental_records 到数据库
            if fundamental_records:
                try:
                    await asyncio.to_thread(self.fundamental_dao.batch_upsert, fundamental_records)
                    logger.info("Inserted %d new fundamental records for stock %s.", len(fundamental_records), full_stock_code)
                except Exception as e:
                    logger.error("Error inserting fundamental records for stock %s: %s", full_stock_code, e)
                    raise e
            else:
                logger.info("No new fundamental records to insert for stock %s.", full_stock_code)
        except Exception as e:
            logger.error("Error processing fundamental data for stock %s: %s", full_stock_code, e)
            self.failed_stocks.append(full_stock_code)
        finally:
            async with self.lock:
                self.completed_num = self.completed_num + 1
                if progress_callback:
                    progress_callback(self.completed_num, self.stock_list_size)

    def sync(self, progress_callback=None):
        self.initialize()
        try:
            logger.info("Starting fundamental data synchronization.")
            # 1. 获取股票列表
            stock_list = self.stock_info_dao.load_stock_info()
            stock_list_codes = [x.stock_code for x in stock_list]
            self.stock_list_size = len(stock_list_codes)
            logger.info("Found %d stocks to process.", len(stock_list_codes))

            batch_size = 20
            def run_tasks(stock_list_codes, batch_size, progress_callback):
                for i in range(0, len(stock_list_codes), batch_size):
                    batch = stock_list_codes[i : i + batch_size]
                    tasks = []
                    for full_stock_code in batch:
                        short_stock_code = full_stock_code[:full_stock_code.index(".")] #去掉股票市场标识，如600519.SH -> 600519
                        logger.info("Processing fundamental data for stock %s", full_stock_code)

                        # 2. 查询 update_flag表中，当前股票代码是否允许更新基本面数据
                        update_flags = self.update_flag_dao.select_one_by_code(full_stock_code)
                        if update_flags and update_flags["fundamental_update_flag"] == "0":
                            logger.info("Fundamental data update for stock %s is disabled.", full_stock_code)
                            self.completed_num = self.completed_num + 1
                            if progress_callback:
                                progress_callback(self.completed_num, self.stock_list_size)
                            continue
                        elif not update_flags:
                            continue

                        tasks.append(self.loop.create_task(self.process_data(short_stock_code, full_stock_code, progress_callback)))
                    if tasks:
                        self.loop.run_until_complete(asyncio.gather(*tasks))

            run_tasks(stock_list_codes, batch_size, progress_callback)
            # 重试 failed_stocks 列表
            failed_stocks_to_try = self.failed_stocks.copy()
            self.failed_stocks.clear()
            run_tasks(failed_stocks_to_try, batch_size, None)  #重试期间不更新进度

        finally:
            if self.failed_stocks:
                logger.error("Failed stocks: %s", self.failed_stocks)
            else:
                logger.info("Fundamental data synchronization completed.")
            self.terminate()

    def sync_by_period(self, period: str):
        """
        同步指定单一季度
        """
        self.initialize()
        try:
            logger.info(f"Syncing specified period {period}...")
            self.loop.run_until_complete(self.process_one_period_by_tushare(period))
        except Exception as e:
            logger.error(f"Error during specified period {period} synchronization: {e}")
        finally:
            self.terminate()


class SuspendDataSynchronizer:
    def __init__(self):
        self.suspend_data_dao = SuspendDataDao._instance

    def _parse_date(self, val):
        """
        将输入值转换为 datetime.date，如果值为 NaT 或无法转换，则返回 None
        """
        if pd.isnull(val):
            return None
        try:
            dt = pd.to_datetime(val, errors="coerce")
            if pd.isnull(dt):
                return None
            return dt.date()
        except Exception as e:
            logger.error("Error parsing date %s: %s", val, e)
            return None
        
    def sync_all(self, date: str, progress_callback=None):
        """
        获取指定日期（全量数据查询日期，例如 "20120222"）的停复牌数据，
        将数据入库。接口返回的是从该日期起的全量数据。
        """
        try:
            logger.info("Fetching full suspend data from date %s", date)
            df = ak.stock_tfp_em(date=date)
        except Exception as e:
            logger.error("Error fetching suspend data for date %s: %s", date, e)
            return

        if df.empty:
            logger.info("No suspend data fetched for date %s.", date)
            return

        # 预处理：转换日期字段
        df["停牌时间"] = df["停牌时间"].apply(self._parse_date)
        df["停牌截止时间"] = df["停牌截止时间"].apply(self._parse_date)
        
        records = []
        for idx, row in df.iterrows():
            stock_code = row.get("代码")
            suspend_date = safe_value(row.get("停牌时间"))
            # 忽略停牌时间为空的记录
            if suspend_date is None:
                continue
            resume_date = safe_value(row.get("停牌截止时间"))
            suspend_period = safe_value(row.get("停牌期限"))
            suspend_reason = safe_value(row.get("停牌原因"))
            market = safe_value(row.get("所属市场"))
            record = SuspendData(
                stock_code = stock_code,
                suspend_date = suspend_date,
                resume_date = resume_date,
                suspend_period = suspend_period,
                suspend_reason = suspend_reason,
                market = market
            )
            records.append(record)
            if progress_callback:
                progress_callback(idx, len(df))
        if records:
            self.suspend_data_dao.batch_upsert(records)
            logger.info("Full sync: Processed %d suspend data records.", len(records))
        else:
            logger.info("Full sync: No suspend data records to process.")

    def sync_today(self, progress_callback=None):
        """
        获取当天增量停复牌数据（接口参数 date 为当天），
        对于每条记录，根据股票代码+停牌时间判断是否已存在，存在则更新，否则新增。
        """
        today = datetime.date.today()
        today_str = today.strftime("%Y%m%d")
        try:
            logger.info("Fetching today's suspend data for date %s", today_str)
            df = ak.stock_tfp_em(date=today_str)
        except Exception as e:
            logger.error("Error fetching today's suspend data for date %s: %s", today_str, e)
            return

        if df.empty:
            logger.info("No suspend data fetched for today %s.", today_str)
            return

        df["停牌时间"] = df["停牌时间"].apply(self._parse_date)
        df["停牌截止时间"] = df["停牌截止时间"].apply(self._parse_date)
        
        records = []
        for idx, row in df.iterrows():
            stock_code = row.get("代码")
            suspend_date = safe_value(row.get("停牌时间"))
            if suspend_date is None:
                continue
            resume_date = safe_value(row.get("停牌截止时间"))
            suspend_period = safe_value(row.get("停牌期限"))
            suspend_reason = safe_value(row.get("停牌原因"))
            market = safe_value(row.get("所属市场"))
            record = SuspendData(
                stock_code = stock_code,
                suspend_date = suspend_date,
                resume_date = resume_date,
                suspend_period = suspend_period,
                suspend_reason = suspend_reason,
                market = market
            )
            records.append(record)
            if progress_callback:
                progress_callback(idx, len(df))
        if records:
            self.suspend_data_dao.batch_upsert(records)
            logger.info("Incremental sync: Processed %d suspend data records for today.", len(records))
        else:
            logger.info("Incremental sync: No suspend data records to process for today.")

def parse_amount(s: Union[str, float, None]) -> Union[float, None]:
    """
    将金额字符串转换为 float 数值，统一单位为亿：
      - 如果金额以“亿”结尾，去除后缀直接转换；
      - 如果以“万”结尾，则转换后除以 10000；
      - 否则尝试直接转换。
    如果转换失败，返回 None。
    """
    if pd.isnull(s) or s is None:
        return None
    try:
        s = str(s).strip()
        if s.endswith("万亿"):
            return float(s[:-2]) * 1e12
        elif s.endswith("亿"):
            return float(s[:-1]) * 100000000
        elif s.endswith("万"):
            return float(s[:-1]) * 10000.0
        else:
            return float(s)
    except Exception as e:
        logger.error("parse_amount error for value %s: %s", s, e)
        return None
    
stock_info_synchronizer = StockInfoSynchronizer()
stock_hist_synchronizer = StockHistSynchronizer()
adj_factor_synchronizer = AdjFactorSynchronizer()
stock_adj_hist_synchronizer = StockAdjHistSynchronizer()
company_action_synchronizer = CompanyActionSynchronizer()
fundamental_data_synchronizer = FundamentalDataSynchronizer()
suspend_data_synchronizer = SuspendDataSynchronizer()