from typing import List
import akshare as ak
import pandas as pd
from sqlalchemy.orm import Session
from app.models.stock_models import StockInfo
from app.dao.stock_info_dao import StockInfoDao

import logging

logger = logging.getLogger(__name__)

class StockInfoSynchronizer:
    """
    同步股票信息数据到数据库。
    
    主要流程：
      1. 调用 akshare 接口获取数据（pandas DataFrame）
      2. 查询数据库中已存在的股票代码集合
      3. 筛选出新增记录（基于证券代码）
      4. 将新增记录插入数据库（仅新增，不更新或删除）
    """
    def __init__(self, symbol="主板A股"):
        self.symbol = symbol
        self.stock_info_dao = StockInfoDao()

    def fetch_data(self) -> pd.DataFrame:
        """
        调用 akshare 接口获取数据，返回 pandas DataFrame
        
        示例接口：
            import akshare as ak
            df = ak.stock_info_sh_name_code(symbol="主板A股")
        """
        logger.info("Fetching data from akshare for symbol: %s", self.symbol)
        sh_a_list = ak.stock_info_sh_name_code(symbol=self.symbol)[['证券代码', '证券简称', '上市日期']]
        sh_a_list = sh_a_list.rename(columns={'证券代码': 'code', '证券简称': 'name', '上市日期': 'ipo_date'})
        sh_a_list['market'] = 'SH'
        # 将上市日期转换为日期类型（若转换失败则置为 NaT）
        if 'ipo_date' in sh_a_list.columns:
            sh_a_list['ipo_date'] = pd.to_datetime(sh_a_list['ipo_date'], errors='coerce').dt.date
        sz_a_list = ak.stock_info_sz_name_code(symbol="A股列表")[['A股代码', 'A股简称', 'A股上市日期']]
        sz_a_list = sz_a_list.rename(columns={'A股代码': 'code', 'A股简称': 'name', 'A股上市日期': 'ipo_date'})
        sz_a_list['market'] = 'SZ'
        if 'ipo_date' in sz_a_list.columns:
            sz_a_list['ipo_date'] = pd.to_datetime(sz_a_list['ipo_date'], errors='coerce').dt.date
        df = pd.concat([sh_a_list, sz_a_list], ignore_index=True).sort_values('ipo_date').reset_index(drop=True)
        return df

    def sync(self):
        """
        同步数据：将接口返回的新增记录插入到数据库中
        """
        logger.info("Starting synchronization for stock info.")
        df = self.fetch_data()
        if df.empty:
            logger.warning("Fetched data is empty.")
            return
        
        # 获取数据库中已有的股票代码集合
        try:
            stock_info_lst : List[StockInfo] = self.stock_info_dao.load_stock_info()
            if not stock_info_lst:
                stock_info_lst = []
            existing_codes = {si.stock_code for si in stock_info_lst}
            logger.debug("Existing stock codes in DB: %s", existing_codes)
            
            # 筛选出新增数据（证券代码不在 existing_codes 中）
            new_data = df[~df['code'].isin(existing_codes)]
            logger.info("Found %d new records to insert.", len(new_data))
            
            if new_data.empty:
                logger.info("No new records to insert.")
                return
            
            # 将 DataFrame 中每一行转换为 StockInfo 对象
            new_records = []
            for idx, row in new_data.iterrows():
                record = StockInfo(
                    stock_code=row['code'],
                    stock_name=row['name'],
                    listing_date=row['ipo_date'],
                    market=row['market']
                )
                new_records.append(record)
            
            # 批量插入新增记录
            self.stock_info_dao.batch_insert(new_records)
            logger.info("Inserted %d new records into the database.", len(new_records))
        except Exception as e:
            logger.exception("Error during synchronization: %s", str(e))