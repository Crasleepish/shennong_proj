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
        df = ak.stock_info_sh_name_code(symbol=self.symbol)[['证券代码', '证券简称', '上市日期']]
        # 将上市日期转换为日期类型（若转换失败则置为 NaT）
        if '上市日期' in df.columns:
            df['上市日期'] = pd.to_datetime(df['上市日期'], errors='coerce').dt.date
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
            new_data = df[~df['证券代码'].isin(existing_codes)]
            logger.info("Found %d new records to insert.", len(new_data))
            
            if new_data.empty:
                logger.info("No new records to insert.")
                return
            
            # 将 DataFrame 中每一行转换为 StockInfo 对象
            new_records = []
            for idx, row in new_data.iterrows():
                record = StockInfo(
                    stock_code=row['证券代码'],
                    stock_name=row['证券简称'],
                    listing_date=row['上市日期']
                )
                new_records.append(record)
            
            # 批量插入新增记录
            self.stock_info_dao.batch_insert(new_records)
            logger.info("Inserted %d new records into the database.", len(new_records))
        except Exception as e:
            logger.exception("Error during synchronization: %s", str(e))