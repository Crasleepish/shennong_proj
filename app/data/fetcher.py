from typing import List
import akshare as ak
import pandas as pd
import datetime
from sqlalchemy.orm import Session
from app.models.stock_models import StockInfo, StockHistUnadj, UpdateFlag, CompanyAction
from app.dao.stock_info_dao import StockInfoDao, StockHistUnadjDao, UpdateFlagDao, CompanyActionDao

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
        self.stock_info_dao = StockInfoDao._instance
        self.update_flag_dao = UpdateFlagDao._instance

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

            # 将新数据插入到update_flag表中
            for record in new_records:
                self.update_flag_dao.insert_one(UpdateFlag(stock_code=record.stock_code, action_update_flag='1'))
            
            logger.info("Inserted %d new records into the database.", len(new_records))
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
      4. 如果最新日期小于等于当前日期，则同步该支股票的历史行情数据（不复权），调用akshare接口（ak.stock_zh_a_hist）获取最新数据次日到当前日期的历史行情数据
      5. 将新增的数据插入数据库
    """
    def __init__(self):
        self.stock_info_dao = StockInfoDao._instance
        self.stock_hist_dao = StockHistUnadjDao._instance
        self.update_flag_dao = UpdateFlagDao._instance

    def sync(self):
        logger.info("Starting stock historical data synchronization.")
        # 1. 获取当前数据库中的股票列表
        stock_list = self.stock_info_dao.load_stock_info()
        logger.info("Found %d stocks to synchronize.", len(stock_list))
        
        for stock in stock_list:
            stock_code = stock.stock_code
            logger.info("Synchronizing stock %s", stock_code)
            # 2. 查询该股票在历史行情数据表中的最新日期
            latest_date = self.stock_hist_dao.get_latest_date(stock_code)
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
                logger.info("Stock %s is up to date. (start_date: %s, current_date: %s)", stock_code, start_date, current_date)
                continue

            # 3. 调用 akshare 接口获取该股票从 start_date 到 current_date 的历史行情数据（不复权）
            try:
                logger.info("Fetching historical data for stock %s from %s to %s", stock_code, start_date, current_date)
                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date,
                    end_date=current_date,
                    adjust=""
                )
            except Exception as e:
                logger.error("Error fetching historical data for stock %s: %s", stock_code, e)
                continue

            if df.empty:
                logger.info("No new historical data for stock %s.", stock_code)
                continue

            # 将日期列转换为日期类型（注意 akshare 返回的“日期”列可能为 object 类型）
            if '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'], errors='coerce').dt.date

            # 4. 处理 DataFrame 数据，构造 StockHist 对象列表
            new_records = []
            for _, row in df.iterrows():
                record = StockHistUnadj(
                    date=row["日期"],
                    stock_code=row["股票代码"],
                    open=row["开盘"],
                    close=row["收盘"],
                    high=row["最高"],
                    low=row["最低"],
                    volume=row["成交量"] * 100,
                    turnover=row["成交额"],
                    amplitude=row["振幅"],
                    change_percent=row["涨跌幅"],
                    change=row["涨跌额"],
                    turnover_rate=row["换手率"]
                )
                new_records.append(record)

            # 5. 批量插入新增记录到数据库
            if new_records:
                try:
                    result_records = self.stock_hist_dao.batch_insert(new_records)
                    logger.info("Inserted %d new historical records for stock %s.", len(result_records), stock_code)
                    self.update_flag_dao.update_action_flag(stock_code, "1")
                except Exception as e:
                    logger.error("Error inserting new records for stock %s: %s", stock_code, e)
                    raise e
            else:
                logger.info("No new records to insert for stock %s.", stock_code)


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

    def sync(self):
        logger.info("Starting company action synchronization.")
        # 1. 获取当前股票列表
        stock_list: List[StockInfo] = self.stock_info_dao.load_stock_info()
        logger.info("Found %d stocks to process.", len(stock_list))
        for stock in stock_list:
            stock_code = stock.stock_code
            update_flags = self.update_flag_dao.select_one_by_code(stock_code)

            if update_flags["action_update_flag"] == "0":
                logger.info("Stock %s is up-to-date.", stock_code)
                continue

            # 3. 分别获取分红和配股数据
            logger.info("Processing stock %s", stock_code)
            try:
                df_dividend = ak.stock_history_dividend_detail(symbol=stock_code, indicator="分红")
            except Exception as e:
                logger.error("Error fetching dividend data for %s: %s", stock_code, e)
                df_dividend = pd.DataFrame()
            try:
                df_rights = ak.stock_history_dividend_detail(symbol=stock_code, indicator="配股")
            except Exception as e:
                logger.error("Error fetching rights data for %s: %s", stock_code, e)
                df_rights = pd.DataFrame()

            # 4. 数据预处理
            # 对分红数据：重命名“除权除息日”为 ex_date，并过滤出 ex_date 不为空 的记录
            if not df_dividend.empty and "除权除息日" in df_dividend.columns:
                df_dividend["ex_date"] = pd.to_datetime(df_dividend["除权除息日"], errors="coerce").dt.date
                df_dividend = df_dividend[df_dividend["ex_date"].notnull()]
            else:
                df_dividend = pd.DataFrame(columns=["ex_date", "公告日期", "送股", "转增", "派息", "股权登记日"])
            # 对配股数据：重命名“除权日”为 ex_date，并过滤出 ex_date 不为空 的记录
            if not df_rights.empty and "除权日" in df_rights.columns:
                df_rights["ex_date"] = pd.to_datetime(df_rights["除权日"], errors="coerce").dt.date
                df_rights = df_rights[df_rights["ex_date"].notnull()]
            else:
                df_rights = pd.DataFrame(columns=["ex_date", "公告日期", "配股方案", "配股价格", "股权登记日"])

            # 5. 合并分红和配股数据：以 ex_date 为键，外连接
            df_merged = pd.merge(df_dividend, df_rights, on="ex_date", how="outer", suffixes=('_div', '_right'))
            logger.debug("Merged company action data for %s:\n%s", stock_code, df_merged)

            # 6. 根据合并结果构造统一的 CompanyAction 对象列表
            new_records = []
            for _, row in df_merged.iterrows():
                try:
                    # 判断 (stock_code, ex_dividend_date) 是否已存在于数据库中
                    if self.company_action_dao.select_by_code_and_date(stock_code, row["ex_date"]) is not None:
                        logger.debug("Record for stock %s on date %s already exists. Skipping.", stock_code, row["ex_date"])
                        continue

                    # 对比例字段除以10转换为每股数据
                    bonus_ratio = row.get("送股")
                    if pd.notnull(bonus_ratio):
                        bonus_ratio = bonus_ratio / 10
                    conversion_ratio = row.get("转增")
                    if pd.notnull(conversion_ratio):
                        conversion_ratio = conversion_ratio / 10
                    dividend_per_share = row.get("派息")
                    if pd.notnull(dividend_per_share):
                        dividend_per_share = dividend_per_share / 10
                    rights_issue_ratio = row.get("配股方案")
                    if pd.notnull(rights_issue_ratio):
                        rights_issue_ratio = rights_issue_ratio / 10
                    rights_issue_price = row.get("配股价格")

                    # 对公告日期和股权登记日，优先选用分红数据，如无则选用配股数据
                    ann_date = None
                    if pd.notnull(row.get("公告日期_div")):
                        ann_date = pd.to_datetime(row.get("公告日期_div"), errors="coerce").date()
                    elif pd.notnull(row.get("公告日期_right")):
                        ann_date = pd.to_datetime(row.get("公告日期_right"), errors="coerce").date()

                    rec_date = None
                    if pd.notnull(row.get("股权登记日_div")):
                        rec_date = pd.to_datetime(row.get("股权登记日_div"), errors="coerce").date()
                    elif pd.notnull(row.get("股权登记日_right")):
                        rec_date = pd.to_datetime(row.get("股权登记日_right"), errors="coerce").date()

                    record = CompanyAction(
                        stock_code = stock_code,
                        ex_dividend_date = row["ex_date"],
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
                    self.company_action_dao.batch_insert(new_records)
                    logger.info("Inserted %d new company action records for %s.", len(new_records), stock_code)
                except Exception as e:
                    logger.error("Error inserting records for stock %s: %s", stock_code, e)
                    raise e
            else:
                logger.info("No new company action records for %s.", stock_code)
