from typing import List
from bisect import bisect_left
import akshare as ak
import pandas as pd
import datetime
import math
from sqlalchemy.orm import Session
from app.models.stock_models import StockInfo, StockHistUnadj, UpdateFlag, CompanyAction, FutureTask, StockHistAdj, FundamentalData, SuspendData
from app.dao.stock_info_dao import StockInfoDao, StockHistUnadjDao, UpdateFlagDao, CompanyActionDao, FutureTaskDao, StockHistAdjDao, FundamentalDataDao, SuspendDataDao
from app.constants.enums import TaskType, TaskStatus
from typing import Union

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
        self.future_task_dao = FutureTaskDao._instance

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
                self.update_flag_dao.insert_one(UpdateFlag(stock_code=record.stock_code, action_update_flag='1', fundamental_update_flag='1'))
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
        self.stock_hist_unadj_dao = StockHistUnadjDao._instance
        self.update_flag_dao = UpdateFlagDao._instance
        self.company_action_dao = CompanyActionDao._instance
        self.stock_hist_adj_dao = StockHistAdjDao()
        self.future_task_dao = FutureTaskDao._instance

    def sync(self):
        logger.info("Starting stock historical data synchronization.")
        # 1. 获取当前数据库中的股票列表
        stock_list = self.stock_info_dao.load_stock_info()
        logger.info("Found %d stocks to synchronize.", len(stock_list))
        
        for stock in stock_list:
            stock_code = stock.stock_code
            logger.info("Synchronizing stock %s", stock_code)
            # 2. 查询该股票在历史行情数据表中的最新日期
            latest_date = self.stock_hist_unadj_dao.get_latest_date(stock_code)
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

            # 将日期列转换为日期类型
            if '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'], errors='coerce')

            # 获取股本变动数据并合并
            # 使用无复权数据的最早日期作为查询起始日期
            min_date = df["日期"].min()
            start_date_share = min_date.strftime("%Y%m%d")
            # 当前日期作为结束日期（已在 YYYYMMDD 格式下）
            end_date_share = current_date

            try:
                share_list_df = ak.stock_ipo_summary_cninfo(
                    symbol=stock_code,
                )
                share_list_df = share_list_df[["上市日期", "总发行数量"]]
                share_list_df.rename(columns={"上市日期": "变动日期", "总发行数量": "总股本"}, inplace=True)
                share_change_df = ak.stock_share_change_cninfo(
                    symbol=stock_code,
                    start_date=start_date_share,
                    end_date=end_date_share
                )
                share_change_df = share_change_df[["变动日期", "总股本"]]
                share_change_df = pd.concat([share_list_df, share_change_df], ignore_index=True).reset_index(drop=True)
                share_change_df.dropna(subset=["变动日期", "总股本"], inplace=True)
            except Exception as e:
                logger.error("Error fetching share change data for stock %s: %s", stock_code, e)
                share_change_df = pd.DataFrame()

            # 如果接口返回不为空，预处理数据
            if not share_change_df.empty and "变动日期" in share_change_df.columns and "总股本" in share_change_df.columns:
                # 将“变动日期”转换为日期对象
                share_change_df["变动日期"] = pd.to_datetime(share_change_df["变动日期"], errors="coerce")
                # 按“变动日期”升序排序
                share_change_df = share_change_df.sort_values(by="变动日期")
                # 仅保留需要的字段
                share_change_df = share_change_df[["变动日期", "总股本"]]
                # 利用 merge_asof 合并（方向为 backward：取最近且不超过当天的记录）
                df = pd.merge_asof(
                    df.sort_values(by="日期"),
                    share_change_df.sort_values(by="变动日期"),
                    left_on="日期",
                    right_on="变动日期",
                    direction="backward"
                )
                # 合并结果中，字段“总股本”即为每个交易日对应的总股本，重命名为 total_shares
                df.rename(columns={"总股本": "total_shares"}, inplace=True)
                # total_shares 是“万股”：
                df['total_shares'] = df['total_shares'] * 1e4
                # 计算总市值，公式： mkt_cap = close * total_shares
                df["mkt_cap"] = df["收盘"] * df["total_shares"]
            else:
                # 如果接口无数据，则填充为空值
                df["total_shares"] = None
                df["mkt_cap"] = None

            # 4. 处理 DataFrame 数据，构造 StockHist 对象列表
            new_records = []
            for _, row in df.iterrows():
                record = StockHistUnadj(
                    date=row["日期"].date(),
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
                    turnover_rate=row["换手率"],
                    total_shares = row.get("total_shares"),
                    mkt_cap = row.get("mkt_cap")
                )
                new_records.append(record)

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

    def sync_adj(self):
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
        today = datetime.date.today()
        # 获取股票列表（此处采用 stock_info 表中的股票）
        stock_list = self.stock_info_dao.load_stock_info()
        logger.info("Processing forward-adjusted data for %d stocks.", len(stock_list))
        
        for stock in stock_list:
            stock_code = stock.stock_code
            logger.info("Processing stock %s", stock_code)
            # 检查未来任务是否存在 UPDATE_ADJ 任务，状态为 INIT
            future_tasks = self.future_task_dao.select_by_code_date_type(stock_code, today, TaskType.UPDATE_ADJ.name)
            if future_tasks is None:
                continue
            for future_task in future_tasks:
                if future_task is not None and future_task.task_status == TaskStatus.INIT.name:
                    logger.info("Found UPDATE_ADJ task for %s, updating forward-adjusted history.", stock_code)
                    # 更新模式：删除该股票所有前复权数据
                    self.stock_hist_adj_dao.delete_by_stock_code(stock_code)
                    # 获取所有无复权数据，按日期升序排列
                    df_unadj = self.stock_hist_unadj_dao.select_all_as_dataframe(stock_code)
                    if df_unadj.empty:
                        logger.info("No non-adjusted data for stock %s, skipping.", stock_code)
                        continue
                    # 获取所有公司行动记录（已转换为每股数据），按 ex_dividend_date 升序排列
                    df_action = self.company_action_dao.select_all_as_dataframe(stock_code)
                    # 遍历无复权记录，确保按交易日期升序排列
                    df_unadj = df_unadj.sort_values(by="date")
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
                            bonus = (action.get("bonus_ratio") or 0.0) if not math.isnan(action.get("bonus_ratio") or 0.0) else 0.0
                            conversion = (action.get("conversion_ratio") or 0.0) if not math.isnan(action.get("conversion_ratio") or 0.0) else 0.0
                            rights_issue = (action.get("rights_issue_ratio") or 0.0) if not math.isnan(action.get("rights_issue_ratio") or 0.0) else 0.0
                            dividend = (action.get("dividend_per_share") or 0.0) if not math.isnan(action.get("dividend_per_share") or 0.0) else 0.0
                            rights_issue_price = (action.get("rights_issue_price")) or 0.0 if not math.isnan((action.get("rights_issue_price")) or 0.0) else 0.0
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
                        adj_open = row["open"] * cumulative_factor if row["open"] is not None else None
                        adj_close = row["close"] * cumulative_factor if row["close"] is not None else None
                        adj_high = row["high"] * cumulative_factor if row["high"] is not None else None
                        adj_low = row["low"] * cumulative_factor if row["low"] is not None else None
                        adj_volume = row["volume"] / cumulative_factor if row["volume"] is not None else None
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
                            amplitude = adj_amplitude,
                            change_percent = adj_change_percent,
                            change = adj_change,
                            turnover_rate = row.get("turnover_rate"),
                            total_shares = row.get("total_shares"),
                            mkt_cap = row.get("mkt_cap")
                        )
                        adj_records.append(record)
                    # 批量插入前复权数据
                    if adj_records:
                        self.stock_hist_adj_dao.batch_insert(adj_records)
                        logger.info("Updated forward-adjusted data for stock %s: %d records inserted.", stock_code, len(adj_records))
                    else:
                        logger.info("No forward-adjusted records computed for stock %s.", stock_code)
                    # 更新未来任务状态为 DONE
                    self.future_task_dao.update_status_by_id(future_task.task_id, "DONE")
                    break
                else:
                    # 无更新任务时：直接复制无复权数据表中，日期在前复权数据表最新日期之后的所有数据到前复权数据表
                    latest_adj_date = self.stock_hist_adj_dao.get_latest_date(stock_code)
                    if latest_adj_date is None:
                        # 如果前复权数据表无记录，则设置为极早日期
                        latest_adj_date = datetime.date(1900, 1, 1)
                    # 查询无复权数据中日期大于最新前复权日期的所有记录
                    df_new = self.stock_hist_unadj_dao.select_after_date_as_dataframe(stock_code, latest_adj_date + datetime.timedelta(days=1))
                    if df_new.empty:
                        logger.info("No new non-adjusted records to copy for stock %s.", stock_code)
                    else:
                        new_adj_records = []
                        for _, row in df_new.iterrows():
                            record = StockHistAdj(
                                stock_code = stock_code,
                                date = row["date"],
                                open = row["open"],
                                close = row["close"],
                                high = row["high"],
                                low = row["low"],
                                volume = row["volume"],
                                amplitude = row["amplitude"],
                                change_percent = row["change_percent"],
                                change = row["change"],
                                turnover_rate = row["turnover_rate"],
                                total_shares = row.get("total_shares"),
                                mkt_cap = row.get("mkt_cap")
                            )
                            new_adj_records.append(record)
                        self.stock_hist_adj_dao.batch_insert(new_adj_records)
                        logger.info("Copied %d new non-adjusted records for stock %s to adjusted table.", len(new_adj_records), stock_code)
                    break


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
                # 按 ex_date 分组，聚合各字段：对于比例字段采用求和（假设同一交易日内多个记录需要合并），对于日期字段采用首个非空值
                df_dividend = df_dividend.groupby("ex_date").agg({
                    "公告日期": "first",
                    "送股": "sum",
                    "转增": "sum",
                    "派息": "sum",
                    "股权登记日": "first"
                }).reset_index()
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
                    # 由于公司行动更新了，因些需要更新前复权数据，更新flag
                    for row in new_records:
                        self.future_task_dao.insert_one(FutureTask(stock_code=stock_code, task_date=row.ex_dividend_date, task_type=TaskType.UPDATE_ADJ.name, task_status=TaskStatus.INIT.name))
                except Exception as e:
                    logger.error("Error inserting records for stock %s: %s", stock_code, e)
                    raise e
            else:
                logger.info("No new company action records for %s.", stock_code)
            self.update_flag_dao.update_action_flag(stock_code, "0")

class FundamentalDataSynchronizer:
    def __init__(self):
        self.stock_info_dao = StockInfoDao._instance  # 单例
        self.fundamental_dao = FundamentalDataDao._instance

    def sync(self):
        logger.info("Starting fundamental data synchronization.")
        # 1. 获取股票列表
        stock_list = self.stock_info_dao.load_stock_info()
        logger.info("Found %d stocks to process.", len(stock_list))
        
        for stock in stock_list:
            stock_code = stock.stock_code
            logger.info("Processing fundamental data for stock %s", stock_code)
            # 2. 查询 fundamental_data 表中最新的报告期
            latest_date = self.fundamental_dao.get_latest_report_date(stock_code)
            if latest_date is None:
                latest_date = datetime.date(1900, 1, 1)
            logger.info("Latest fundamental report date for %s: %s", stock_code, latest_date)
            
            # 3. 调用 akshare 接口获取三个数据源，indicator 均采用 "按报告期"
            try:
                df_debt = ak.stock_financial_debt_ths(symbol=stock_code, indicator="按报告期")
            except Exception as e:
                logger.error("Error fetching debt data for %s: %s", stock_code, e)
                continue
            try:
                df_benefit = ak.stock_financial_benefit_ths(symbol=stock_code, indicator="按报告期")
            except Exception as e:
                logger.error("Error fetching benefit data for %s: %s", stock_code, e)
                continue
            try:
                df_cash = ak.stock_financial_cash_ths(symbol=stock_code, indicator="按报告期")
            except Exception as e:
                logger.error("Error fetching cash data for %s: %s", stock_code, e)
                continue

            # 4. 将“报告期”转换为 Pandas Timestamp 类型，并过滤出报告期 > latest_date 的记录
            for df in [df_debt, df_benefit, df_cash]:
                if "报告期" in df.columns:
                    df["报告期"] = pd.to_datetime(df["报告期"], errors="coerce")
                else:
                    logger.error("DataFrame missing '报告期' column for stock %s", stock_code)
                    continue

            df_debt = df_debt[df_debt["报告期"].dt.date > latest_date]
            df_benefit = df_benefit[df_benefit["报告期"].dt.date > latest_date]
            df_cash = df_cash[df_cash["报告期"].dt.date > latest_date]

            if df_debt.empty or df_benefit.empty or df_cash.empty:
                logger.info("No new fundamental data for stock %s.", stock_code)
                continue

            # 5. 合并三个 DataFrame，按 "报告期" 列内连接（假设各报表数据报告期一致）
            df_merge = pd.merge(df_debt, df_benefit, on="报告期", suffixes=("_debt", "_benefit"))
            df_merge = pd.merge(df_merge, df_cash, on="报告期")
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
                    logger.error("Error parsing report_date for stock %s: %s", stock_code, e)
                    continue

                total_equity = parse_amount(row.get("*归属于母公司所有者权益合计"))
                total_assets = parse_amount(row.get("*资产合计"))
                current_liabilities = parse_amount(row.get("流动负债合计"))
                noncurrent_liabilities = parse_amount(row.get("非流动负债合计"))
                net_profit = parse_amount(row.get("*归属于母公司所有者的净利润"))
                operating_profit = parse_amount(row.get("三、营业利润"))
                total_revenue = parse_amount(row.get("*营业总收入"))
                total_cost = parse_amount(row.get("*营业总成本"))
                net_cash_from_operating = parse_amount(row.get("*经营活动产生的现金流量净额"))
                cash_for_fixed_assets = parse_amount(row.get("购建固定资产、无形资产和其他长期资产支付的现金"))
                
                record = FundamentalData(
                    stock_code = stock_code,
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
                    self.fundamental_dao.batch_insert(fundamental_records)
                    logger.info("Inserted %d new fundamental records for stock %s.", len(fundamental_records), stock_code)
                except Exception as e:
                    logger.error("Error inserting fundamental records for stock %s: %s", stock_code, e)
            else:
                logger.info("No new fundamental records to insert for stock %s.", stock_code)


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
        
    def sync_all(self, date: str):
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
        for _, row in df.iterrows():
            stock_code = row.get("代码")
            suspend_date = row.get("停牌时间")
            # 忽略停牌时间为空的记录
            if suspend_date is None:
                continue
            resume_date = row.get("停牌截止时间")
            suspend_period = row.get("停牌期限")
            suspend_reason = row.get("停牌原因")
            market = row.get("所属市场")
            record = SuspendData(
                stock_code = stock_code,
                suspend_date = suspend_date,
                resume_date = resume_date,
                suspend_period = suspend_period,
                suspend_reason = suspend_reason,
                market = market
            )
            records.append(record)
        if records:
            self.suspend_data_dao.batch_upsert(records)
            logger.info("Full sync: Processed %d suspend data records.", len(records))
        else:
            logger.info("Full sync: No suspend data records to process.")

    def sync_today(self):
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
        for _, row in df.iterrows():
            stock_code = row.get("代码")
            suspend_date = row.get("停牌时间")
            if suspend_date is None:
                continue
            resume_date = row.get("停牌截止时间")
            suspend_period = row.get("停牌期限")
            suspend_reason = row.get("停牌原因")
            market = row.get("所属市场")
            record = SuspendData(
                stock_code = stock_code,
                suspend_date = suspend_date,
                resume_date = resume_date,
                suspend_period = suspend_period,
                suspend_reason = suspend_reason,
                market = market
            )
            records.append(record)
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