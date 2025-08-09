from app.dao.stock_info_dao import StockHistUnadjDao, AdjFactorDao, StockInfoDao, FundamentalDataDao, SuspendDataDao
from app.dao.index_info_dao import IndexInfoDao, IndexHistDao
from app.dao.fund_info_dao import FundInfoDao, FundHistDao
import pandas as pd
from typing import List
from app.utils.data_utils import change_date
import logging

logger = logging.getLogger(__name__)

class StockHistHolder:

    stock_hist_all = None

    def __init__(self):
        pass

    def get_stock_hist(self, start_date, end_date):
        logging.warning("This method is deprecated.")
        if self.stock_hist_all is None:
            self.stock_hist_all = StockHistUnadjDao.select_dataframe_by_date_range(stock_code=None, start_date=start_date, end_date=end_date)
            self.stock_hist_all["date"] = pd.to_datetime(self.stock_hist_all["date"], errors="coerce")
        return self.stock_hist_all
    
stock_hist_holder = StockHistHolder()

class AdjFactorHolder:

    adj_factor_all = None

    def __init__(self):
        pass

    def get_adj_factor(self, start_date, end_date):
        logging.warning("This method is deprecated.")
        if self.adj_factor_all is None:
            adj_factor_dao = AdjFactorDao._instance
            self.adj_factor_all = adj_factor_dao.get_adj_factor_dataframe(stock_code=None, start_date=start_date, end_date=end_date)
            self.adj_factor_all["date"] = pd.to_datetime(self.adj_factor_all["date"], errors="coerce")
        return self.adj_factor_all
    
adj_factor_holder = AdjFactorHolder()

class StockInfoHolder:

    stock_info_all = None

    def __init__(self):
        pass

    def get_stock_info_all(self):
        logging.warning("This method is deprecated.")
        if self.stock_info_all is None:
            self.stock_info_all = StockInfoDao.select_dataframe_all()
            self.stock_info_all["listing_date"] = pd.to_datetime(self.stock_info_all["listing_date"], errors="coerce")
        return self.stock_info_all
    
stock_info_holder = StockInfoHolder()

class FundamentalDataHolder:
    fundamental_data_all = None

    def __init__(self):
        pass

    def get_fundamental_data_all(self):
        logging.warning("This method is deprecated.")
        if self.fundamental_data_all is None:
            self.fundamental_data_all = FundamentalDataDao.select_dataframe_all()
            self.fundamental_data_all["report_date"] = pd.to_datetime(self.fundamental_data_all["report_date"], errors="coerce")
        return self.fundamental_data_all
    
fundamental_data_holder = FundamentalDataHolder()

class SuspendDataHolder:
    suspend_data_all = None

    def __init__(self):
        pass

    def get_suspend_data_all(self):
        logging.warning("This method is deprecated.")
        if self.suspend_data_all is None:
            suspend_data_dao = SuspendDataDao._instance
            self.suspend_data_all = suspend_data_dao.select_dataframe_all()

            # 确保字段存在再进行转换
            if 'trade_date' in self.suspend_data_all.columns:
                self.suspend_data_all["trade_date"] = pd.to_datetime(self.suspend_data_all["trade_date"], errors="coerce")
        return self.suspend_data_all

suspend_data_holder = SuspendDataHolder()

class IndexHistHolder:

    index_hist_all = None

    def __init__(self):
        pass

    def get_index_hist_all(self):
        logging.warning("This method is deprecated.")
        if self.index_hist_all is None:
            index_hist_dao = IndexHistDao._instance
            self.index_hist_all = index_hist_dao.select_dataframe_all()
            self.index_hist_all["date"] = pd.to_datetime(self.index_hist_all["date"], errors="coerce")
        return self.index_hist_all
    
    def get_index_hist_by_code(self, index_code: str) -> pd.DataFrame:
        logging.warning("This method is deprecated.")
        index_hist_dao = IndexHistDao._instance
        return index_hist_dao.select_dataframe_by_code(index_code)

index_hist_holder = IndexHistHolder()


class FundHistHolder:

    fund_hist_all = None

    def __init__(self):
        pass

    def get_fund_hist_all(self):
        logging.warning("This method is deprecated.")
        if self.fund_hist_all is None:
            fund_hist_dao = FundHistDao._instance
            self.fund_hist_all = fund_hist_dao.select_dataframe_all()
            self.fund_hist_all["date"] = pd.to_datetime(self.fund_hist_all["date"], errors="coerce")
        return self.fund_hist_all
    
    def get_fund_hist_by_code(self, fund_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        logging.warning("This method is deprecated.")
        fund_hist_dao = FundHistDao._instance
        return fund_hist_dao.select_dataframe_by_code(fund_code, start_date, end_date)

fund_hist_holder = FundHistHolder()

def to_adjusted_hist(unadj_df: pd.DataFrame, adj_factor_df: pd.DataFrame, hist_columns: List[str], adj_factor_column: str, date_column: str):
    """
    将未复权的历史数据转换为复权后的历史数据。
    
    :param unadj_df: 未复权的历史数据
    :param adj_factor_df: 复权因子数据
    :param hist_columns: 历史数据的列名，如 ["open", "close", "high", "low", "volume", "amount"]
    :param adj_factor_column: 复权因子数据的列名，如 "adj_factor"
    :param date_column: 日期数据的列名，如 "trade_date"
    :return: 复权后的历史数据
    """
    # 复制原始数据，避免修改原始数据
    result_df = unadj_df.copy()
    
    # 确定基准复权因子（最新日期的复权因子）
    adj_factor_df_fill = adj_factor_df.ffill()
    latest_date = adj_factor_df_fill[date_column].max()
    base_adj_factor = adj_factor_df_fill.loc[adj_factor_df_fill[date_column] == latest_date, adj_factor_column].iloc[0]
    
    # 将复权因子数据合并到历史数据中
    merged_df = result_df.merge(adj_factor_df_fill[[date_column, adj_factor_column]], 
                               on=date_column, how='left')
    
    # 计算调整比率
    merged_df['adj_ratio'] = merged_df[adj_factor_column] / base_adj_factor
    
    # 调整价格列
    for column in hist_columns:
        if column in merged_df.columns:
            merged_df[column] = merged_df[column] * merged_df['adj_ratio']
    
    # 删除合并后添加的列
    result_df = merged_df.drop([adj_factor_column, 'adj_ratio'], axis=1)
    
    return result_df

def get_qfq_price_by_code(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    返回指定股票的每日价格数据(前复权)。
    """
    adj_factor_dao = AdjFactorDao._instance
    hist = StockHistUnadjDao.select_dataframe_by_date_range(stock_code, start_date, end_date)
    adjf = adj_factor_dao.get_adj_factor_dataframe(stock_code, start_date, end_date)
    adj_hist = to_adjusted_hist(hist, adjf, ["open", "high", "low", "close"], "adj_factor", "date")
    adjf["pre_adj_factor"] = adjf["adj_factor"].shift(1)
    adjf["pre_adj_factor"] = adjf["pre_adj_factor"].bfill()
    adjf = pd.concat([adjf, pd.DataFrame([{"date": change_date(end_date, 1), "pre_adj_factor": adjf.iloc[-1]["adj_factor"]}])], ignore_index=True)
    adj_hist["pre_close"] = to_adjusted_hist(hist, adjf, ["pre_close"], "pre_adj_factor", "date")["pre_close"]
    adj_hist["change"] = adj_hist["close"] - adj_hist["pre_close"]
    adj_hist["pct_chg"] = (adj_hist["close"] - adj_hist["pre_close"]) / adj_hist["pre_close"] * 100
    adj_hist = adj_hist[adj_hist["date"] >= start_date]
    return adj_hist

def get_prices_df(start_date: str, end_date: str) -> pd.DataFrame:
    """
    返回股票历史价格数据，使用向量化操作实现前复权。
    
    DataFrame 格式要求：
      - 索引为交易日期（datetime64[ns]）
      - 列为股票代码
      - 值为股票的收盘价（或其它价格，根据需求）
      
    样例输出：
                600012   600016   600018
    Date                                
    2021-01-04   10.20    15.30    8.45
    2021-01-05   10.40    15.50    8.50
    2021-01-06   10.35    15.40    8.55
    ...
    """
    # 获取原始数据并转换为数据透视表
    df_hist = StockHistUnadjDao.select_dataframe_by_date_range(stock_code=None, start_date=start_date, end_date=end_date)
    df_hist["date"] = pd.to_datetime(df_hist["date"], errors="coerce")
    pivot_df_hist = df_hist.pivot(index="date", columns="stock_code", values="close")
    pivot_df_hist = pivot_df_hist.sort_index(ascending=True)
    
    df_adjf = AdjFactorDao._instance.get_adj_factor_dataframe(stock_code=None, start_date=start_date, end_date=end_date)
    df_adjf["date"] = pd.to_datetime(df_adjf["date"], errors="coerce")
    pivot_adjf = df_adjf.pivot(index="date", columns="stock_code", values="adj_factor")
    pivot_adjf = pivot_adjf.sort_index(ascending=True)
    
    # 确保两个数据框的列(股票代码)一致
    if set(pivot_df_hist.columns) != set(pivot_adjf.columns):
        logger.warning("Columns of pivot_df_hist and pivot_adjf are not equal. Using intersection ====> %s.", pivot_df_hist.columns.difference(pivot_adjf.columns))
    common_stocks = pivot_df_hist.columns.intersection(pivot_adjf.columns)
    pivot_df_hist = pivot_df_hist[common_stocks]
    pivot_adjf = pivot_adjf[common_stocks]
    
    # 确保索引对齐
    pivot_adjf = pivot_adjf.reindex(pivot_df_hist.index)
    pivot_adjf = pivot_adjf.ffill()
    
    # 获取每只股票的最新复权因子(最后一行)
    latest_adj_factors = pivot_adjf.iloc[-1]
    
    # 广播操作：将最新复权因子转换为与复权因子数据框相同形状的矩阵
    # 使用 outer division 计算调整比率
    adj_ratios = pivot_adjf.div(latest_adj_factors, axis='columns')
    
    # 一次性计算所有股票的前复权价格
    adjusted_prices = pivot_df_hist * adj_ratios
    
    # 确保日期索引是datetime64[ns]类型
    adjusted_prices.index = pd.to_datetime(adjusted_prices.index)
    
    return adjusted_prices

def get_volume_df(start_date: str, end_date: str) -> pd.DataFrame:
    """
    返回成交量数据。
    
    DataFrame 格式要求：
      - 索引为交易日期（datetime64[ns]）
      - 列为股票代码
      - 值为股票的成交量
     
    样例输出：
                600012     600016     600018
    Date                                
    2021-01-04   1000000   2000000    1500000
    2021-01-05   1100000   2100000    1600000
    2021-01-06   1050000   2050000    1550000
    ...
    """
    df_all = StockHistUnadjDao.select_dataframe_by_date_range(stock_code=None, start_date=start_date, end_date=end_date)
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    pivot_df = df_all.pivot(index="date", columns="stock_code", values="volume")
    pivot_df = pivot_df.fillna(0)
    pivot_df = pivot_df.sort_index(ascending=True)
    return pivot_df

def get_amount_df(start_date: str, end_date: str) -> pd.DataFrame:
    """
    返回成交额数据。
    
    DataFrame 格式要求：
      - 索引为交易日期（datetime64[ns]）
      - 列为股票代码
      - 值为股票的成交额（单位：元）
     
    样例输出：
                 600012      600016       600018
    Date                                
    2021-01-04   305350549   425133654    386441924
    2021-01-05   152483088   144274065    216927102
    2021-01-06   170145432   128599108    180497477
    ...
    """
    df_all = StockHistUnadjDao.select_dataframe_by_date_range(stock_code=None, start_date=start_date, end_date=end_date)
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    pivot_df = df_all.pivot(index="date", columns="stock_code", values="amount")
    pivot_df = pivot_df.sort_index(ascending=True)
    return pivot_df

def get_return_df(start_date: str, end_date: str) -> pd.DataFrame:
    """
    返回每日收益率数据。
    
    DataFrame 格式要求：
      - 索引为交易日期（datetime64[ns]）
      - 列为股票代码
      - 值为股票的收益率
     
    样例输出：
                 600012      600016       600018
    Date                                
    2021-01-04   0.01        0.01         0.01
    2021-01-05   0.02        0.02         0.02
    2021-01-06   0.12        0.12         0.12
    ...
    """
    df_prices = get_prices_df(change_date(start_date, -10), end_date)
    df_return = df_prices.ffill().pct_change(fill_method=None)
    df_return = df_return[df_return.index >= start_date]
    return df_return

def get_mkt_cap_df(start_date: str, end_date: str) -> pd.DataFrame:
    """
    返回股票市值数据。
    
    DataFrame 格式要求：
      - 索引为交易日期（datetime64[ns]）
      - 列为股票代码
      - 值为股票的市值（单位：元或亿，根据实际情况统一）
      
    样例输出：
                600012      600016      600018
    Date                                
    2021-01-04   100000000  150000000   120000000
    2021-01-05   101000000  152000000   121000000
    2021-01-06   102000000  153000000   122000000
    ...
    """
    df_all = StockHistUnadjDao.select_dataframe_by_date_range(stock_code=None, start_date=start_date, end_date=end_date)
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    pivot_df = df_all.pivot(index="date", columns="stock_code", values="mkt_cap")
    pivot_df = pivot_df.sort_index(ascending=True)
    return pivot_df

def get_stock_info_df() -> pd.DataFrame:
    """
    返回股票基本信息数据，包括股票代码、上市日期、所属市场等。
    
    DataFrame 格式要求：
      - 索引为股票代码（字符串）
      - 至少包含一列 listing_date（datetime64[ns]），以及市场信息，例如 market
     
    样例输出：
                listing_date     market
    stock_code                        
    600012       1990-05-01      Main Board
    600016       1988-06-15      Main Board
    600018       2000-09-30      GEM
    ...
    """
    df = StockInfoDao.select_dataframe_all()
    df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
    df = df.set_index("stock_code", drop=True)
    return df

def get_fundamental_df() -> pd.DataFrame:
    """
    返回基本面数据。
    
    DataFrame 格式要求：
      - 包含字段：stock_code, report_date, total_equity, total_assets, current_liabilities, 
        noncurrent_liabilities, net_profit, operating_profit, total_revenue, total_cost,
        net_cash_from_operating, cash_for_fixed_assets 等。
      - report_date 列为 datetime64[ns] 类型
      - 数值字段单位需要与市值数据统一（例如 total_equity 以亿为单位）
      
    样例输出：
       stock_code  report_date  total_equity  total_assets  current_liabilities  noncurrent_liabilities  net_profit  operating_profit  total_revenue  total_cost  net_cash_from_operating  cash_for_fixed_assets
    0     600012  2020-12-31          50.0         200.0                 80.0                    60.0       10.5               8.0         150.0      100.0                    12.5                   5.0
    1     600012  2019-12-31          48.0         190.0                 75.0                    58.0       10.0               7.5         145.0       98.0                    12.0                   4.8
    ...
    """
    df = FundamentalDataDao.select_dataframe_all()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    return df

def get_index_daily_return(index_code: str) -> pd.DataFrame:
    """
    返回指数历史每日回报率：
    包含字段：date, daily_return
    """
    df = IndexHistDao._instance.select_dataframe_by_code(index_code)[['date', 'change_percent']]
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df['change_percent'] = df['change_percent'] / 100.0
    df = df.rename(columns={'change_percent': 'daily_return'})
    df = df.set_index('date', drop=True)
    return df

def get_fund_daily_return(fund_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    返回基金历史每日回报率：
    包含字段：date, daily_return
    """
    df = FundHistDao._instance.select_dataframe_by_code(fund_code, start_date, end_date)
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df = df.sort_values('date')
    df = df.set_index('date', drop=False)
    return df

def get_fund_prices_by_code_list(code_list: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    返回基金的历史净值数据。
    
    DataFrame 格式要求：
      - 索引为交易日期（datetime64[ns]）
      - 列为基金代码
      - 值为基金净值
      
    样例输出：
                600012   600016   600018
    Date                                
    2021-01-04   10.20    15.30    8.45
    2021-01-05   10.40    15.50    8.50
    2021-01-06   10.35    15.40    8.55
    ...
    """
    fund_hist_dao = FundHistDao._instance
    df_list = []
    for fund_code in code_list:
        df = fund_hist_dao.select_dataframe_by_code(fund_code)
        df['date'] = pd.to_datetime(df['date'], errors="coerce")
        df = df.sort_values('date')
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date) ]
        df = df.reset_index(drop=True)
        df_list.append(df)
    
    df_all = pd.concat(df_list, axis=0)
    df_all = df_all.reset_index(drop=True)
    pivot_df = df_all.pivot(index="date", columns="fund_code", values="net_value")
    pivot_df = pivot_df.ffill()
    pivot_df = pivot_df.dropna()
    return pivot_df

def get_fund_current_prices_by_code_list(code_list: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    返回基金的当前价格数据。
    
    DataFrame 格式要求：
      - 索引为交易日期（datetime64[ns]）
      - 列为基金代码
      - 值为基金价格
      
    样例输出：
                600012   600016   600018
    Date                                
    2021-01-04   10.20    15.30    8.45
    2021-01-05   10.40    15.50    8.50
    2021-01-06   10.35    15.40    8.55
    ...
    """
    fund_hist_dao = FundHistDao._instance
    df_list = []
    for fund_code in code_list:
        df = fund_hist_dao.select_dataframe_by_code(fund_code)
        df['date'] = pd.to_datetime(df['date'], errors="coerce")
        df = df.sort_values('date')
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date) ]
        df = df.reset_index(drop=True)
        df_list.append(df)
    
    df_all = pd.concat(df_list, axis=0)
    df_all = df_all.reset_index(drop=True)
    pivot_df = df_all.pivot(index="date", columns="fund_code", values="value")
    pivot_df = pivot_df.ffill()
    pivot_df = pivot_df.dropna()
    return pivot_df


def get_fund_fees_by_code_list(code_list: list):
    """
    返回基金的费用数据dict。
    dict 格式要求：
      - key 为基金代码
      - value 为基金费用数据
    """
    fees_dict = {}
    fund_info_dao = FundInfoDao._instance
    for fund_code in code_list:
        fund_info_df = fund_info_dao.select_dataframe_by_code([fund_code])
        fund_info_dict = fund_info_df.iloc[0].to_dict()
        fees_dict[fund_code] = fund_info_dict['fee_rate'] / 100.0
    return fees_dict
    
    
def get_stock_status_map() -> pd.DataFrame:
    """
    获取所有股票的上市状态列表，返回 DataFrame，索引为 stock_code，值为 list_status（'L' 或 'D'）
    """
    try:
        df = StockInfoDao.select_dataframe_all()

        # 只保留必要列并设置索引
        if not df.empty:
            df = df[["stock_code", "list_status"]].dropna()
            df["list_status"] = df["list_status"].str.upper().str.strip()
            return df.set_index("stock_code")
        else:
            return pd.DataFrame(columns=["list_status"])
    except Exception as e:
        return pd.DataFrame(columns=["list_status"])
    

from app.data_fetcher import EtfDataReader


def get_fund_daily_return_for_beta_regression(fund_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    返回基金历史每日回报率：
    包含字段：date, daily_return
    """
    df = fund_hist_holder.get_fund_hist_by_code(fund_code, start_date, end_date)
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df = df.sort_values('date')
    df = df.set_index('date', drop=True)
    df = df[["change_percent"]].rename(columns={"change_percent": "daily_return"})
    return df

def get_etf_daily_return_for_beta_regression(etf_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    返回ETF历史每日回报率：
    包含字段：date, daily_return
    """
    df = EtfDataReader(etf_code, start_date, end_date)
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df = df.sort_values('date')
    df = df.set_index('date', drop=False)
    df = df[["change_percent"]].rename(columns={"change_percent": "daily_return"})
    return df


from app.dao.fund_info_dao import FundInfoDao
from app.data_fetcher.etf_data_fetcher import EtfDataFetcher

def get_all_fund_codes_with_source() -> pd.DataFrame:
    """
    合并 FundInfo 和 EtfInfo，输出所有基金代码及其来源。
    返回：
        DataFrame，包含字段：
        - fund_code: 基金代码（如 510300.SH）
        - source: "fund_info" 或 "etf_info"
    """
    try:
        # 从 FundInfo 表提取
        df1 = FundInfoDao().select_dataframe_all()
        df1 = df1[["fund_code"]].dropna().copy()
        df1["fund_code"] = df1["fund_code"].astype(str).str.strip()
        df1["source"] = "fund_info"

        # 从 EtfInfo 表提取
        df2 = EtfDataFetcher.get_etf_info()
        df2 = df2[["ts_code"]].rename(columns={"ts_code": "fund_code"}).dropna().copy()
        df2["fund_code"] = df2["fund_code"].astype(str).str.strip()
        df2["source"] = "etf_info"

        # 合并 & 去重
        df_all = pd.concat([df1, df2], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["fund_code"]).reset_index(drop=True)

        return df_all

    except Exception as e:
        return pd.DataFrame(columns=["fund_code", "source"])