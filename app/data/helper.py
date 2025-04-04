from app.dao.stock_info_dao import StockHistAdjDao, StockInfoDao, FundamentalDataDao, SuspendDataDao
from app.dao.index_info_dao import IndexInfoDao, IndexHistDao
from app.dao.fund_info_dao import FundInfoDao, FundHistDao
import pandas as pd

class StockHistAdjHolder:

    stock_hist_adj_all = None

    def __init__(self):
        pass

    def get_stock_hist_adj_all(self):
        if self.stock_hist_adj_all is None:
            stock_hist_adj_dao = StockHistAdjDao._instance
            self.stock_hist_adj_all = stock_hist_adj_dao.select_dataframe_all()
            self.stock_hist_adj_all["date"] = pd.to_datetime(self.stock_hist_adj_all["date"], errors="coerce")
        return self.stock_hist_adj_all

stock_hist_adj_holder = StockHistAdjHolder()

class StockInfoHolder:

    stock_info_all = None

    def __init__(self):
        pass

    def get_stock_info_all(self):
        if self.stock_info_all is None:
            stock_info_dao = StockInfoDao._instance
            self.stock_info_all = stock_info_dao.select_dataframe_all()
            self.stock_info_all["listing_date"] = pd.to_datetime(self.stock_info_all["listing_date"], errors="coerce")
        return self.stock_info_all
    
stock_info_holder = StockInfoHolder()

class FundamentalDataHolder:
    fundamental_data_all = None

    def __init__(self):
        pass

    def get_fundamental_data_all(self):
        if self.fundamental_data_all is None:
            fundamental_data_dao = FundamentalDataDao._instance
            self.fundamental_data_all = fundamental_data_dao.select_dataframe_all()
            self.fundamental_data_all["report_date"] = pd.to_datetime(self.fundamental_data_all["report_date"], errors="coerce")
        return self.fundamental_data_all
    
fundamental_data_holder = FundamentalDataHolder()

class SuspendDataHolder:
    suspend_data_all = None
    def __init__(self):
        pass

    def get_suspend_data_all(self):
        if self.suspend_data_all is None:
            suspend_data_dao = SuspendDataDao._instance
            self.suspend_data_all = suspend_data_dao.select_dataframe_all()
            self.suspend_data_all["suspend_date"] = pd.to_datetime(self.suspend_data_all["suspend_date"], errors="coerce")
            self.suspend_data_all["resume_date"] = pd.to_datetime(self.suspend_data_all["resume_date"], errors="coerce")
        return self.suspend_data_all
    
suspend_data_holder = SuspendDataHolder()

class IndexHistHolder:

    index_hist_all = None

    def __init__(self):
        pass

    def get_index_hist_all(self):
        if self.index_hist_all is None:
            index_hist_dao = IndexHistDao._instance
            self.index_hist_all = index_hist_dao.select_dataframe_all()
            self.index_hist_all["date"] = pd.to_datetime(self.index_hist_all["date"], errors="coerce")
        return self.index_hist_all
    
    def get_index_hist_by_code(self, index_code: str) -> pd.DataFrame:
        index_hist_dao = IndexHistDao._instance
        return index_hist_dao.select_dataframe_by_code(index_code)

index_hist_holder = IndexHistHolder()


class FundHistHolder:

    fund_hist_all = None

    def __init__(self):
        pass

    def get_fund_hist_all(self):
        if self.fund_hist_all is None:
            fund_hist_dao = FundHistDao._instance
            self.fund_hist_all = fund_hist_dao.select_dataframe_all()
            self.fund_hist_all["date"] = pd.to_datetime(self.fund_hist_all["date"], errors="coerce")
        return self.fund_hist_all
    
    def get_fund_hist_by_code(self, fund_code: str) -> pd.DataFrame:
        fund_hist_dao = FundHistDao._instance
        return fund_hist_dao.select_dataframe_by_code(fund_code)

fund_hist_holder = FundHistHolder()


def get_prices_df() -> pd.DataFrame:
    """
    返回股票历史价格数据。
    
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
    df_all = stock_hist_adj_holder.get_stock_hist_adj_all()
    pivot_df = df_all.pivot(index="date", columns="stock_code", values="close")
    pivot_df = pivot_df.ffill()
    return pivot_df

def get_volume_df() -> pd.DataFrame:
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
    df_all = stock_hist_adj_holder.get_stock_hist_adj_all()
    pivot_df = df_all.pivot(index="date", columns="stock_code", values="volume")
    pivot_df = pivot_df.fillna(0)
    return pivot_df

def get_mkt_cap_df() -> pd.DataFrame:
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
    df_all = stock_hist_adj_holder.get_stock_hist_adj_all()
    pivot_df = df_all.pivot(index="date", columns="stock_code", values="mkt_cap")
    pivot_df = pivot_df.ffill()
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
    df = stock_info_holder.get_stock_info_all()
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
    df = fundamental_data_holder.get_fundamental_data_all()
    return df

def get_suspend_df() -> pd.DataFrame:
    """
    返回停牌数据。
    
    DataFrame 格式要求：
      - 包含字段：stock_code, suspend_date, resume_date, suspend_period, suspend_reason, market 等。
      - suspend_date 和 resume_date 为 datetime64[ns] 类型
     
    样例输出：
         stock_code  suspend_date  resume_date suspend_period  suspend_reason      market
    0       600012   2020-06-15   2020-06-16      "1天"         "公告停牌"      Main Board
    1       600016   2021-01-10        NaT         "连续停牌"     "重要公告"      Main Board
    2       600018   2020-12-20   2020-12-22      "3天"         "公告停牌"      GEM
    ...
    """
    df = suspend_data_holder.get_suspend_data_all()
    return df

def get_index_daily_return(index_code: str) -> pd.DataFrame:
    """
    返回指数历史每日回报率：
    包含字段：date, daily_return
    """
    df = index_hist_holder.get_index_hist_by_code(index_code)[['date', 'change_percent']]
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    df['change_percent'] = df['change_percent'] / 100.0
    df = df.rename(columns={'change_percent': 'daily_return'})
    df = df.set_index('date', drop=True)
    return df

def get_fund_daily_return(fund_code: str) -> pd.DataFrame:
    """
    返回基金历史每日回报率：
    包含字段：date, daily_return
    """
    df = fund_hist_holder.get_fund_hist_by_code(fund_code)
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    return df