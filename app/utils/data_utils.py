from typing import List, Callable, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_in_batches(data_list: List[Any], 
                       func: Callable[[List[Any]], Any], 
                       batch_size: int = 1000) -> List[Any]:
    """
    将 data_list 按照 batch_size 分成若干子列表，并对每个子列表调用函数 func。
    
    :param data_list: 需要处理的原始列表
    :param func: 处理函数，接受一个列表作为参数，并返回处理结果
    :param batch_size: 每个子列表的最大长度，默认为 1000
    :return: 返回一个列表，包含对每个子列表调用 func 后的返回结果
    """
    results = []
    # 从索引 0 开始，每次切出 batch_size 长度的子列表
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i : i + batch_size]
        result = func(batch)
        results.append(result)
    return results

def calculate_volatility(prices, stock_code, start_date, end_date):
    """
    计算指定股票在给定时间窗口内的波动率（日收益率标准差）
    
    参数:
        prices (pd.DataFrame): 日行情数据，行索引为日期，列索引为股票代码
        stock_code (str): 目标股票代码（必须存在于prices的列中）
        start_date (str): 起始日期（格式：'YYYY-MM-DD'）
        end_date (str): 结束日期（格式：'YYYY-MM-DD'）
        
    返回:
        float: 年化波动率（标准差）
    """
    try:
        # 转换为时间戳并验证日期范围
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if start >= end:
            raise ValueError("结束日期必须晚于起始日期")
            
        # 筛选指定股票和日期范围的数据
        price_series = prices.loc[start:end, stock_code]
        
        if len(price_series) < 2:
            raise ValueError("有效数据不足（至少需要2个交易日）")
        
        # 计算日收益率
        returns = price_series.pct_change().dropna()
        
        # 计算日波动率并年化（假设252个交易日）
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility
        
    except KeyError:
        raise ValueError(f"股票代码 {stock_code} 不存在于数据中")
    except Exception as e:
        raise RuntimeError(f"计算波动率时出错: {str(e)}")
    
def calculate_column_volatility(prices_series):
    """
    计算指定股票在给定时间窗口内的波动率（日收益率标准差）
    
    参数:
        prices_series (pd.Series): 日行情数据列
        start_date (str): 起始日期（格式：'YYYY-MM-DD'）
        end_date (str): 结束日期（格式：'YYYY-MM-DD'）
        
    返回:
        float: 年化波动率（标准差）
    """
    try:
        if len(prices_series) < 2:
            return pd.NaT
        
        # 计算日收益率
        returns = prices_series.pct_change().dropna()
        
        # 计算日波动率并年化（假设252个交易日）
        daily_volatility = returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility
        
    except Exception as e:
        return pd.NaT
    

def calculate_column_illiquidity(returns_series: pd.Series, amount_series: pd.Series) -> float:
    """
    计算Amihud illiquidity指标：日度 |收益率| / 成交额 的均值。
    输入为日度数据，建议使用过去1个月或12个月作为窗口。

    参数:
        returns_series: 日收益率序列（pd.Series）
        amount_series: 成交额序列（pd.Series）
    
    返回:
        float: 平均 illiquidity（越低越流动）
    """
    try:
        illiq = returns_series.abs() / amount_series.shift(1)
        return illiq.dropna().mean()
    except:
        return np.nan


def get_nearest_data_front(prices: pd.DataFrame, refer_date: str):
    """
    prices 是一个索引为日期（Timestamp）的 DataFrame，
    refer_date 是一个日期字符串，格式为 "yyyy-MM-dd"。
    
    返回 prices 中 refer_date 之前的最后一个日期对应的数据行。
    如果没有符合条件的数据，则返回 None。
    """
    # 将日期字符串转换为 Timestamp 对象
    ref_ts = pd.to_datetime(refer_date, format='%Y-%m-%d')
    # 筛选出索引小于 refer_date 的所有行
    subset = prices[prices.index < ref_ts]
    if subset.empty:
        return None
    # 返回最后一行，即最大索引对应的那一行数据
    return subset.iloc[-1]

def change_date(date_str: str, days):
    """
    计算字符串日期的前x天或后x天
    
    参数:
        date_str: 格式为'%Y-%m-%d'的日期字符串
        days: 天数变化，正数表示后x天，负数表示前x天
    
    返回:
        格式为'%Y-%m-%d'的新日期字符串
    """
    # 将字符串转换为datetime对象
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # 计算新的日期
    new_date = date_obj + timedelta(days=days)
    
    # 将新日期转换回字符串格式
    return new_date.strftime('%Y-%m-%d')

def format_date(date_str):
    """
    将日期字符串转换为 YYYYMMDD 格式的字符串。
    
    参数:
        date_str (str): 日期字符串，格式为 "YYYY-MM-DD"
        
    返回:
        str: YYYYMMDD 格式的日期字符串，如果解析失败则返回 "invaliddate"
    """
    try:
        # 尝试解析日期字符串（格式：YYYY-MM-DD）
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # 转换为 YYYYMMDD 格式
        return date_obj.strftime("%Y%m%d")
    except ValueError:
        # 如果解析失败，返回 "invaliddate"
        return "invaliddate"