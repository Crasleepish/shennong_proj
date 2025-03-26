from typing import List, Callable, Any
import pandas as pd
import numpy as np

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