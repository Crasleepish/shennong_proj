import pandas as pd
import os
import logging
from collections import defaultdict
from app.data.helper import get_stock_info_df

# 假设已有的计算盈利能力函数
# compute_profit(code, rb_date, fundamental_df)
# 假设已有获取股票列表的函数
# select_dataframe_all()

logger = logging.getLogger(__name__)

def safe_value(val):
    """将 NaN 转换为 None"""
    return None if pd.isna(val) else val

def get_latest_fundamental(stock_code: str, rb_date: pd.Timestamp, fundamental_df: pd.DataFrame, check_fields: list) -> pd.Series:
    """
    对于给定股票和再平衡日期，从 fundamental_df 中选取报告期不超过 rb_date 的90天前的最新记录，
    返回一行 Series，包含 total_equity 等基本面数据；若不存在则返回 None。
    """
    latest_report_date = rb_date - pd.Timedelta(days=120)
    df = fundamental_df[fundamental_df["stock_code"] == stock_code]
    df = df[df["report_date"] <= latest_report_date]
    if df.empty:
        return None
    if check_fields is not None:
        for field in check_fields:
            df = df[df[field].notna()]
    if df.empty:
        return None
    return df.sort_values("report_date").iloc[-1]

def get_ttm_value(stock_code: str, rb_date: pd.Timestamp, fundamental_df: pd.DataFrame, field: str) -> float:
    """
    1. 循环遍历满足日期条件的财报（从新到旧）
    2. 对每个财报尝试计算 TTM
    3. 若某个财报满足条件则返回，否则继续
    """
    # 获取所有满足日期条件的财报（按日期倒序排列）
    latest_report_date = rb_date - pd.Timedelta(days=120)
    candidate_reports = fundamental_df[
        (fundamental_df["stock_code"] == stock_code) &
        (fundamental_df["report_date"] <= latest_report_date) &
        (fundamental_df["net_profit"].notna())
    ].sort_values("report_date", ascending=False)
    
    if candidate_reports.empty:
        return None
    
    # 遍历所有候选财报
    for _, report in candidate_reports.iterrows():
        report_date = report["report_date"]
        
        # Case 1: 如果是 Q4 财报，直接返回
        if report_date.month == 12:
            return report[field]
        
        # Case 2: 非 Q4 财报，需要计算 TTM
        current_year = report_date.year
        current_month = report_date.month
        
        # 检查上一年年报是否存在
        last_annual_date = pd.Timestamp(year=current_year-1, month=12, day=31)
        last_annual_report = candidate_reports[
            (candidate_reports["stock_code"] == stock_code) &
            (candidate_reports["report_date"] == last_annual_date)
        ]
        if last_annual_report.empty:
            continue  # 缺失年报，跳过此财报
        
        # 检查去年同期累计值（可能需向前查找多个季度）
        same_period_last_year = pd.Timestamp(year=current_year-1, month=report_date.month, day=report_date.day)
        same_period_reports = candidate_reports[
            (candidate_reports["stock_code"] == stock_code) &
            (candidate_reports["report_date"] <= same_period_last_year) &
            (candidate_reports["report_date"].dt.month == current_month)
        ]
        
        if same_period_reports.empty:
            continue  # 缺失同期数据，跳过此财报
        
        # 取最近一期去年同期累计值
        last_period_value = same_period_reports.sort_values("report_date", ascending=False).iloc[0][field]
        
        # 计算 TTM
        ttm_value = report[field] + last_annual_report.iloc[0][field] - last_period_value
        return ttm_value
    
    # 所有候选财报均不满足条件
    return None

def compute_profit(code, rb_date, fundamental_df):
    """
        code: 股票代码
        rb_date: 观察日期
        fundamental_df: 财务报表数据DataFrame
    """
    try:
        fundamental = get_latest_fundamental(code, rb_date, fundamental_df, ["total_equity"])
        net_profit = safe_value(get_ttm_value(code, rb_date, fundamental_df, "net_profit"))
        if fundamental is not None and net_profit is not None and fundamental["total_equity"] != 0:
            return net_profit / fundamental["total_equity"]
        else:
            return None
    except Exception as e:
        logger.error("Error calculating OP for stock %s on %s: %s", code, rb_date.strftime("%Y-%m-%d"), str(e))
        return None

def compute_and_cache_profitability(rb_date, fundamental_df, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)
    profitability_cache = os.path.join(cache_dir, f'profitability_{rb_date}.csv')

    stock_df = get_stock_info_df()
    stock_df = stock_df[stock_df["listing_date"] <= rb_date]
    profits_data = []
    profits_dict = {}

    logging.info("Computing profitability for stocks on %s", rb_date.strftime("%Y-%m-%d"))

    for idx, row in stock_df.iterrows():
        logging.info("Computing profitability for stock %s", idx)
        code = idx
        industry = row['industry']
        if industry is None:
            continue
        profit = safe_value(compute_profit(code, rb_date, fundamental_df))

        if profit is not None:
            profits_data.append({'stock_code': code, 'industry': industry, 'profitability': profit})
            profits_dict[code] = profit

    profits_df = pd.DataFrame(profits_data)
    profits_df.to_csv(profitability_cache, index=False)

    return profits_df, profits_dict

def compute_and_cache_industry_avg_profit(profits_df, rb_date, cache_dir='.cache'):
    industry_cache = os.path.join(cache_dir, f'industry_profits_{rb_date}.csv')

    def winsorize(series):
        if len(series) > 2:
            return series.sort_values().iloc[1:-1]
        return series

    industry_grouped = profits_df.groupby('industry')['profitability']

    # 缩尾后再分别计算平均值和标准差
    industry_avg_df = industry_grouped.apply(lambda x: winsorize(x).mean()).reset_index(name='avg_profitability')
    industry_std_df = industry_grouped.apply(lambda x: winsorize(x).std()).reset_index(name='std_profitability')

    industry_final_df = pd.merge(industry_avg_df, industry_std_df, on='industry')
    industry_final_df.to_csv(industry_cache, index=False)

    industry_avg_profit_dict = industry_final_df.set_index('industry')['avg_profitability'].to_dict()
    industry_std_profit_dict = industry_final_df.set_index('industry')['std_profitability'].to_dict()

    return industry_avg_profit_dict, industry_std_profit_dict

# 示例调用
# rb_date = '2024-03-31'
# fundamental_df = pd.read_csv('fundamental_data.csv')
# profits_df = compute_and_cache_profitability(rb_date, fundamental_df)
# industry_avg_profit = compute_and_cache_industry_avg_profit(profits_df)
# print(industry_avg_profit)