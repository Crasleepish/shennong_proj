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

def compute_asset_growth(code, rb_date, fundamental_df):
    """
        code: 股票代码
        rb_date: 观察日期
        fundamental_df: 财务报表数据DataFrame
    """
    try:
        fundamental = get_latest_fundamental(code, rb_date, fundamental_df, ["total_assets"])
        if fundamental is None:
            return None
        latest_report_date = fundamental["report_date"]
        latest_total_assets = safe_value(fundamental["total_assets"])

        same_period_last_year = pd.Timestamp(year=latest_report_date.year - 1, month=latest_report_date.month, day=latest_report_date.day)
        # previous_fundamental = get_latest_fundamental(code, same_period_last_year, fundamental_df, ["total_assets"])
        previous_fundamentals = fundamental_df[
            (fundamental_df["stock_code"] == code) & 
            (fundamental_df["report_date"] <= same_period_last_year) & 
            (fundamental_df["total_assets"].notna())
            ].sort_values("report_date")
        if previous_fundamentals is None or previous_fundamentals.empty:
            return 0.0
        previous_fundamental = previous_fundamentals.iloc[-1]
        previous_total_assets = safe_value(previous_fundamental["total_assets"])
        if previous_total_assets is None or previous_total_assets == 0:
            return 0.0
        if latest_total_assets is not None and previous_total_assets is not None:
            return (latest_total_assets - previous_total_assets) / previous_total_assets
        else:
            return None
    except Exception as e:
        logger.error("Error calculating OP for stock %s on %s: %s", code, rb_date.strftime("%Y-%m-%d"), str(e))
        return None

def compute_and_cache_asset_growth(rb_date, fundamental_df, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)
    growth_cache = os.path.join(cache_dir, f'asset_growth_{rb_date}.csv')

    stock_df = get_stock_info_df()
    stock_df = stock_df[stock_df["listing_date"] <= rb_date]
    growth_data = []
    growth_dict = {}

    logging.info("Computing asset growth for stocks on %s", rb_date.strftime("%Y-%m-%d"))

    for idx, row in stock_df.iterrows():
        logging.info("Computing asset growth for stock %s", idx)
        code = idx
        industry = row['industry']
        if industry is None:
            continue
        growth = safe_value(compute_asset_growth(code, rb_date, fundamental_df))

        if growth is not None:
            growth_data.append({'stock_code': code, 'industry': industry, 'asset_growth': growth})
            growth_dict[code] = growth

    growth_df = pd.DataFrame(growth_data)
    growth_df.to_csv(growth_cache, index=False)

    return growth_df, growth_dict


def compute_and_cache_industry_avg_asset_growth(growth_df, rb_date, cache_dir='.cache'):
    industry_cache = os.path.join(cache_dir, f'industry_asset_growth_{rb_date}.csv')

    def winsorize(series):
        if len(series) > 2:
            return series.sort_values().iloc[1:-1]
        return series

    industry_grouped = growth_df.groupby('industry')['asset_growth']

    # 缩尾处理后计算均值和标准差
    industry_avg_df = industry_grouped.apply(lambda x: winsorize(x).mean()).reset_index(name='avg_asset_growth')
    industry_std_df = industry_grouped.apply(lambda x: winsorize(x).std()).reset_index(name='std_asset_growth')

    industry_final_df = pd.merge(industry_avg_df, industry_std_df, on='industry')
    industry_final_df.to_csv(industry_cache, index=False)

    industry_avg_growth_dict = industry_final_df.set_index('industry')['avg_asset_growth'].to_dict()
    industry_std_growth_dict = industry_final_df.set_index('industry')['std_asset_growth'].to_dict()

    return industry_avg_growth_dict, industry_std_growth_dict


# 示例调用
# rb_date = '2024-03-31'
# fundamental_df = pd.read_csv('fundamental_data.csv')
# profits_df = compute_and_cache_profitability(rb_date, fundamental_df)
# industry_avg_profit = compute_and_cache_industry_avg_profit(profits_df)
# print(industry_avg_profit)