import pandas as pd
import os
import logging
from collections import defaultdict
from app.data.helper import get_stock_info_df

logger = logging.getLogger(__name__)

def safe_value(val):
    """将 NaN 转换为 None"""
    return None if pd.isna(val) else val

def get_latest_fundamental(stock_code: str, rb_date: pd.Timestamp, fundamental_df: pd.DataFrame, check_fields: list) -> pd.Series:
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

def compute_cash_flow_quality(code, rb_date, fundamental_df):
    """
    计算现金流质量: 经营活动产生的现金流量净额 / 归属于母公司所有者的净利润
    """
    try:
        fundamental = get_latest_fundamental(code, rb_date, fundamental_df,
                                             ["net_cash_from_operating", "net_profit"])
        if fundamental is not None and fundamental["net_profit"] != 0:
            return fundamental["net_cash_from_operating"] / fundamental["net_profit"]
        else:
            return None
    except Exception as e:
        logger.error("Error calculating cash flow quality for stock %s on %s: %s", code, rb_date.strftime("%Y-%m-%d"), str(e))
        return None

def compute_and_cache_cash_flow_quality(stock_universe, rb_date, fundamental_df, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)
    cash_flow_quality_cache = os.path.join(cache_dir, f'cash_flow_quality_{rb_date}.csv')

    stock_df = get_stock_info_df()
    stock_df = stock_df.loc[stock_universe]
    cash_flow_quality_data = []
    cash_flow_quality_dict = {}

    logging.info("Computing cash flow quality for stocks on %s", rb_date.strftime("%Y-%m-%d"))

    for idx, row in stock_df.iterrows():
        logging.info("Computing cash flow quality for stock %s", idx)
        code = idx
        industry = row['industry']
        if industry is None:
            continue
        cash_flow_quality = safe_value(compute_cash_flow_quality(code, rb_date, fundamental_df))

        if cash_flow_quality is not None:
            cash_flow_quality_data.append({'stock_code': code, 'industry': industry, 'cash_flow_quality': cash_flow_quality})
            cash_flow_quality_dict[code] = cash_flow_quality

    cash_flow_quality_df = pd.DataFrame(cash_flow_quality_data)
    cash_flow_quality_df.to_csv(cash_flow_quality_cache, index=False)

    return cash_flow_quality_df, cash_flow_quality_dict

def compute_and_cache_industry_avg_cash_flow_quality(cash_flow_quality_df, rb_date, cache_dir='.cache'):
    industry_cache = os.path.join(cache_dir, f'industry_cash_flow_quality_{rb_date}.csv')

    def winsorize(series):
        if len(series) > 2:
            return series.sort_values().iloc[1:-1]
        return series

    industry_grouped = cash_flow_quality_df.groupby('industry')['cash_flow_quality']

    industry_avg_df = industry_grouped.apply(lambda x: winsorize(x).mean()).reset_index(name='avg_cash_flow_quality')
    industry_std_df = industry_grouped.apply(lambda x: winsorize(x).std()).reset_index(name='std_cash_flow_quality')

    industry_final_df = pd.merge(industry_avg_df, industry_std_df, on='industry')
    industry_final_df.to_csv(industry_cache, index=False)

    industry_avg_cash_flow_quality_dict = industry_final_df.set_index('industry')['avg_cash_flow_quality'].to_dict()
    industry_std_cash_flow_quality_dict = industry_final_df.set_index('industry')['std_cash_flow_quality'].to_dict()

    return industry_avg_cash_flow_quality_dict, industry_std_cash_flow_quality_dict

# 示例调用
# rb_date = pd.Timestamp('2024-03-31')
# fundamental_df = pd.read_csv('fundamental_data.csv')
# cash_flow_quality_df, cash_flow_quality_dict = compute_and_cache_cash_flow_quality(rb_date, fundamental_df)
# industry_avg_cash_flow_quality, industry_std_cash_flow_quality = compute_and_cache_industry_avg_cash_flow_quality(cash_flow_quality_df, rb_date)
# print(industry_avg_cash_flow_quality)
