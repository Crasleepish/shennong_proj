import pandas as pd
import os
import logging
from collections import defaultdict
from app.data.helper import get_stock_info_df

logger = logging.getLogger(__name__)

def safe_value(val):
    """将 NaN 转换为 None"""
    return None if pd.isna(val) else val

def safe_num(val):
    return 0 if pd.isna(val) else val

def get_latest_fundamental(stock_code: str, rb_date: pd.Timestamp, fundamental_df: pd.DataFrame, all_not_none: list, exsit_not_none: list) -> pd.Series:
    latest_report_date = rb_date - pd.Timedelta(days=120)
    df = fundamental_df[fundamental_df["stock_code"] == stock_code]
    df = df[df["report_date"] <= latest_report_date]
    if df.empty:
        return None
    if all_not_none is not None:
        for field in all_not_none:
            df = df[df[field].notna()]
    if exsit_not_none is not None:
        mask = df[exsit_not_none].notna().any(axis=1)
        df = df[mask]
    if df.empty:
        return None
    return df.sort_values("report_date").iloc[-1]

def compute_leverage_ratio(code, rb_date, fundamental_df):
    """
    计算资产负债率: (流动负债合计 + 非流动负债合计) / 资产合计
    """
    try:
        fundamental = get_latest_fundamental(code, rb_date, fundamental_df,
                                             ["total_assets"], ["current_liabilities", "noncurrent_liabilities"])
        if fundamental is not None and fundamental["total_assets"] != 0:
            liabilities = safe_num(fundamental["current_liabilities"]) + safe_num(fundamental["noncurrent_liabilities"])
            return liabilities / fundamental["total_assets"]
        else:
            return None
    except Exception as e:
        logger.error("Error calculating leverage ratio for stock %s on %s: %s", code, rb_date.strftime("%Y-%m-%d"), str(e))
        return None

def compute_and_cache_leverage(rb_date, fundamental_df, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)
    leverage_cache = os.path.join(cache_dir, f'leverage_{rb_date}.csv')

    stock_df = get_stock_info_df()
    stock_df = stock_df[stock_df["listing_date"] <= rb_date]
    leverage_data = []
    leverage_dict = {}

    logging.info("Computing leverage ratio for stocks on %s", rb_date.strftime("%Y-%m-%d"))

    for idx, row in stock_df.iterrows():
        logging.info("Computing leverage ratio for stock %s", idx)
        code = idx
        industry = row['industry']
        if industry is None:
            continue
        leverage = safe_value(compute_leverage_ratio(code, rb_date, fundamental_df))

        if leverage is not None:
            leverage_data.append({'stock_code': code, 'industry': industry, 'leverage_ratio': leverage})
            leverage_dict[code] = leverage

    leverage_df = pd.DataFrame(leverage_data)
    leverage_df.to_csv(leverage_cache, index=False)

    return leverage_df, leverage_dict

def compute_and_cache_industry_avg_leverage(leverage_df, rb_date, cache_dir='.cache'):
    industry_cache = os.path.join(cache_dir, f'industry_leverage_{rb_date}.csv')

    def winsorize(series):
        if len(series) > 2:
            return series.sort_values().iloc[1:-1]
        return series

    industry_grouped = leverage_df.groupby('industry')['leverage_ratio']

    industry_avg_df = industry_grouped.apply(lambda x: winsorize(x).mean()).reset_index(name='avg_leverage_ratio')
    industry_std_df = industry_grouped.apply(lambda x: winsorize(x).std()).reset_index(name='std_leverage_ratio')

    industry_final_df = pd.merge(industry_avg_df, industry_std_df, on='industry')
    industry_final_df.to_csv(industry_cache, index=False)

    industry_avg_leverage_dict = industry_final_df.set_index('industry')['avg_leverage_ratio'].to_dict()
    industry_std_leverage_dict = industry_final_df.set_index('industry')['std_leverage_ratio'].to_dict()

    return industry_avg_leverage_dict, industry_std_leverage_dict

# 示例调用
# rb_date = pd.Timestamp('2024-03-31')
# fundamental_df = pd.read_csv('fundamental_data.csv')
# leverage_df, leverage_dict = compute_and_cache_leverage(rb_date, fundamental_df)
# industry_avg_leverage, industry_std_leverage = compute_and_cache_industry_avg_leverage(leverage_df, rb_date)
# print(industry_avg_leverage)
