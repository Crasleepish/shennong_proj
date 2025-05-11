import pandas as pd
import numpy as np
import statsmodels.api as sm
import akshare as ak
import logging
import sys
import os
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.dao.fund_info_dao import FundInfoDao
from app.dao.stock_info_dao import MarketFactorsDao
from app.data.helper import get_fund_daily_return
from app import create_app

app = create_app()

shibor_data = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="1月")
logger = logging.getLogger(__name__)

def calculate_factor_exposure_daily(fund_daily_returns: pd.DataFrame,
                                     factor_daily_data: pd.DataFrame,
                                     shibor_data: pd.DataFrame) -> pd.Series:
    """
    使用每日数据计算五因子回归暴露
    """
    # 基金数据预处理
    fund = (
        fund_daily_returns[["date", "change_percent"]]
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
        .sort_index()
    )

    # Shibor 转日利率
    rf_daily = (
        shibor_data[["报告日", "利率"]]
        .rename(columns={"报告日": "date", "利率": "rf_rate"})
        .assign(date=lambda x: pd.to_datetime(x["date"]),
                rf_daily_rate=lambda x: x["rf_rate"] / 100 / 252)
        .set_index("date")
        .sort_index()
    )

    # 合并超额收益
    fund_rf = pd.merge(fund, rf_daily, left_index=True, right_index=True, how="left")
    fund_rf["excess_return"] = fund_rf["change_percent"] - fund_rf["rf_daily_rate"]

    # 合并因子数据
    factors = (
        factor_daily_data
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
        .sort_index()
    )

    merged = pd.merge(fund_rf[["excess_return"]], factors, left_index=True, right_index=True, how="inner")
    if merged.empty:
        raise ValueError("基金数据与因子数据无重叠时间区间")

    # 回归分析
    y = merged["excess_return"]
    X = merged[["MKT", "SMB", "HML", "QMJ"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop').fit()
    exposures = model.params
    exposures.name = "factor_exposures"
    return exposures

def regress_one_fund(fund_code: str, factor_daily_data: pd.DataFrame):
    fund_daily_returns = get_fund_daily_return(fund_code)
    if len(fund_daily_returns) < 252:
        return None
    exposures = calculate_factor_exposure_daily(fund_daily_returns, factor_daily_data, shibor_data)
    return exposures

def annualized_return(df, value_col='net_value'):
    start_value = df[value_col].iloc[0]
    end_value = df[value_col].iloc[-1]
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    days = (end_date - start_date).days
    if days == 0:
        return np.nan
    return (end_value / start_value) ** (365 / days) - 1

def annualized_volatility(df, value_col='net_value'):
    df['daily_return'] = df[value_col].ffill().pct_change(fill_method=None)
    return df['daily_return'].std() * np.sqrt(252)

def max_drawdown(df, value_col='net_value'):
    running_max = df[value_col].cummax()
    drawdown = df[value_col] / running_max - 1
    return drawdown.min()

def sharpe_ratio(df, value_col='net_value', risk_free_rate=0):
    ar = annualized_return(df, value_col)
    av = annualized_volatility(df, value_col)
    if av == 0:
        return np.nan
    return (ar - risk_free_rate) / av

def calmar_ratio(df, value_col='net_value'):
    ar = annualized_return(df, value_col)
    md = max_drawdown(df, value_col)
    if md == 0:
        return np.nan
    return ar / abs(md)

def monthly_positive_return_probability(df, value_col='net_value'):
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_return = df.groupby('year_month')[value_col].agg(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    return (monthly_return > 0).mean()

def compute_metrics(fund_code: str):
    df = get_fund_daily_return(fund_code)
    if len(df) < 252:
        return None
    ar = annualized_return(df)
    av = annualized_volatility(df)
    md = max_drawdown(df)
    sr = sharpe_ratio(df)
    cr = calmar_ratio(df)
    mprp = monthly_positive_return_probability(df)
    return ar, av, md, sr, cr, mprp

def main():
    fund_info_dao = FundInfoDao._instance
    market_factors_dao = MarketFactorsDao._instance
    fund_info_df = fund_info_dao.select_dataframe_all()
    factor_df = market_factors_dao.select_dataframe_by_date('2007-01-01', datetime.datetime.today().strftime('%Y-%m-%d'))

    fund_factors_list = []
    for _, fund_info in fund_info_df.iterrows():
        logger.info(f"正在处理{fund_info['fund_code']}")
        fund_code = fund_info["fund_code"]
        exposures = regress_one_fund(fund_code, factor_df)
        if exposures is None:
            continue
        metrics = compute_metrics(fund_code)
        if metrics is None:
            continue
        row_df = (
            exposures.to_frame().T
            .assign(code=fund_code,
                    name=fund_info["fund_name"],
                    ann_return=metrics[0], ann_vol=metrics[1], max_dd=metrics[2],
                    sharpe_ratio=metrics[3], calmar_ratio=metrics[4],
                    mprp=metrics[5])
        )
        fund_factors_list.append(row_df)

    fund_factors_df = pd.concat(fund_factors_list, ignore_index=True)
    fund_factors_df.to_csv("output/fund_factors.csv", index=False)

if __name__ == '__main__':
    with app.app_context():
        main()
