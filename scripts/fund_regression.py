import pandas as pd
import numpy as np
import statsmodels.api as sm
import akshare as ak
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.dao.fund_info_dao import FundInfoDao
from app.data.helper import get_fund_daily_return
from app import create_app

app = create_app()

factors_df = pd.read_csv("output/factorsW-FRI.csv", 
                         parse_dates=["date"], 
                         dtype={"MKT": float, "SMB": float, "HML": float, "QMJ": float, "VOL": float})
shibor_data = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="1月")

logger = logging.getLogger(__name__)

def calculate_factor_exposure(fund_daily_returns: pd.DataFrame,
                            factor_weekly_data: pd.DataFrame,
                            shibor_data: pd.DataFrame) -> pd.Series:
    """
    计算基金在五因子模型上的曝险系数（考虑无风险利率和交易日检查）
    
    参数：
        fund_daily_returns : 基金日收益率数据，包含date和change_percent列
        factor_weekly_data : 五因子周度数据，包含date和MKT/SMB/HML/QMJ/VOL列
        shibor_data : Shibor利率数据，包含报告日和利率列
        
    返回：
        pd.Series : 各因子曝险系数（含截距项alpha）
    """
    # ====================== 数据预处理 ======================
    # 1. 处理基金数据（标记是否为交易日）
    fund = (
        fund_daily_returns[["date", "change_percent"]]
        .assign(date=lambda x: pd.to_datetime(x["date"]),
                is_trading_day=1)  # 标记交易日
        .set_index("date")
        .sort_index()
    )
    
    # 2. 处理Shibor数据（转换为日频无风险利率）
    rf_daily = (
        shibor_data[["报告日", "利率"]]
        .rename(columns={"报告日": "date", "利率": "rf_rate"})
        .assign(date=lambda x: pd.to_datetime(x["date"]),
                rf_daily_rate=lambda x: x["rf_rate"] / 100 / 252)  # 年化利率转日利率
        .set_index("date")
        .sort_index()
    )
    
    # 3. 合并基金收益与无风险收益
    fund_rf = (
        pd.merge(fund, rf_daily, left_index=True, right_index=True, how="left")
        .sort_index()
    )
    
    # 4. 计算基金超额收益（日频）
    fund_rf["excess_return"] = fund_rf["change_percent"] - fund_rf["rf_daily_rate"]
    
    # ====================== 周度转换（跳过无交易日的周） ======================
    def safe_weekly_resample(df):
        """安全转换为周度数据（跳过无交易日的周）"""
        return (
            df.assign(growth_factor=df["excess_return"] + 1)
            .resample("W-FRI")
            .apply(lambda x: x["growth_factor"].prod() - 1 
                   if x["is_trading_day"].sum() > 0 else np.nan)
            .dropna()
        )
    
    fund_weekly = safe_weekly_resample(fund_rf).to_frame("fund_excess_return")
    
    # ====================== 因子数据预处理 ======================
    factors = (
        factor_weekly_data
        .assign(date=lambda x: pd.to_datetime(x["date"]))
        .set_index("date")
        .sort_index()
    )
    
    # ====================== 数据合并 ======================
    merged = pd.merge(fund_weekly, factors, left_index=True, right_index=True, how="inner")
    
    if merged.empty:
        raise ValueError("基金数据与因子数据无重叠时间区间")
    
    # ====================== 回归分析 ======================
    y = merged["fund_excess_return"]
    X = merged[["MKT", "SMB", "HML", "QMJ", "VOL"]]
    X = sm.add_constant(X)  # 添加截距项
    
    model = sm.OLS(y, X, missing='drop').fit()
    
    # 提取因子曝险系数（含alpha）
    exposures = model.params
    exposures.name = "factor_exposures"
    
    return exposures

def regress_one_fund(fund_code: str):
    fund_daily_returns = get_fund_daily_return(fund_code)
    # 若数据不足一年数据，不进行回归分析
    if len(fund_daily_returns) < 252:
        return None
    factor_weekly_data = factors_df.copy()
    exposures = calculate_factor_exposure(fund_daily_returns, factor_weekly_data, shibor_data)
    return exposures


def annualized_return(df, value_col='net_value'):
    """
    计算年化收益率
    使用公式： (最终净值/初始净值)^(365/持有天数) - 1
    """
    start_value = df[value_col].iloc[0]
    end_value = df[value_col].iloc[-1]
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    days = (end_date - start_date).days
    # 防止天数为 0 的情况
    if days == 0:
        return np.nan
    ann_return = (end_value / start_value) ** (365 / days) - 1
    return ann_return

def annualized_volatility(df, value_col='net_value'):
    """
    计算年化波动率
    基于日收益率的标准差，再乘以 sqrt(252)
    """
    df['daily_return'] = df[value_col].pct_change()
    daily_std = df['daily_return'].std()
    ann_vol = daily_std * np.sqrt(252)
    return ann_vol

def max_drawdown(df, value_col='net_value'):
    """
    计算最大回撤
    算法：计算净值的累计最高值，然后求每个时点的回撤，取最小值（负值最大跌幅）
    """
    running_max = df[value_col].cummax()
    drawdown = df[value_col] / running_max - 1
    max_dd = drawdown.min()  # 最大回撤（负值）
    return max_dd

def sharpe_ratio(df, value_col='net_value', risk_free_rate=0):
    """
    计算夏普比率：
      (年化收益率 - 无风险利率) / 年化波动率
    """
    ann_return = annualized_return(df, value_col)
    ann_vol = annualized_volatility(df, value_col)
    if ann_vol == 0:
        return np.nan
    return (ann_return - risk_free_rate) / ann_vol

def calmar_ratio(df, value_col='net_value'):
    """
    计算卡玛比率：
      年化收益率 / |最大回撤|
    """
    ann_return = annualized_return(df, value_col)
    max_dd = max_drawdown(df, value_col)
    if max_dd == 0:
        return np.nan
    return ann_return / abs(max_dd)

def monthly_positive_return_probability(df, value_col='net_value'):
    """
    计算月度正收益概率：
    1. 将数据按月（year-month）分组，
    2. 计算每个月的收益率：最后一个净值/第一个净值 - 1，
    3. 计算正收益月份的比例
    """
    # 提取年-月信息
    df['year_month'] = df['date'].dt.to_period('M')
    # 按月计算每个月的收益率（当月最后净值与首个净值）
    monthly_return = df.groupby('year_month')[value_col].agg(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    pos_prob = (monthly_return > 0).mean()  # 返回比例
    return pos_prob

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
    fund_info_df = fund_info_dao.select_dataframe_all()
    fund_factors_list = []
    for _, fund_info in fund_info_df.iterrows():
        logger.info(f"正在处理{fund_info['fund_code']}")
        fund_code = fund_info["fund_code"]
        exposures = regress_one_fund(fund_code)
        if exposures is None:
            continue
        metrics = compute_metrics(fund_code)
        if metrics is None:
            continue
        row_df = (
            exposures.to_frame().T
            .assign(code=fund_code,
                    ann_return=metrics[0], ann_vol=metrics[1], max_dd=metrics[2],
                    sharpe_ratio=metrics[3], calmar_ratio=metrics[4],
                    mprp=metrics[5])
        )
        fund_factors_list.append(row_df)
    fund_factors_df = pd.concat(fund_factors_list, ignore_index=True)
    fund_factors_df.to_csv("output/fund_factors.csv", index=False)

if __name__ == '__main__':
    with app.app_context():
        # 获取用户输入
        main()