import pandas as pd
import numpy as np
import statsmodels.api as sm
import akshare as ak
from app.dao.fund_info_dao import FundInfoDao
from app.data.helper import get_fund_daily_return

factors_df = pd.read_csv("output/factorsW-FRI.csv", 
                         parse_dates=["date"], 
                         dtype={"MKT": float, "SMB": float, "HML": float, "QMJ": float, "VOL": float})

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
    shibor_data = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="1月")
    exposures = calculate_factor_exposure(fund_daily_returns, factor_weekly_data, shibor_data)
    return exposures


def main():
    fund_info_dao = FundInfoDao._instance
    fund_info_df = fund_info_dao.select_dataframe_all()
    for _, fund_info in fund_info_df.iterrows():
        fund_code = fund_info["fund_code"]
        exposures = regress_one_fund(fund_code)
        print(f"{fund_code} exposures: {exposures}")