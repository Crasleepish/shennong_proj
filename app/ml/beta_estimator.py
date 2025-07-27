import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from app.data.helper import get_fund_daily_return_for_beta_regression, get_etf_daily_return_for_beta_regression
from app.dao.stock_info_dao import MarketFactorsDao
from app.dao.betas_dao import FundBetaDao
from app.ml.kalman_beta.kalman_filter import KalmanFilter
from app.ml.kalman_beta import QREstimator
from app.data_fetcher.trade_calender_reader import TradeCalendarReader

FACTOR_NAMES = ["MKT", "SMB", "HML", "QMJ"]


def run_historical_beta(code: str, asset_type: str, start_date: str, end_date: str, window_size: int = 60):
    market_factors_dao = MarketFactorsDao._instance
    factor_df = market_factors_dao.select_dataframe_by_date(start_date, end_date)
    if asset_type == "fund_info":
        return_df = get_fund_daily_return_for_beta_regression(code, start_date, end_date)
    elif asset_type == "etf_info":
        return_df = get_etf_daily_return_for_beta_regression(code, start_date, end_date)
    factor_df = factor_df.set_index("date", drop=True)
    factor_df.index = pd.to_datetime(factor_df.index)
    return_df.index = pd.to_datetime(return_df.index)
    df = pd.merge(factor_df, return_df, on="date").sort_values("date")
    df = df.dropna(how="any")
    df["intercept"] = 1.0

    qr = QREstimator(window_size=window_size)

    # 初始化：前 window 做 OLS 得到 z0
    init_df = df.iloc[:window_size]
    X_ols = init_df[FACTOR_NAMES + ["intercept"]].values
    y_ols = init_df["daily_return"].values
    model = LinearRegression().fit(X_ols, y_ols)
    z0 = model.coef_.reshape(-1, 1)
    P0 = np.diag([1.0, 1.0, 1.0, 1.0, 0.1])

    kf = KalmanFilter(state_dim=5, z0=z0, P0=P0)

    for idx_date, row in df.iterrows():
        x = np.array([row[f] for f in FACTOR_NAMES] + [1.0])
        y = row["daily_return"]

        qr.update_data(np.array([x]), y)
        Q, R = qr.estimate()
        z = kf.step(H=x.reshape(-1, 1), y=y, Q=Q, R=R)

        beta_dict = dict(zip(FACTOR_NAMES + ["const"], z.flatten().tolist()))
        FundBetaDao.upsert_one(
            code, idx_date.strftime("%Y-%m-%d"), beta_dict, P=kf.P
        )


def run_realtime_update(fund_code: str, end_date: str = None, window_size: int = 60):
    latest_df = FundBetaDao.select_latest_by_code(fund_code)
    if latest_df.empty:
        raise ValueError("未找到基金的历史状态记录，请先运行 run_historical_beta")

    latest = latest_df.iloc[0]
    z_prev = np.array([latest.MKT, latest.SMB, latest.HML, latest.QMJ, latest.const]).reshape(-1, 1)
    P_prev = (
        np.array(json.loads(latest.P_json))
        if pd.notna(latest.P_json)
        else np.diag([1.0, 1.0, 1.0, 1.0, 0.1])
    )
    start_date = latest.date.strftime("%Y-%m-%d")
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    factor_df = MarketFactorsDao.select_dataframe_by_date(start_date, end_date)
    fund_df = get_fund_daily_return_for_beta_regression(fund_code, start_date, end_date)
    df = pd.merge(factor_df, fund_df, on="date").sort_values("date")
    df["intercept"] = 1.0

    kf = KalmanFilter(state_dim=5, z0=z_prev, P0=P_prev)
    qr = QREstimator(window_size=window_size)

    for _, row in df.iterrows():
        x = np.array([row[f] for f in FACTOR_NAMES] + [1.0])
        y = row["daily_return"]

        qr.update_data(np.array([x]), y)
        Q, R = qr.estimate()
        z = kf.step(H=x.reshape(-1, 1), y=y, Q=Q, R=R)

        beta_dict = dict(zip(FACTOR_NAMES + ["const"], z.flatten().tolist()))
        FundBetaDao.upsert_one(
            fund_code, row["date"].strftime("%Y-%m-%d"), beta_dict, P=kf.P
        )

def run_historical_beta_batch(fund_codes: list[str], asset_type: str, start_date: str, end_date: str, window_size: int = 60):
    for code in fund_codes:
        try:
            print(f"[历史回填] 正在处理基金 {code}...")
            run_historical_beta(code, asset_type, start_date, end_date, window_size)
        except Exception as e:
            print(f"[历史回填] 基金 {code} 处理失败: {e}")


def run_realtime_update_batch(fund_codes: list[str], end_date: str = None, window_size: int = 60):
    for code in fund_codes:
        try:
            print(f"[实时更新] 正在处理基金 {code}...")
            run_realtime_update(code, end_date, window_size)
        except Exception as e:
            print(f"[实时更新] 基金 {code} 处理失败: {e}")
