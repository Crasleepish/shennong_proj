import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from app.data.helper import get_fund_daily_return_for_beta_regression, get_etf_daily_return_for_beta_regression
from app.dao.stock_info_dao import MarketFactorsDao
from app.dao.betas_dao import FundBetaDao
from app.ml.kalman_beta.kalman_filter import KalmanFilter
from app.ml.kalman_beta import QREstimator
from app.ml.kalman_beta.q_r_estimator import _bootstrap_qr_from_history
import logging
from datetime import timedelta
from app.data_fetcher import CalendarFetcher

FACTOR_NAMES = ["MKT", "SMB", "HML", "QMJ"]
logger = logging.getLogger(__name__)


def run_historical_beta(code: str, asset_type: str, start_date: str, end_date: str, window_size: int = 60):
    factor_df = MarketFactorsDao._instance.select_dataframe_by_date(start_date, end_date).set_index("date")
    if asset_type == "fund_info":
        return_df = get_fund_daily_return_for_beta_regression(code, start_date, end_date)
    elif asset_type == "etf_info":
        return_df = get_etf_daily_return_for_beta_regression(code, start_date, end_date)
    return_df = return_df.dropna(how="any")
    if len(return_df) == 0:
        raise RuntimeError("No daily return data")
    factor_df = factor_df.set_index("date", drop=True)
    factor_df.index = pd.to_datetime(factor_df.index)
    return_df.index = pd.to_datetime(return_df.index)
    df = factor_df.join(return_df).sort_index()
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

def run_realtime_update(fund_code: str, start_date: str = None, end_date: str = None, window_size: int = 60):
    if start_date:
        pre_date = CalendarFetcher().get_prev_trade_date(start_date.replace("-", ""))
        latest_df = FundBetaDao.select_by_code_date(fund_code, pre_date)
        if latest_df.empty:
            raise ValueError("未找到基金的历史状态记录，确认start_date是合理的数据日期，并确认存在start_date之前的历史数据")
        latest = latest_df.iloc[0]
    else:
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
    start_date = (latest.date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    # ========= 新增：在 start_date 当天计算前，重建第 n-1 天的 QR 状态 =========
    # 这里把 ref_date 设为 start_date，这样会用 start_date 之前的 window_size 个交易日做窗口
    qr = _bootstrap_qr_from_history(fund_code, ref_date=start_date, window_size=window_size)

    factor_df = MarketFactorsDao._instance.select_dataframe_by_date(start_date, end_date).set_index("date")
    fund_df = get_fund_daily_return_for_beta_regression(fund_code, start_date, end_date)
    df = factor_df.join(fund_df).sort_index()
    df["intercept"] = 1.0

    kf = KalmanFilter(state_dim=5, z0=z_prev, P0=P_prev)

    for _, row in df.iterrows():
        x = np.array([row[f] for f in FACTOR_NAMES] + [1.0])
        y = row["daily_return"]

        qr.update_data(np.array([x]), y)
        Q, R = qr.estimate()
        z = kf.step(H=x.reshape(-1, 1), y=y, Q=Q, R=R)

        beta_dict = dict(zip(FACTOR_NAMES + ["const"], z.flatten().tolist()))
        FundBetaDao.upsert_one(
            fund_code, row.name.strftime("%Y-%m-%d"), beta_dict, P=kf.P
        )

def run_historical_beta_batch(fund_codes: list[str], asset_type: str, start_date: str, end_date: str, window_size: int = 60):
    for code in fund_codes:
        try:
            logger.info(f"[历史回填] 正在处理基金 {code}...")
            run_historical_beta(code, asset_type, start_date, end_date, window_size)
        except Exception as e:
            logger.error(f"[历史回填] 基金 {code} 处理失败: {e}")


def run_realtime_update_batch(fund_codes: list[str], start_date: str = None, end_date: str = None, window_size: int = 60):
    for code in fund_codes:
        try:
            logger.info(f"[实时更新] 正在处理基金 {code}...")
            run_realtime_update(code, start_date, end_date, window_size)
        except Exception as e:
            logger.error(f"[实时更新] 基金 {code} 处理失败: {e}")
