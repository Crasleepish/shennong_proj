import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from app.data.helper import get_fund_daily_return_for_beta_regression, get_etf_daily_return_for_beta_regression
from app.dao.stock_info_dao import MarketFactorsDao
from app.dao.betas_dao import FundBetaDao
from app.ml.kalman_beta.kalman_filter import KalmanFilter
from app.ml.kalman_beta import QREstimator, ResidualDebiasor
from app.ml.kalman_beta.q_r_estimator import _bootstrap_qr_from_history
from app.ml.kalman_beta.residual_debiasor import _bootstrap_debiasor_from_history
import logging
from datetime import timedelta
from app.data_fetcher import CalendarFetcher
from app.utils.cov_packer import unpack_covariance

FACTOR_NAMES = ["MKT", "SMB", "HML", "QMJ"]
logger = logging.getLogger(__name__)

def safe_value(val):
    return None if pd.isna(val) else val


def run_historical_beta(code: str, asset_type: str, start_date: str, end_date: str, window_size: int = 60):
    factor_df = MarketFactorsDao._instance.select_dataframe_by_date(start_date, end_date).set_index("date")
    if asset_type == "fund_info":
        return_df = get_fund_daily_return_for_beta_regression(code, start_date, end_date)
    elif asset_type == "etf_info":
        return_df = get_etf_daily_return_for_beta_regression(code, start_date, end_date)
    return_df = return_df.dropna(how="any")
    if len(return_df) == 0:
        raise RuntimeError("No daily return data")

    factor_df.index = pd.to_datetime(factor_df.index)
    return_df.index = pd.to_datetime(return_df.index)

    df = factor_df.join(return_df).sort_index().dropna(how="any")
    df["intercept"] = 1.0

    # ---------- Q/R 估计器 ----------
    qr = QREstimator(window_size=window_size)

    # ---------- 初始化：前 window 做 OLS 得到 z0 ----------
    init_df = df.iloc[:window_size]
    X_ols = init_df[FACTOR_NAMES + ["intercept"]].values
    y_ols = init_df["daily_return"].values
    model = LinearRegression().fit(X_ols, y_ols)
    # 注意：sklearn 的 coef_ 是 (n_features,)；这里包含了截距列，因此 intercept_ 不用
    z0 = model.coef_.reshape(-1, 1)              # shape: (5,1) = 4 beta + 1 alpha(const)
    P0 = np.diag([1.0, 1.0, 1.0, 1.0, 0.1])

    # ---------- KF：启用 ECM（gamma） ----------
    kf = KalmanFilter(
        state_dim=5,              # 原始维度（不含 ECM）
        z0=z0,
        P0=P0,
        alpha_index=-1,
        alpha_rho=0.98,
        use_joseph=True,
        use_ecm=True,             # <=== 开启 ECM
        gamma_rho=0.99,           # 轻度均值回复
        q_gamma=1e-5              # gamma 的过程噪声略大，便于快响应
    )

    # ---------- 累计净值（log 域）跟踪 ----------
    # 以循环开始前的“基点”为 0：log_nav_true_{t0-1}=0，log_nav_fit_{t0-1}=0
    log_nav_true = 0.0
    log_nav_fit = 0.0

    for idx_date, row in df.iterrows():
        # 基础观测向量（不含 ECM 列）；ECM 的 TE 由 te_prev 传入 KF
        x = np.array([row[f] for f in FACTOR_NAMES] + [1.0], dtype=float)  # shape: (5,)

        # 当期真实日收益
        y = float(row["daily_return"])

        # —— 1) 先用“上一期状态”预测当期收益，用于推进拟合净值（严格一侧）——
        z_prev = kf.current_state()                   # shape: (5 [+1 if ECM], 1)
        # 由于 z 内部可能已扩一维（gamma），这里只取前 5 维参与 r_fit
        y_fit_pred = float(x @ z_prev[:5, :].ravel())
        # log1p 的定义域要求 > -1
        y_fit_pred = np.clip(y_fit_pred, -0.999999, None)

        # —— 2) 计算 te_prev（上一期累计跟踪误差，log 域）——
        te_prev = float(log_nav_true - log_nav_fit)   # TE_{t-1}

        # —— 3) Q/R 估计：仍然用 (X,y)，不含 TE 列 —— 
        qr.update_data(np.array([x]), y)
        Q, R = qr.estimate()                          # Q: (5,5) 或 (state_dim,state_dim)，R: 标量
        # 若 Q 是 (5,5)，KF 内部会自动扩一维给 gamma

        # —— 4) KF 更新：把 te_prev 传入（ECM 生效）——
        z = kf.step(H=x.reshape(-1, 1), y=y, Q=Q, R=R, te_prev=te_prev)

        # —— 5) 推进累计净值（log 域）——
        log_nav_true += np.log1p(np.clip(y, -0.999999, None))
        log_nav_fit  += np.log1p(y_fit_pred)

        # —— 6) 入库（注意：z 包含 gamma；要一并持久化）——
        z_all = z.flatten().tolist()                 # 长度 = 6 : 4 beta + alpha + gamma
        z_plain = z_all[:5]                          # 前5维：MKT, SMB, HML, QMJ, const
        gamma_val = z_all[5]                         # 第6维：gamma（ECM 系数）

        beta_dict = dict(zip(FACTOR_NAMES + ["const"], (safe_value(v) for v in z_plain)))
        beta_dict["gamma"] = safe_value(gamma_val)

        FundBetaDao.upsert_one(
            code,
            idx_date.strftime("%Y-%m-%d"),
            beta_dict,
            P=kf.current_cov(),                      # 6x6
            log_nav_true=float(log_nav_true),
            log_nav_fit=float(log_nav_fit),
        )

def _earliest_common_date(code: str, asset_type: str, end_date: str) -> str:
    """找到“因子 × 资产收益”的最早可用共同日期（闭区间到 end_date）。"""
    # 为简单稳健，取一个足够早的起点
    start_probe = "1990-01-01"
    factor = MarketFactorsDao._instance.select_dataframe_by_date(start_probe, end_date).set_index("date")
    if asset_type == "fund_info":
        ret = get_fund_daily_return_for_beta_regression(code, start_probe, end_date)
    elif asset_type == "etf_info":
        ret = get_etf_daily_return_for_beta_regression(code, start_probe, end_date)
    else:
        raise ValueError(f"Unknown asset_type: {asset_type}")
    factor.index = pd.to_datetime(factor.index)
    ret.index = pd.to_datetime(ret.index)
    df = factor.join(ret).dropna(how="any").sort_index()
    if df.empty:
        raise RuntimeError("无法找到可用的共同历史数据（因子或收益缺失）。")
    return df.index[0].strftime("%Y-%m-%d")

def run_realtime_update(
    fund_code: str,
    start_date: str = None,
    end_date: str = None,
    window_size: int = 60,
    *,
    asset_type: str = "fund_info",
    fallback_to_full: bool = True
):
    """
    断点续算 + 无历史时自动从头跑（等价于自动调用 run_historical_beta）。
    - asset_type: "fund_info" | "etf_info"
    - fallback_to_full: True 时，缺少历史就自动从头初始化并跑到 end_date
    """
    # ===== A) 尝试找到断点 =====
    latest = None
    if start_date:
        # 要从某天开始，则看它的前一交易日是否有记录
        pre_date = CalendarFetcher().get_prev_trade_date(start_date.replace("-", ""))
        latest_df = FundBetaDao.select_by_code_date(fund_code, pre_date)
        if not latest_df.empty:
            latest = latest_df.iloc[0]
        elif not fallback_to_full:
            raise ValueError("未找到start_date之前的历史状态记录，且未启用fallback_to_full。")
    else:
        # 未指定start_date，则取表里的最后一条作为断点
        latest_df = FundBetaDao.select_latest_by_code(fund_code)
        if not latest_df.empty:
            latest = latest_df.iloc[0]
        elif not fallback_to_full:
            raise ValueError("未找到任何历史状态记录，且未启用fallback_to_full。")

    # ===== B) 如果没有断点：自动从头跑一次（等价 run_historical_beta）后返回 =====
    if latest is None:
        hist_start = _earliest_common_date(fund_code, asset_type, end_date or pd.Timestamp.today().strftime("%Y-%m-%d"))
        hist_end = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        # 直接用已有的历史批处理函数（它现在已是 ECM 版、会写入 log_nav_true/fit 与 P）
        run_historical_beta(fund_code, asset_type, hist_start, hist_end, window_size=window_size)
        return  # 从头已跑完，本次实时不再重复

    # ===== C) 有断点：按 ECM 方案续算 =====
    z_prev = np.array([latest.MKT, latest.SMB, latest.HML, latest.QMJ, latest.const, latest.gamma], dtype=float).reshape(-1, 1)
    P_prev = (
        unpack_covariance(latest.P_bin, "{\"dtype\": \"float32\", \"n\": 6}")
        if pd.notna(latest.P_bin)
        else np.diag([1.0, 1.0, 1.0, 1.0, 0.1])
    )

    # ECM 所需累计对数净值
    if pd.isna(latest.log_nav_true) or pd.isna(latest.log_nav_fit):
        if not fallback_to_full:
            raise ValueError("历史状态缺少 log_nav_true/log_nav_fit，且未启用fallback_to_full。")
        # 若历史是老版本（未持久化log字段），最稳的是从头回填；走一次全量历史以对齐口径
        hist_start = _earliest_common_date(fund_code, asset_type, (latest.date + timedelta(days=0)).strftime("%Y-%m-%d"))
        run_historical_beta(fund_code, asset_type, hist_start, (latest.date).strftime("%Y-%m-%d"), window_size=window_size)
        # 重新读取最新断点
        latest_df = FundBetaDao.select_latest_by_code(fund_code)
        if latest_df.empty:
            raise RuntimeError("回填历史后仍未找到断点。")
        latest = latest_df.iloc[0]

    log_nav_true = float(latest.log_nav_true)
    log_nav_fit  = float(latest.log_nav_fit)

    # 续跑区间
    start_date = (latest.date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    # ===== D) 重建 Q/R 窗口（严格一侧） =====
    qr = _bootstrap_qr_from_history(fund_code, ref_date=start_date, asset_type=asset_type, window_size=window_size)

    # ===== E) 取数据并循环 =====
    factor_df = MarketFactorsDao._instance.select_dataframe_by_date(start_date, end_date).set_index("date")
    if asset_type == "fund_info":
        ret_df = get_fund_daily_return_for_beta_regression(fund_code, start_date, end_date)
    elif asset_type == "etf_info":
        ret_df = get_etf_daily_return_for_beta_regression(fund_code, start_date, end_date)
    else:
        raise ValueError(f"Unknown asset_type: {asset_type}")

    df = factor_df.join(ret_df).sort_index()
    if df.empty:
        return
    df["intercept"] = 1.0

    # ===== F) KF（ECM 版） =====
    kf = KalmanFilter(
        state_dim=5, z0=z_prev, P0=P_prev,
        alpha_index=-1, alpha_rho=0.98,
        use_joseph=True,
        use_ecm=True,
        gamma_rho=0.99,
        q_gamma=1e-5
    )

    for idx_date, row in df.iterrows():
        x = np.array([row[f] for f in FACTOR_NAMES] + [1.0], dtype=float)  # (5,)
        y = float(row["daily_return"])

        # —— 用上一期参数预测 y_fit（用于推进拟合净值；不含ECM列）——
        z_prev_full = kf.current_state()
        y_fit_pred = float(x @ z_prev_full[:5, :].ravel())
        y_fit_pred = np.clip(y_fit_pred, -0.999999, None)

        # —— ECM 误差 TE_{t-1} —— 
        te_prev = float(log_nav_true - log_nav_fit)

        # —— Q/R —— 
        qr.update_data(np.array([x]), y)
        Q, R = qr.estimate()

        # —— KF 更新（ECM）——
        z = kf.step(H=x.reshape(-1, 1), y=y, Q=Q, R=R, te_prev=te_prev)

        # —— 推进累计对数净值 —— 
        log_nav_true += np.log1p(np.clip(y, -0.999999, None))
        log_nav_fit  += np.log1p(y_fit_pred)

        # —— 入库：仅前 5 维 + P + 累计log净值 —— 
        z_plain = z[:5, :].flatten().tolist()
        beta_dict = dict(zip(FACTOR_NAMES + ["const"], (safe_value(v) for v in z_plain)))
        FundBetaDao.upsert_one(
            fund_code,
            idx_date.strftime("%Y-%m-%d"),
            beta_dict,
            P=kf.current_cov(),
            log_nav_true=float(log_nav_true),
            log_nav_fit=float(log_nav_fit),
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
