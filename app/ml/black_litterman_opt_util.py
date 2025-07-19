# app/scripts/optimize_portfolio.py

import pandas as pd
from typing import List, Tuple
from app.ml.view_builder import build_view_matrix
from app.ml.inference import get_softprob_dict, get_label_to_ret
import numpy as np
from app.ml.inference import get_softprob_dict, get_label_to_ret
from scipy.optimize import minimize
from app.ml.dataset_builder import DatasetBuilder

FUND_BETA_PATH = "output/fund_factors.csv"

def load_fund_betas(codes: List[str]) -> pd.DataFrame:
    """
    从 output/fund_factors.csv 中读取指定 code 的因子暴露数据。
    返回包含 ['code', 'MKT', 'SMB', 'HML', 'QMJ'] 的 DataFrame。
    """
    df = pd.read_csv(FUND_BETA_PATH)
    df = df[df["code"].isin(codes)].reset_index(drop=True)
    return df[["code", "MKT", "SMB", "HML", "QMJ"]]

def build_bl_views(
    code_type_map: dict,
    code_factors_map: dict,
    trade_date: str,
    mu_prior: dict,
    dataset_builder: DatasetBuilder = None
) -> Tuple:
    """
    构造 Black-Litterman 所需的观点：P, q, omega, code_list。
    """
    softprob_dict = get_softprob_dict(trade_date, dataset_builder=dataset_builder)
    label_to_ret = get_label_to_ret(trade_date)
    factor_type_keys = [k for k, v in code_type_map.items() if v == "factor"]
    df_beta_all = load_fund_betas(factor_type_keys)

    # 收集所有出现过的因子，并排序统一列顺序
    all_factors = sorted({f for fs in code_factors_map.values() for f in fs})
    beta_records = []
    for code, asset_type in code_type_map.items():
        factors = code_factors_map.get(code, [])

        if asset_type != "factor" and len(factors) != 1:
            raise ValueError(f"Non-factor asset {code} must have exactly one factor, got {factors}")

        beta_row = {"code": code}
        if asset_type == "factor":
            row = df_beta_all[df_beta_all["code"] == code]
            if row.empty:
                raise ValueError(f"Missing beta data for {code}")
            for f in factors:
                beta_row[f] = row[f].values[0]
        else:
            beta_row[factors[0]] = 1.0

        beta_records.append(beta_row)

    # 构造完整 beta DataFrame，未指定的因子填 0
    df_beta = pd.DataFrame(beta_records).fillna(0.0)
    df_beta = df_beta[["code"] + all_factors]

    P, q, omega, code_list = build_view_matrix(
        df_beta=df_beta,
        softprob_dict=softprob_dict,
        label_to_ret=label_to_ret,
        mu_prior=mu_prior,
        asset_codes=list(code_type_map.keys())
    )
    return P, q, omega, code_list

def ewma_cov(
    ret: pd.DataFrame,
    lambda_: float = 0.975,
    adaptive: bool = True,
    lambda_min: float = 0.95,
    lambda_max: float = 0.99,
    gamma: float = 0.025
) -> np.ndarray:
    """
    自适应 EWMA 协方差估计

    参数：
        ret: 收益率 DataFrame（行=时间，列=资产）
        lambda_: 默认衰减因子
        adaptive: 是否启用自适应 λ
        lambda_min/max: λ 的上下限
        gamma: λ对短期波动的响应强度

    返回：
        协方差矩阵 ndarray
    """
    ret = ret.dropna(how="any")

    if adaptive:
        # 计算市场整体的短期 / 长期波动比率
        vol_short = ret.rolling(10).std().mean(axis=1).dropna()
        vol_long = ret.rolling(250).std().mean(axis=1).dropna()
        common_idx = vol_short.index.intersection(vol_long.index)
        ratio_series = (vol_short[common_idx] / vol_long[common_idx]).dropna()

        if not ratio_series.empty:
            ratio = ratio_series.iloc[-1]  # 使用最新一日的波动比
            lambda_adapted = 1 - gamma * ratio
            lambda_ = float(np.clip(lambda_adapted, lambda_min, lambda_max))

    weights = np.array([lambda_**i for i in range(len(ret) - 1, -1, -1)])
    weights /= weights.sum()

    demeaned = ret - ret.mean()
    weighted_outer_products = np.zeros((ret.shape[1], ret.shape[1]))
    for i in range(len(ret)):
        x = demeaned.iloc[i].values.reshape(-1, 1)
        weighted_outer_products += weights[i] * (x @ x.T)

    return weighted_outer_products

def hybrid_cov(
    ret: pd.DataFrame,
    lambda_: float = 0.975,
    alpha: float = 0.5,
    adaptive: bool = True,
    lambda_min: float = 0.95,
    lambda_max: float = 0.99,
    gamma: float = 0.025
) -> np.ndarray:
    """
    混合型协方差估计器：EWMA + 历史等权协方差融合
    参数同 ewma_cov，新增 alpha：短期占比
    """
    ret = ret.dropna(how="any")
    if alpha == 0:
        return ret.cov().values
    Sigma_ewma = ewma_cov(ret, lambda_, adaptive, lambda_min, lambda_max, gamma)
    Sigma_hist = ret.cov().values  # 长期等权
    return alpha * Sigma_ewma + (1 - alpha) * Sigma_hist

def compute_prior_mu_sigma(
    price_df: pd.DataFrame,
    window: int = 20,
    method: str = "log"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    计算先验期望收益（mu_prior）和协方差矩阵（Sigma）
    参数：
        price_df: 行为日期，列为资产，值为价格
        window: 滚动窗口，默认为20日
        method: "log" 或 "linear"
    返回：
        mu_prior: ndarray (n_assets,)
        Sigma: ndarray (n_assets, n_assets)
        codes: List[str]
    """
    if method == "log":
        log_price = np.log(price_df)
        ret = log_price.diff(window)
    elif method == "linear":
        ret = price_df.pct_change(window)
    else:
        raise ValueError("method must be 'log' or 'linear'")

    ret = ret.dropna(how="any")
    mu_series = ret.mean()
    Sigma = hybrid_cov(ret, lambda_=0.975, alpha=0.2, adaptive=True)
    codes = mu_series.index.tolist()
    return mu_series.values, Sigma, codes

def compute_prior_mu_fixed_window(
    price_df: pd.DataFrame,
    window: int = 20,
    lookback_days: int = 252,
    method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    基于最近一整年的滑动收益率样本构造先验期望收益
    支持冷启动：若历史不足一年则用所有可用数据

    参数：
        price_df: 行为日期，列为资产，值为价格（建议为daily close）
        window: 每段收益期的天数（如20）
        lookback_days: 向前回溯的自然日长度（如252个自然日）
        method: 'log' or 'linear'，收益率计算方式

    返回：
        mu_series: 先验期望收益
    """
    required_len = lookback_days + window
    n_obs = len(price_df)

    if n_obs < window + 5:
        raise ValueError(f"数据太少，无法构造有效滚动收益样本（至少 {window + 5} 行）")

    # 冷启动：若不足一年则用全部数据；否则截取最近一年
    if n_obs >= required_len:
        recent_prices = price_df.iloc[-required_len:].copy()
    else:
        recent_prices = price_df.copy()

    # 计算滚动收益率
    if method == "log":
        log_price = np.log(recent_prices)
        ret = log_price.diff(periods=window)
    elif method == "linear":
        ret = recent_prices.pct_change(periods=window)
    else:
        raise ValueError("method must be 'log' or 'linear'")

    ret = ret.dropna(how="any")

    # 均值和协方差
    mu_series = ret.mean()

    return mu_series

def compute_bl_posterior(
    mu_prior: np.ndarray,
    Sigma: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    tau: float = 0.05
) -> np.ndarray:
    """
    Black-Litterman 后验收益公式实现
    """
    tau_Sigma_inv = np.linalg.inv(tau * Sigma)
    omega_inv = np.linalg.inv(omega)
    middle = np.linalg.inv(tau_Sigma_inv + P.T @ omega_inv @ P)
    rhs = tau_Sigma_inv @ mu_prior + P.T @ omega_inv @ q
    mu_post = middle @ rhs
    return mu_post

def optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray) -> tuple:
    """
    使用最大 Sharpe 比策略进行组合优化。
    参数：
        mu: ndarray, shape (n_assets,) - 后验期望收益率（单位为20日累计）
        cov: ndarray, shape (n_assets, n_assets) - 协方差矩阵
    返回：
        weights: ndarray, shape (n_assets,) - 最优资产权重
        expected_return: float - 最终组合的期望收益率
        expected_vol: float - 最终组合的年化波动率
    """
    n = len(mu)

    def neg_sharpe(w):
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        return -port_ret / port_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0=x0, bounds=bounds, constraints=constraints)
    weights = result.x
    expected_return = np.dot(weights, mu)
    expected_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))

    return weights, expected_return, expected_vol

def optimize_mean_variance(mu: np.ndarray, cov: np.ndarray, max_variance: float) -> tuple:
    """
    最大化期望收益，约束组合风险（方差）上限。

    参数：
        mu: ndarray, shape (n_assets,) - 后验期望收益率
        cov: ndarray, shape (n_assets, n_assets) - 协方差矩阵
        max_variance: float - 最大允许的组合方差（不是标准差）

    返回：
        weights: ndarray, shape (n_assets,) - 最优资产权重
        expected_return: float - 最终组合的期望收益率
        expected_vol: float - 最终组合的波动率（std）
    """
    n = len(mu)

    def neg_return(w):
        return -np.dot(w, mu)

    def risk_constraint(w):
        return max_variance - np.dot(w, np.dot(cov, w))

    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
        {'type': 'ineq', 'fun': risk_constraint}         # 风险约束
    )
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_return, x0=x0, bounds=bounds, constraints=constraints)
    weights = result.x
    expected_return = np.dot(weights, mu)
    expected_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))

    return weights, expected_return, expected_vol