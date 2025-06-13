# 文件路径建议：app/scripts/optimize_portfolio.py

import pandas as pd
from typing import List, Tuple
from app.ml.view_builder import build_view_matrix
from app.ml.inference import get_softprob_dict, get_label_to_ret
import numpy as np
from app.data_fetcher.factor_data_reader import FactorDataReader
from app.data_fetcher.csi_index_data_fetcher import CSIIndexDataFetcher
from app.dao.fund_info_dao import FundHistDao
from scipy.optimize import minimize

def load_fund_betas(codes: List[str]) -> pd.DataFrame:
    """
    从 output/fund_factors.csv 中读取指定 code 的因子暴露数据。
    返回包含 ['code', 'MKT', 'SMB', 'HML', 'QMJ'] 的 DataFrame。
    """
    df = pd.read_csv("output/fund_factors.csv")
    df = df[df["code"].isin(codes)].reset_index(drop=True)
    return df[["code", "MKT", "SMB", "HML", "QMJ"]]


def build_bl_views(codes: List[str], trade_date: str) -> Tuple:
    """
    构造 Black-Litterman 所需的观点：P, q, omega, code_list。
    """
    df_beta = load_fund_betas(codes)
    softprob_dict = get_softprob_dict(trade_date)
    label_to_ret = get_label_to_ret()

    P, q, omega, code_list = build_view_matrix(
        df_beta=df_beta,
        softprob_dict=softprob_dict,
        label_to_ret=label_to_ret,
        asset_codes=codes,
        top_k=min(10, len(codes))
    )
    return P, q, omega, code_list

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
    Sigma = ret.cov()
    codes = mu_series.index.tolist()
    return mu_series.values, Sigma.values, codes


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

def optimize(asset_source_map: dict, trade_date: str, window: int = 20, view_codes: List[str] = None):
    # 1. 根据不同数据来源构造资产净值矩阵
    factor_codes = [code for code, src in asset_source_map.items() if src == "factor"]
    index_codes = [code for code, src in asset_source_map.items() if src == "index"]
    hist_codes = [code for code, src in asset_source_map.items() if src == "hist"]

    net_value_df = pd.DataFrame()

    if factor_codes:
        df_beta = load_fund_betas(factor_codes).set_index("code")[["MKT", "SMB", "HML", "QMJ"]]
        df_factors = FactorDataReader.read_daily_factors()[["MKT", "SMB", "HML", "QMJ"]].dropna()
        for code in factor_codes:
            beta = df_beta.loc[code].values
            cumret = df_factors.values @ beta
            net_value_df[code] = (1 + pd.Series(cumret, index=df_factors.index)).cumprod()

    for code in index_codes:
        df = CSIIndexDataFetcher.get_data_by_code_and_date(code=code)
        df = df[["date", "close"]].dropna().set_index("date")
        df = df.sort_index()
        net_value_df[code] = df["close"]

    dao = FundHistDao._instance
    for code in hist_codes:
        df = dao.select_dataframe_by_code(code)
        df = df[["date", "net_value"]].dropna().set_index("date").sort_index()
        net_value_df[code] = df["net_value"]

    net_value_df = net_value_df.dropna(how="any").sort_index()
    fund_codes = list(asset_source_map.keys())

    # 2. 构造先验收益与协方差矩阵（使用净值曲线）
    mu_prior, Sigma, code_list_mu = compute_prior_mu_sigma(net_value_df, window=window, method="linear")

    # 3. 构造观点（P, q, omega）（仅使用 view_codes 子集）
    view_asset_codes = view_codes if view_codes is not None else fund_codes
    P, q, omega, code_list_view = build_bl_views(view_asset_codes, trade_date)

    # 对齐 mu_prior 和 Sigma 的顺序与 fund_codes 保持一致
    # code_index_map = {code: i for i, code in enumerate(code_list_mu)}
    fund_indices = [code_list_mu.index(code) for code in fund_codes]
    mu_prior_full = mu_prior[fund_indices]
    Sigma_full = Sigma[np.ix_(fund_indices, fund_indices)]

    # 提取观点相关子集
    view_indices = [fund_codes.index(code) for code in code_list_view]
    mu_prior_view = mu_prior_full[view_indices]
    Sigma_view = Sigma_full[np.ix_(view_indices, view_indices)]

    # 计算后验收益率（仅观点子集）
    mu_post_view = compute_bl_posterior(
        mu_prior=mu_prior_view,
        Sigma=Sigma_view,
        P=P,
        q=q,
        omega=omega,
        tau=0.5
    )

    # 将后验结果更新到完整序列中
    mu_post_full = mu_prior_full.copy()
    for i, idx in enumerate(view_indices):
        mu_post_full[idx] = mu_post_view[i]

    # ✅ 使用最大 Sharpe 比策略进行组合优化
    mu = mu_post_full
    cov = Sigma_full
    n = len(mu)

    def neg_sharpe(w):
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        return -port_ret / port_vol  # 负的 Sharpe 比

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0.0, 1.0)] * n  # 权重在 [0,1] 区间
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0=x0, bounds=bounds, constraints=constraints)
    weights = result.x
    expected_return = np.dot(weights, mu)
    expected_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))

    return {
        'weights': dict(zip(fund_codes, weights)),
        'expected_return': expected_return,
        'expected_volatility': expected_vol,
        'sharpe_ratio': expected_return / expected_vol
    }
