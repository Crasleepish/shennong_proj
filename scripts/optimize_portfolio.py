# 文件路径建议：app/scripts/optimize_portfolio.py

import pandas as pd
from typing import List, Tuple
from app.ml.view_builder import build_view_matrix
from app.ml.inference import get_softprob_dict, get_label_to_ret
import numpy as np
from app.dao.fund_info_dao import FundHistDao
from app.data.factor_data_reader import FactorDataReader


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

def optimize(fund_codes: List[str], trade_date: str, window: int = 20, view_codes: List[str] = None):
    # 1. 使用资产因子暴露与每日因子收益率，构造资产的历史收益率矩阵
    df_beta = pd.read_csv("output/fund_factors.csv")
    df_beta = df_beta[df_beta["code"].isin(fund_codes)].reset_index(drop=True)
    df_beta = df_beta.set_index("code")[["MKT", "SMB", "HML", "QMJ"]]

    df_factors = FactorDataReader.read_daily_factors()
    df_factors = df_factors[["MKT", "SMB", "HML", "QMJ"]].dropna()

    # 构造资产收益矩阵： asset_ret[t][i] = sum_j beta[i][j] * factor_ret[t][j]
    asset_ret_df = pd.DataFrame(index=df_factors.index)
    for code in fund_codes:
        beta = df_beta.loc[code].values  # shape: (4,)
        asset_ret_df[code] = df_factors.values @ beta  # matrix dot

    # 2. 构造先验收益与协方差矩阵（针对 fund_codes）
    mu_prior, Sigma, code_list_mu = compute_prior_mu_sigma(asset_ret_df, window=window)

    # 3. 构造观点（P, q, omega）（仅使用 view_codes 子集）
    view_asset_codes = view_codes if view_codes is not None else fund_codes
    P, q, omega, code_list_view = build_bl_views(view_asset_codes, trade_date)

    # ✅ 对齐 mu_prior 和 Sigma 的顺序与 fund_codes 保持一致
    code_index_map = {code: i for i, code in enumerate(code_list_mu)}
    fund_indices = [code_index_map[code] for code in fund_codes]
    mu_prior_full = mu_prior[fund_indices]
    Sigma_full = Sigma[np.ix_(fund_indices, fund_indices)]

    # ✅ 对齐观点所需子集顺序（用于 mu_post 更新）
    view_indices = [fund_codes.index(code) for code in code_list_view]
    mu_prior_view = mu_prior_full[view_indices]
    Sigma_view = Sigma_full[np.ix_(view_indices, view_indices)]

    # ✅ 计算后验期望收益（仅观点子集）
    mu_post_view = compute_bl_posterior(
        mu_prior=mu_prior_view,
        Sigma=Sigma_view,
        P=P,
        q=q,
        omega=omega,
        tau=0.05
    )

    # ✅ 将后验结果合并回原始 fund_codes 序列
    mu_post_full = mu_prior_full.copy()
    for i, idx in enumerate(view_indices):
        mu_post_full[idx] = mu_post_view[i]

    # TODO: 进行组合优化，如 mean-variance, max-Sharpe, CVaR 等
    return mu_post_full, Sigma_full, fund_codes
