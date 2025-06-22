# 文件路径建议：app/scripts/optimize_portfolio.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from typing import List, Tuple
from app.ml.view_builder import build_view_matrix
from app.ml.inference import get_softprob_dict, get_label_to_ret
import numpy as np
from app.data_fetcher.factor_data_reader import FactorDataReader
from app.data_fetcher.csi_index_data_fetcher import CSIIndexDataFetcher
from app.dao.fund_info_dao import FundHistDao
from app.ml.inference import get_softprob_dict, get_label_to_ret
from scipy.optimize import minimize
import cvxpy as cp
from app import create_app

app = create_app()

def load_fund_betas(codes: List[str]) -> pd.DataFrame:
    """
    从 output/fund_factors.csv 中读取指定 code 的因子暴露数据。
    返回包含 ['code', 'MKT', 'SMB', 'HML', 'QMJ'] 的 DataFrame。
    """
    df = pd.read_csv("output/fund_factors.csv")
    df = df[df["code"].isin(codes)].reset_index(drop=True)
    return df[["code", "MKT", "SMB", "HML", "QMJ"]]


def build_bl_views(
    code_type_map: dict,
    code_factors_map: dict,
    trade_date: str
) -> Tuple:
    """
    构造 Black-Litterman 所需的观点：P, q, omega, code_list。
    """
    softprob_dict = get_softprob_dict(trade_date)
    label_to_ret = get_label_to_ret()
    df_beta_all = load_fund_betas(list(code_type_map.keys()))

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
        asset_codes=list(code_type_map.keys()),
        top_k=min(20, len(code_type_map))
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

def optimize(asset_source_map: dict, code_factors_map: dict, trade_date: str, window: int = 20, view_codes: List[str] = None):
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
    if not view_codes:
        view_asset_source_map = asset_source_map
        view_code_factors_map = code_factors_map
    else:
        view_asset_source_map = {code: asset_source_map[code] for code in view_codes if code in asset_source_map}
        view_code_factors_map = {code: code_factors_map[code] for code in view_codes if code in code_factors_map}
    P, q, omega, code_list_view = build_bl_views(view_asset_source_map, view_code_factors_map, trade_date)

    # 对齐 mu_prior 和 Sigma 的顺序与 fund_codes 保持一致
    # code_index_map = {code: i for i, code in enumerate(code_list_mu)}
    fund_indices = [code_list_mu.index(code) for code in fund_codes]
    mu_prior_full = mu_prior[fund_indices]
    Sigma_full = Sigma[np.ix_(fund_indices, fund_indices)]

    # 提取观点相关子集，将先验mu和Sigma调整与顺序与code_list_view一致
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

    # 将后验结果更新到完整序列中，顺序与fund_codes保持一致
    mu_post_full = mu_prior_full.copy()
    for i, idx in enumerate(view_indices):
        mu_post_full[idx] = mu_post_view[i]

    # 4. Max Sharpe 组合优化
    # weights, expected_return, expected_vol = optimize_max_sharpe(mu_post_full, Sigma_full)
    weights, expected_return, expected_vol = optimize_mean_variance(mu_post_full, Sigma_full, 0.0006)

    return {
        'weights': dict(zip(fund_codes, weights)),
        'expected_return': expected_return,
        'expected_volatility': expected_vol,
        'sharpe_ratio': expected_return / expected_vol
    }

if __name__ == '__main__':
    with app.app_context():
        # 创建测试数据
        asset_source_map = {
            'H11004.CSI': 'index',
            'Au99.99.SGE': 'index',
            '008114.OF': 'factor',
            '020602.OF': 'factor',
            '019918.OF': 'factor', 
            '002236.OF': 'factor',
            '019311.OF': 'factor',
            '006712.OF': 'factor',
            '011041.OF': 'factor',
            '110003.OF': 'factor',
            '019702.OF': 'factor',
            '006342.OF': 'factor',
            '020466.OF': 'factor',
            '018732.OF': 'factor',
        }
        code_factors_map = {
            "H11004.CSI": ["10YBOND"], 
            "Au99.99.SGE": ["GOLD"],
            "008114.OF": ["MKT", "SMB", "HML", "QMJ"],
            "020602.OF": ["MKT", "SMB", "HML", "QMJ"],
            "019918.OF": ["MKT", "SMB", "HML", "QMJ"],
            "002236.OF": ["MKT", "SMB", "HML", "QMJ"],
            "019311.OF": ["MKT", "SMB", "HML", "QMJ"],
            "006712.OF": ["MKT", "SMB", "HML", "QMJ"],
            "011041.OF": ["MKT", "SMB", "HML", "QMJ"],
            "110003.OF": ["MKT", "SMB", "HML", "QMJ"],
            "019702.OF": ["MKT", "SMB", "HML", "QMJ"],
            '006342.OF': ["MKT", "SMB", "HML", "QMJ"],
            '020466.OF': ["MKT", "SMB", "HML", "QMJ"],
            '018732.OF': ["MKT", "SMB", "HML", "QMJ"],
        }
        trade_date = '2025-06-19'
        window = 20
        # view_codes = ['H11004.CSI', 'Au99.99.SGE', '008114.OF', '020602.OF', '019918.OF', '002236.OF', '019311.OF', '006712.OF', '011041.OF', '110003.OF', '019702.OF', '006342.OF']
        portfolio_plan = optimize(asset_source_map, code_factors_map, trade_date, window, None)
        print(portfolio_plan)
    