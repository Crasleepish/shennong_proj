# app/scripts/optimize_portfolio.py

import pandas as pd
from typing import List, Tuple
from app.ml.view_builder import build_view_matrix
from app.ml.inference import get_softprob_dict, get_label_to_ret
import numpy as np
from app.ml.inference import get_softprob_dict, get_label_to_ret
from scipy.optimize import minimize
from app.ml.dataset_builder import DatasetBuilder
from app.dao.betas_dao import FundBetaDao

def load_fund_betas(codes: List[str], trade_date: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    从 output/fund_factors.csv 中读取指定 code 的因子暴露数据。
    返回包含 ['code', 'MKT', 'SMB', 'HML', 'QMJ'] 的 DataFrame。
    """
    some_days_ago = (pd.to_datetime(trade_date) - pd.DateOffset(days=lookback_days)).strftime('%Y-%m-%d')
    df = FundBetaDao.get_latest_fund_betas(fund_type_list=["股票型"], invest_type_list=["被动指数型", "增强指数型"], found_date_limit=some_days_ago, as_of_date=trade_date)
    df = df.set_index("code", drop=True)
    df = df[df.index.isin(codes)]
    return df[["MKT", "SMB", "HML", "QMJ"]]

def load_fund_const(codes: List[str], trade_date: str) -> pd.DataFrame:
    some_days_ago = (pd.to_datetime(trade_date)).strftime('%Y-%m-%d')
    df = FundBetaDao.select_const_by_code(codes, some_days_ago)
    return df

def build_bl_views(
    code_type_map: dict,
    code_factors_map: dict,
    trade_date: str,
    mu_prior: dict,
    dataset_builder: DatasetBuilder = None,
    window: int = 20
) -> Tuple:
    """
    构造 Black-Litterman 所需的观点：P, q, omega, code_list。
    """
    softprob_dict = get_softprob_dict(trade_date, dataset_builder=dataset_builder)
    label_to_ret = get_label_to_ret(trade_date)
    factor_type_keys = [k for k, v in code_type_map.items() if v == "factor"]
    df_beta_all = load_fund_betas(factor_type_keys, trade_date, lookback_days=90)

    # 收集所有出现过的因子，并排序统一列顺序
    all_factors = sorted({f for fs in code_factors_map.values() for f in fs})
    beta_records = []
    for code, asset_type in code_type_map.items():
        factors = code_factors_map.get(code, [])

        if asset_type != "factor" and len(factors) != 1:
            raise ValueError(f"Non-factor asset {code} must have exactly one factor, got {factors}")

        beta_row = {"code": code}
        if asset_type == "factor":
            row = df_beta_all.loc[code]
            if row.empty:
                raise ValueError(f"Missing beta data for {code}")
            for f in factors:
                beta_row[f] = row[f]
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
    gamma: float = 0.025,
    jitter: float = 1e-10
) -> np.ndarray:
    """
    自适应 EWMA 协方差估计（权重归一、加权均值去中心、向量化、PSD 处理）

    参数：
        ret        : pd.DataFrame
                     收益率矩阵，行=时间（按先后排序），列=资产代码/名称。元素为同一口径的
                     日/周/月算术收益（与下游优化所用口径一致）。

        lambda_    : float，默认 0.975
                     EWMA 衰减因子（固定值）。越接近 1，历史越“长”、平滑越强；越小则更敏感于近期。

        adaptive   : bool，默认 True
                     是否启用“自适应 λ”。开启后，会根据“短期/长期”市场波动比自动调整 λ，
                     在高波动期降低 λ（更敏感），低波动期提高 λ（更平滑）。

        lambda_min : float，默认 0.95
                     自适应 λ 的下界（防止 λ 被调得过小导致噪声过大）。

        lambda_max : float，默认 0.99
                     自适应 λ 的上界（防止 λ 被调得过大导致响应过慢）。

        gamma      : float，默认 0.025
                     自适应强度系数。自适应公式近似为：lambda_adapted = clip(1 - gamma * ratio, lambda_min, lambda_max)，
                     其中 ratio 为“10日平均波动 / 250日平均波动”的最新比值。gamma 越大，λ 对短期波动越敏感。

        jitter     : float，默认 1e-10
                     数值稳健用的对角微扰。返回前对协方差矩阵加 jitter*I，确保数值上对称且（近）正定，
                     便于后续求逆/分解。

    返回：
        np.ndarray，形状 (n_assets, n_assets)
        对应 ret 列顺序的 EWMA 协方差矩阵。
    """
    # 1) 对齐并清洗
    ret = ret.copy()
    ret = ret.dropna(how="any")

    T, N = ret.shape
    if T == 0:
        return np.zeros((N, N))

    # 2) 自适应 λ（可用更稳健的中位数代替 mean）
    if adaptive:
        vol_short = ret.rolling(10).std().mean(axis=1).dropna()
        vol_long  = ret.rolling(250).std().mean(axis=1).dropna()
        common_idx = vol_short.index.intersection(vol_long.index)
        if len(common_idx) > 0:
            ratio = float((vol_short[common_idx] / vol_long[common_idx]).iloc[-1])
            lambda_ = float(np.clip(1.0 - gamma * ratio, lambda_min, lambda_max))

    # 3) 归一化权重（最近期权重最大）
    idx = np.arange(T-1, -1, -1)                    # [T-1, ..., 0]
    w = (lambda_ ** idx).astype(float)
    w /= w.sum()                                     # sum w_i = 1

    X = ret.values.astype(float)

    # 4) 用“同一权重”计算加权均值，并去中心（关键修正点）
    mu_w = (w[:, None] * X).sum(axis=0)              # shape (N,)
    Xc = X - mu_w                                    # 去中心

    # 5) 向量化加权协方差：Xc^T diag(w) Xc
    #   等价于 sum_i w_i * (x_i - mu_w)(x_i - mu_w)^T
    S = Xc.T @ (w[:, None] * Xc)

    # 6) 数值稳健化：对称化 + jitter，保证 PSD
    S = 0.5 * (S + S.T)
    if jitter and jitter > 0:
        S += jitter * np.eye(N)

    return S

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
) -> np.ndarray:
    """
    基于最近一整年的滑动收益率样本构造先验期望收益
    支持冷启动：若历史不足一年则用所有可用数据

    参数：
        price_df: 行为日期，列为资产，值为价格（建议为daily close）
        window: 每段收益期的天数（如20）
        lookback_days: 向前回溯的交易日长度（如252个交易日）
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
    最大化期望收益，约束组合风险（方差）上限，并限制单一资产权重不超过 0.5。

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
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # 权重和为 1
        {'type': 'ineq', 'fun': risk_constraint}           # 风险约束
    )
    # 单一资产上限由 1.0 改为 0.5
    bounds = [(0.0, 0.5)] * n
    x0 = np.ones(n) / n
    # 初值若超界（极端 n<2 时）也会被 bounds 处理，但这里保持等权
    result = minimize(neg_return, x0=x0, bounds=bounds, constraints=constraints, method='SLSQP')

    if not result.success:
        # 兜底：投影到边界后归一
        w = np.clip(x0, 0.0, 0.5)
        s = w.sum()
        w = w / s if s > 0 else np.ones(n) / n
    else:
        w = result.x

    expected_return = float(np.dot(w, mu))
    expected_vol = float(np.sqrt(np.dot(w, np.dot(cov, w))))
    return w, expected_return, expected_vol