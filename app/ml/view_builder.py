# 文件路径建议：app/ml/view_builder.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def build_view_matrix(
    df_beta: pd.DataFrame,
    softprob_dict: Dict[str, np.ndarray],
    label_to_ret: Dict[str, List[float]],
    mu_prior: Dict[str, float],
    asset_codes: List[str] = None,
    eps_var: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    构造 Black–Litterman 所需的 P, q, omega，仅使用“绝对观点”。

    参数：
    - df_beta: DataFrame，含 'code' 列及因子列（如 MKT/SMB/HML/QMJ/10YBOND/GOLD 等）
    - softprob_dict: {factor: probs(np.ndarray)}，如 {"MKT": [p0,p1,p2], ...}
    - label_to_ret: {factor: labels_ret(List[float])}，与 probs 维度一致
    - mu_prior: {code: μ_prior}（口径需与 label_to_ret 一致）
    - asset_codes: 需要生成观点的资产代码列表（默认取 df_beta['code'] 全部）
    - eps_var: ω 对角线的最小方差下限，防止 0 方差导致数值问题

    返回：
    - P: I，对角矩阵（资产数 × 资产数）
    - q: 各资产 μ_view − μ_prior
    - omega: 对角方差矩阵
    - code_list: 资产顺序
    """
    asset_codes = asset_codes or df_beta['code'].tolist()
    code_list = asset_codes
    df_beta = df_beta.set_index('code')

    pred_mu = {}
    pred_var = {}

    for code in code_list:
        mu = 0
        var = 0
        valid_factors = df_beta.columns[df_beta.loc[code] != 0].tolist()
        for f in valid_factors:
            probs = softprob_dict[f]
            labels_ret = label_to_ret[f]
            expected_ret = np.dot(probs, labels_ret)
            variance = np.dot(probs, (np.array(labels_ret) - expected_ret) ** 2)
            beta = df_beta.loc[code, f]
            mu += beta * expected_ret
            var += (beta ** 2) * variance
        pred_mu[code] = mu
        pred_var[code] = max(var, eps_var)

    # 构造绝对观点矩阵：每行对应一个资产，P = Identity
    num_assets = len(code_list)
    P = np.eye(num_assets)
    q = np.array([
        pred_mu[code] - mu_prior.get(code, 0.0)
        for code in code_list
    ])
    omega = np.diag([pred_var[code] for code in code_list])

    return P, q, omega, code_list