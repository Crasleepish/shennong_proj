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
    window: int = 20,
    eps_var: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    构造 Black–Litterman 所需的 P, q, omega，仅使用“绝对观点”。

    参数：
    - df_beta: DataFrame，含每个资产对各因子的暴露（如 MKT, SMB, HML, QMJ）
               必须包含 'code' 列和多个因子列
    - softprob_dict: 每个因子的 softmax 输出，如 {"MKT": [0.2, 0.3, 0.5], ...}
    - label_to_ret: 每个因子标签对应的预期收益（如 {0: -0.02, 1: 0, 2: 0.02}）
    - mu_prior: 每个资产的先验收益估计，如 {'510300.SH': 0.01, ...}
    - asset_codes: 参与构造观点的资产代码列表（默认全部）

    返回：
    - P: 观点矩阵（对角矩阵）
    - q: 每个资产的 μ_ML − μ_prior
    - omega: 每个资产的预测方差，对角矩阵
    - code_list: 返回资产顺序
    """
    asset_codes = asset_codes or df_beta['code'].tolist()
    code_list = asset_codes
    df_beta = df_beta.set_index('code')

    pred_mu = {}
    pred_var = {}
    valid_factors = ["MKT", "SMB", "HML", "QMJ"]

    for code in code_list:
        mu = 0
        var = 0
        for f in valid_factors:
            probs  = np.asarray(softprob_dict[f], dtype=float)      # e.g. [p0,p1,p2]
            labels_ret = np.asarray(label_to_ret[f], dtype=float)       # e.g. [-0.02,0,0.02]，已为（window）日
            expected_ret = float(np.dot(probs, labels_ret))
            variance = float(np.dot(probs, (labels_ret - expected_ret) ** 2))
            beta = df_beta.loc[code, f]
            mu += beta * expected_ret
            var += (beta ** 2) * variance
        pred_mu[code] = mu
        pred_var[code] = var

    # 构造绝对观点矩阵：每行对应一个资产，P = Identity
    num_assets = len(code_list)
    P = np.eye(num_assets)
    q = np.array([
        pred_mu[code] - mu_prior.get(code, 0.0)
        for code in code_list
    ])
    omega = np.diag([pred_var[code] for code in code_list])

    return P, q, omega, code_list