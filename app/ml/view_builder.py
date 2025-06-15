# 文件路径建议：app/ml/view_builder.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def build_view_matrix(df_beta: pd.DataFrame, softprob_dict: dict, label_to_ret: dict, asset_codes: list, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    构造 Black–Litterman 所需的 P, q, omega 矩阵。

    参数：
    - df_beta: DataFrame, 含每个资产对各因子的暴露（如 MKT, SMB, HML, QMJ）
    - softprob_dict: Dict[str, np.ndarray]，每个因子的三分类 softprob，如：
          {"MKT": [0.2, 0.3, 0.5], "SMB": [0.4, 0.4, 0.2], ...}
    - label_to_ret: 每个因子标签对应的预期收益值（从训练标签计算得到）
    - asset_codes: 要参与构造观点的资产列表，若为空则使用 df_beta 中所有资产
    - top_k: 选取排名前top_k资产构造比较观点（减少P行数）

    返回：
    - P: 观点矩阵（shape: [num_views, num_assets]）
    - q: 观点值差（shape: [num_views]）
    - omega: 协方差矩阵（shape: [num_views, num_views]）
    - code_list: 资产顺序（供对应mu_prior/Σ）
    """
    asset_codes = asset_codes or df_beta['code'].tolist()
    code_list = asset_codes  # 资产顺序

    pred_mu = {}
    pred_var = {}
    df_beta = df_beta.set_index('code')
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
        pred_var[code] = var

    # 根据置信度排序（mu / sqrt(var)）
    scores = {code: pred_mu[code] / (np.sqrt(pred_var[code]) + 1e-8) for code in code_list}
    ranked = sorted(code_list, key=lambda x: scores[x], reverse=True)[:top_k]

    P, q, omega = [], [], []
    for i in range(len(ranked) - 1):
        a, b = ranked[i], ranked[i + 1]
        row = np.zeros(len(code_list))
        idx_a = code_list.index(a)
        idx_b = code_list.index(b)
        row[idx_a] = 1
        row[idx_b] = -1
        P.append(row)
        q.append(pred_mu[a] - pred_mu[b])
        omega_i = pred_var[a] + pred_var[b]
        omega.append(omega_i)

    return np.array(P), np.array(q), np.diag(omega), code_list
