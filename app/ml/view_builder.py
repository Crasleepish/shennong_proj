# 文件路径建议：app/ml/view_builder.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def build_view_matrix(
    df_beta: pd.DataFrame,
    df_softprob: pd.DataFrame,
    label_to_ret: Dict[str, Tuple[float, float, float]],
    asset_codes: List[str] = None,
    min_confidence: float = 0.5,
    top_k: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    构造 Black–Litterman 所需的 P, q, omega 矩阵。

    参数：
    - df_beta: DataFrame, 含每个资产对各因子的暴露（如 MKT, SMB, HML, QMJ）
    - df_softprob: DataFrame, 每个因子预测的 softprob，如：
          index=asset code, columns=['SMB_prob0', 'SMB_prob1', 'SMB_prob2', ...]
    - label_to_ret: Dict[str, Tuple[float, float, float]]，每个因子标签对应的预期收益值（从训练标签计算得到）
    - asset_codes: 要参与构造观点的资产列表，若为空则使用 df_beta 中所有资产
    - min_confidence: 置信度门槛（如 max_prob >= 0.5 才采纳）
    - top_k: 选取排名前top_k资产构造比较观点（减少P行数）

    返回：
    - P: 观点矩阵（shape: [num_views, num_assets]）
    - q: 观点值差（shape: [num_views]）
    - omega: 协方差矩阵（shape: [num_views, num_views]）
    - code_list: 资产顺序（供对应mu_prior/Σ）
    """
    factor_names = [f.split('_prob')[0] for f in df_softprob.columns if '_prob' in f]
    asset_codes = asset_codes or df_beta['code'].tolist()
    code_list = asset_codes  # 资产顺序

    # 1. 构建每个资产的预测收益（加总因子贡献）
    pred_mu = {}
    confidence_scores = {}

    for code in code_list:
        mu = 0
        conf = 0
        for f in factor_names:
            probs = df_softprob.loc[code, [f"{f}_prob0", f"{f}_prob1", f"{f}_prob2"]].values
            labels_ret = label_to_ret[f]  # (ret0, ret1, ret2)
            expected_ret = np.dot(probs, labels_ret)
            beta = df_beta[df_beta['code'] == code][f].values[0]
            mu += beta * expected_ret
            conf += max(probs)  # 简单求和表示整体置信度
        pred_mu[code] = mu
        confidence_scores[code] = conf / len(factor_names)

    # 2. 过滤置信度低的资产
    filtered = [code for code in code_list if confidence_scores[code] >= min_confidence]
    if len(filtered) < 2:
        raise ValueError("置信度合格的资产不足以构造相对观点")

    # 3. 排序取前top_k
    ranked = sorted(filtered, key=lambda x: pred_mu[x], reverse=True)[:top_k]

    # 4. 构造 P, q, ω
    P = []
    q = []
    omega = []

    for i in range(len(ranked) - 1):
        a, b = ranked[i], ranked[i + 1]
        row = np.zeros(len(code_list))
        idx_a = code_list.index(a)
        idx_b = code_list.index(b)
        row[idx_a] = 1
        row[idx_b] = -1
        P.append(row)
        q.append(pred_mu[a] - pred_mu[b])

        # 使用置信度构造 ω：越接近1说明置信越高 → 方差越小
        conf_avg = (confidence_scores[a] + confidence_scores[b]) / 2
        var = (1 - conf_avg) ** 2  # 可调函数
        omega.append(var)

    return np.array(P), np.array(q), np.diag(omega), code_list
