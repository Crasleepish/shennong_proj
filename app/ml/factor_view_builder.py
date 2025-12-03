# app/ml/factor_view_builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.ml.inference import get_softprob_dict  # 复用你已有的 ML 推理接口
from app.ml.rolling_three_class import (
    RollingThreeClassConfig,
    ThreeClassStats,
    compute_three_class_stats_for_date,
)


# 因子信息源配置
FACTOR_SOURCE: Dict[str, str] = {
    "MKT": "ml",        # 纯 ML
    "SMB": "rolling",   # 只用统计
    "HML": "rolling",
    "QMJ": "rolling",
    "10YBOND": "mix",   # ML + 统计 融合
    "GOLD": "none",     # 不参与 BL
}


# 各因子的 Rolling 配置（可以后续在配置文件里抽出来）
ROLLING_CFG: Dict[str, RollingThreeClassConfig] = {
    # 这里先给出你刚刚定的大致值，后续你可以从外部注入
    "SMB": RollingThreeClassConfig(
        window_days=8 * 252,
        half_life_days=2 * 252,
        prob_window_days=1 * 252,
        q_low=1.0 / 3.0,
        q_high=2.0 / 3.0,
        min_samples=500,
    ),
    "HML": RollingThreeClassConfig(
        window_days=8 * 252,
        half_life_days=2 * 252,
        prob_window_days=1 * 252,
        q_low=1.0 / 3.0,
        q_high=2.0 / 3.0,
        min_samples=500,
    ),
    "QMJ": RollingThreeClassConfig(
        window_days=8 * 252,
        half_life_days=2 * 252,
        prob_window_days=1 * 252,
        q_low=1.0 / 3.0,
        q_high=2.0 / 3.0,
        min_samples=500,
    ),
    "10YBOND": RollingThreeClassConfig(
        window_days=5 * 252,
        half_life_days=1.5 * 252,
        prob_window_days=1 * 252,
        q_low=1.0 / 3.0,
        q_high=2.0 / 3.0,
        min_samples=400,
    ),
    # "MKT": RollingThreeClassConfig(
    #     window_days=8 * 252,
    #     half_life_days=4 * 252,
    #     prob_window_days=3 * 252,
    #     q_low=1.0 / 3.0,
    #     q_high=2.0 / 3.0,
    #     min_samples=500,
    # ),
}


@dataclass
class FactorViewResult:
    """
    因子层视图计算结果，供 BL 使用。
    """
    # 每个因子 → Rolling 统计结果
    stats: Dict[str, ThreeClassStats]

    # 使用 Rolling payoff + 选定 softprob 得到的最终 ER
    er_final: Dict[str, float]

    # 仅 Rolling 得到的 ER（softprob_stat × payoff），可以用于分析 / debug
    er_stat: Dict[str, float]

    # 仅 ML 得到的 ER（softprob_ml × payoff），仅对有 ML 的因子有值
    er_ml: Dict[str, float]

    # 最终给 BL 用的 softprob（可能来自 Rolling，也可能来自 ML）
    softprob_final: Dict[str, np.ndarray]

    # 最终给 BL 用的 label_to_ret（目前全部来自 Rolling）
    label_to_ret: Dict[str, np.ndarray]


def compute_factor_views(
    factor_future_ret_map: Dict[str, pd.Series],
    as_of: pd.Timestamp,
    dataset_builder,
    alpha_10ybond: float = 0.3,
) -> FactorViewResult:
    """
    核心入口：给定各因子的 future_ret 序列和 as_of，
    计算 Rolling + ML 融合后的因子期望收益率与三分类视图。

    参数
    ----
    factor_future_ret_map:
        形如 {"MKT": Series, "SMB": Series, ...}，
        index 为日期，值为对应因子的 future_ret。
        本函数不关心 future_ret 如何构造，只要“往前看”时不泄露未来即可。
    as_of:
        当前截面日期（调仓日 / 预测日）。
    dataset_builder:
        你项目中现有的 DatasetBuilder 实例，
        会被传给 get_softprob_dict 用于 XGBoost 推理。
    alpha_10ybond:
        10YBOND 的 ER_mix = (1 - alpha) * ER_stat + alpha * ER_ml 中的 alpha。
    """
    if not isinstance(as_of, pd.Timestamp):
        as_of = pd.Timestamp(as_of)

    # ---------- 1. Rolling 统计（所有配置了 ROLLING_CFG 的因子） ----------
    stats: Dict[str, ThreeClassStats] = {}
    er_stat: Dict[str, float] = {}
    label_to_ret: Dict[str, np.ndarray] = {}
    softprob_stat: Dict[str, np.ndarray] = {}

    for factor, cfg in ROLLING_CFG.items():
        if factor not in factor_future_ret_map:
            continue

        future_ret = factor_future_ret_map[factor]
        res = compute_three_class_stats_for_date(future_ret, as_of, cfg)
        if res is None:
            # 样本不足：保守起见，ER_stat = 0，softprob 用均匀分布
            stats[factor] = ThreeClassStats(
                q_low=np.nan,
                q_high=np.nan,
                label_to_ret=np.zeros(3, dtype=float),
                softprob=np.array([1 / 3, 1 / 3, 1 / 3], dtype=float),
                n_payoff_samples=0,
                n_prob_samples=0,
            )
        else:
            stats[factor] = res

        lt = stats[factor]["label_to_ret"]
        sp = stats[factor]["softprob"]

        label_to_ret[factor] = lt
        softprob_stat[factor] = sp
        er_stat[factor] = float(np.dot(sp, lt))

    # ---------- 2. ML softprob（只对有 ML 模型的因子调用） ----------
    # get_softprob_dict 内部会根据 trade_date、dataset_builder 调 XGBoost。
    trade_date_str = as_of.strftime("%Y-%m-%d")
    softprob_ml_all = get_softprob_dict(trade_date_str, dataset_builder=dataset_builder)

    er_ml: Dict[str, float] = {}

    # MKT：纯 ML softprob × Rolling payoff
    if "MKT" in softprob_ml_all and "MKT" in label_to_ret:
        sp_mkt = softprob_ml_all["MKT"]
        lt_mkt = label_to_ret["MKT"]
        er_ml["MKT"] = float(np.dot(sp_mkt, lt_mkt))
    # 10YBOND：需要 ML softprob 参与融合
    if "10YBOND" in softprob_ml_all and "10YBOND" in label_to_ret:
        sp_bond = softprob_ml_all["10YBOND"]
        lt_bond = label_to_ret["10YBOND"]
        er_ml["10YBOND"] = float(np.dot(sp_bond, lt_bond))

    # ---------- 3. 根据 FACTOR_SOURCE 选择最终 ER / softprob ----------
    er_final: Dict[str, float] = {}
    softprob_final: Dict[str, np.ndarray] = {}

    for factor, source in FACTOR_SOURCE.items():
        if factor == "GOLD":
            # GOLD 完全不进入 BL，这里直接跳过
            continue

        if factor not in label_to_ret:
            # 没有 Rolling 统计结果就没法给 BL 观点，跳过
            continue

        lt = label_to_ret[factor]

        if source == "rolling":
            sp = softprob_stat[factor]
            softprob_final[factor] = sp
            er_final[factor] = er_stat[factor]

        elif source == "ml":
            # 目前只有 MKT 走这条路：softprob 用 ML，payoff 用 Rolling
            sp_ml = softprob_ml_all.get(factor)
            if sp_ml is None:
                # 没有 ML softprob 就退化成统计
                sp_ml = softprob_stat[factor]
            sp_ml = np.asarray(sp_ml, dtype=float)
            softprob_final[factor] = sp_ml
            er_final[factor] = float(np.dot(sp_ml, lt))
            er_ml.setdefault(factor, er_final[factor])

        elif source == "mix":
            # 目前只有 10YBOND：ER_mix = (1 - alpha) * ER_stat + alpha * ER_ml
            sp_stat = softprob_stat[factor]
            er_s = er_stat[factor]

            sp_ml = softprob_ml_all.get(factor)
            if sp_ml is None or factor not in er_ml:
                # 如果 ML 侧没有结果，就退化为纯统计
                softprob_final[factor] = sp_stat
                er_final[factor] = er_s
            else:
                sp_ml = np.asarray(sp_ml, dtype=float)
                softprob_final[factor] = sp_ml  # softprob 给 BL 用 ML 的
                er_m = er_ml[factor]
                er_final[factor] = (1.0 - alpha_10ybond) * er_s + alpha_10ybond * er_m

        else:
            # "none" 或未知，暂不参与 BL
            continue

    return FactorViewResult(
        stats=stats,
        er_final=er_final,
        er_stat=er_stat,
        er_ml=er_ml,
        softprob_final=softprob_final,
        label_to_ret=label_to_ret,
    )
