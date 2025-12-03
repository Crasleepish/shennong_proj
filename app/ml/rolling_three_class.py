# app/ml/rolling_three_class.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, TypedDict

import numpy as np
import pandas as pd


@dataclass
class RollingThreeClassConfig:
    """
    Rolling 三分类配置：

    - window_days:      payoff / 分位数使用的长窗口长度（按“自然日”计）
    - half_life_days:   时间衰减半衰期（天），用于 payoff 与加权分位数
    - prob_window_days: softprob 使用的短窗口长度（天），不做时间加权
    - q_low/q_high:     分位点（例如 1/3, 2/3 或 0.25, 0.75）
    - min_samples:      至少多少条样本才开始滚动，否则返回 None
    """
    window_days: int
    half_life_days: float
    prob_window_days: int
    q_low: float = 1.0 / 3.0
    q_high: float = 2.0 / 3.0
    min_samples: int = 250


class ThreeClassStats(TypedDict):
    q_low: float
    q_high: float
    label_to_ret: np.ndarray      # shape (3,)
    softprob: np.ndarray          # shape (3,)
    n_payoff_samples: int
    n_prob_samples: int


def _weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: Sequence[float],
) -> np.ndarray:
    """简单加权分位数实现，weights >= 0。"""
    if values.size == 0:
        return np.array([0.0 for _ in quantiles], dtype=float)

    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]

    w_cum = np.cumsum(w)
    total = float(w_cum[-1])

    if total <= 0:
        # 极端情况：所有权重为 0
        return np.array([float(v[0]) for _ in quantiles], dtype=float)

    qs = []
    for q in quantiles:
        q = float(min(max(q, 0.0), 1.0))
        threshold = q * total
        idx = int(np.searchsorted(w_cum, threshold, side="left"))
        if idx >= len(v):
            idx = len(v) - 1
        qs.append(float(v[idx]))
    return np.array(qs, dtype=float)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    s = float(weights.sum())
    if s <= 0:
        return float(values.mean())
    return float((values * weights).sum() / s)


def compute_three_class_stats_for_date(
    future_ret: pd.Series,
    as_of: pd.Timestamp,
    cfg: RollingThreeClassConfig,
) -> Optional[ThreeClassStats]:
    """
    按“长窗口加权 payoff + 短窗口无权 softprob”的方案，计算三分类统计量。

    参数
    ----
    future_ret:
        index: datetime-like（交易日）
        value: 这里我们用“滚动过去 horizon_days 的收益”
    as_of:
        截面日期（调仓日）
    cfg:
        RollingThreeClassConfig

    返回
    ----
    ThreeClassStats 或 None（样本不足）
    """
    if future_ret.empty:
        return None

    as_of = pd.Timestamp(as_of)

    # ---------- 统一用“交易日索引”取窗口 ----------
    # 先截到 as_of 之前（含 as_of），再按行数取最近 N 个样本
    past_all = future_ret.loc[:as_of].dropna()

    # ---------- Step 1: payoff 样本 + 时间衰减权重（长窗口，按交易日计数） ----------
    if len(past_all) < cfg.min_samples:
        # 样本太少，直接放弃/返回 None，让上游决定 fallback 策略
        return None

    # 最近 window_days 个交易日的样本
    ret_payoff = past_all.iloc[-cfg.window_days:]
    n_payoff = len(ret_payoff)
    if n_payoff < cfg.min_samples:
        # 虽然总历史够，但最近 window_days 样本本身太少，也可以直接放弃
        return None

    # 时间衰减权重（half_life 仍然按“自然日”定义）
    age_days = (as_of - ret_payoff.index).days.astype(float)
    age_days = np.maximum(age_days, 0.0)
    w = np.exp(-np.log(2.0) * age_days / float(cfg.half_life_days))
    w = w.to_numpy().astype(float)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return None
    w_norm = w / w_sum

    # ---------- Step 2: weighted quantile 边界 ----------
    q_low_val, q_high_val = _weighted_quantiles(
        values=ret_payoff.values,
        weights=w_norm,
        quantiles=[cfg.q_low, cfg.q_high],
    )

    # ---------- Step 3: label_to_ret（加权 payoff） ----------
    vals = ret_payoff.values.astype(float)
    bucket0 = vals <= q_low_val
    bucket2 = vals >= q_high_val
    bucket1 = (~bucket0) & (~bucket2)

    def weighted_mean(v, w):
        if v.size == 0:
            return 0.0
        ws = float(w.sum())
        return float((v * w).sum() / ws) if ws > 0 else float(v.mean())

    w_arr = w_norm
    r0 = weighted_mean(vals[bucket0], w_arr[bucket0])
    r1 = weighted_mean(vals[bucket1], w_arr[bucket1])
    r2 = weighted_mean(vals[bucket2], w_arr[bucket2])
    label_to_ret = np.array([r0, r1, r2], dtype=float)

    # ---------- Step 4: softprob 使用短窗口 + 频率（同样按交易日计数） ----------
    # 用 past_all 的尾部 prob_window_days 条作为短窗口样本
    ret_prob = past_all.iloc[-cfg.prob_window_days:]
    n_prob = len(ret_prob)

    if n_prob <= 0:
        # 没有短窗口样本，退化 softprob
        softprob = np.array(
            [cfg.q_low, cfg.q_high - cfg.q_low, 1.0 - cfg.q_high],
            dtype=float,
        )
    else:
        v_prob = ret_prob.values.astype(float)
        bucket0_prob = v_prob <= q_low_val
        bucket2_prob = v_prob >= q_high_val
        bucket1_prob = (~bucket0_prob) & (~bucket2_prob)

        p0 = bucket0_prob.sum() / n_prob
        p1 = bucket1_prob.sum() / n_prob
        p2 = bucket2_prob.sum() / n_prob
        softprob = np.array([p0, p1, p2], dtype=float)

    return {
        "q_low": float(q_low_val),
        "q_high": float(q_high_val),
        "label_to_ret": label_to_ret,
        "softprob": softprob,
        "n_payoff_samples": int(n_payoff),
        "n_prob_samples": int(n_prob),
    }