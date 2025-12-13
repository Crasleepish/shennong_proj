# app/ml/factor_black_litterman.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import logging  # NEW

from app.data_fetcher.factor_data_reader import FactorDataReader
from app.data_fetcher.csi_index_data_fetcher import CSIIndexDataFetcher
from app.ml.rolling_three_class import (
    RollingThreeClassConfig,
    ThreeClassStats,
    compute_three_class_stats_for_date,
)
from app.ml.inference import get_softprob_dict, get_model_val_loss, load_val_history_for_task
from app.ml.black_litterman_opt_util import compute_prior_mu_sigma

# 注意：load_fund_betas 在 black_litterman_opt_util.py 中，你那里已经写好了
from app.ml.black_litterman_opt_util import load_fund_betas
from app.ai.gold_view_llm import GoldViewLLM
from app.models.scaler_models.crowding_gold_scaler import GoldRiskScaler
from app.models.scaler_models.crowding_equity_scaler import EquityStyleCrowdingScaler
from app.models.regime.gold_regime_model import GoldRegimeModel

logger = logging.getLogger(__name__)


# 因子视图来源类型
FACTOR_SOURCE: Dict[str, str] = {
    "MKT": "ml",        # 纯 ML softprob × rolling payoff
    "SMB": "rolling",   # 纯 rolling
    "HML": "rolling",
    "QMJ": "rolling",
    "10YBOND": "mix",   # ER_mix = (1 - alpha) * ER_stat + alpha * ER_ml
    # GOLD 不在因子 BL 里处理
}


# 各因子的 Rolling 配置
ROLLING_CFG: Dict[str, RollingThreeClassConfig] = {
    "MKT": RollingThreeClassConfig(
        window_days=6 * 252,       # payoff/分位数窗口：约 6 年
        half_life_days=3 * 252,    # 半衰期约 3 年
        prob_window_days=3 * 252,  # softprob：约 3 年 (softprob由ML模型预测，此处暂无影响)
        q_low=0.25,
        q_high=0.75,
        min_samples=700,
    ),
    "SMB": RollingThreeClassConfig(
        window_days=7 * 252,
        half_life_days=3.5 * 252,
        prob_window_days=3 * 252,
        q_low=1.0 / 3.0,
        q_high=2.0 / 3.0,
        min_samples=700,
    ),
    "HML": RollingThreeClassConfig(
        window_days=7 * 252,
        half_life_days=3.5 * 252,
        prob_window_days=3 * 252,
        q_low=1.0 / 3.0,
        q_high=2.0 / 3.0,
        min_samples=700,
    ),
    "QMJ": RollingThreeClassConfig(
        window_days=7 * 252,
        half_life_days=3.5 * 252,
        prob_window_days=3 * 252,
        q_low=1.0 / 3.0,
        q_high=2.0 / 3.0,
        min_samples=700,
    ),
    "10YBOND": RollingThreeClassConfig(
        window_days=4 * 252,
        half_life_days=2 * 252,
        prob_window_days=2 * 252,
        q_low=0.25,
        q_high=0.75,
        min_samples=500,
    ),
    "GOLD": RollingThreeClassConfig(
        window_days=4 * 252,       # 约 4 年
        half_life_days=2 * 252,    # 半衰期约 2 年
        prob_window_days=2 * 252,  # 这里只是占位，对方差逻辑实际不使用
        q_low=0.25,
        q_high=0.75,
        min_samples=500,
    ),
}


@dataclass
class FactorViewResult:
    stats: Dict[str, ThreeClassStats]
    er_final: Dict[str, float]
    er_stat: Dict[str, float]
    er_ml: Dict[str, float]
    softprob_final: Dict[str, np.ndarray]
    label_to_ret: Dict[str, np.ndarray]


# ---------- 工具函数：构造因子日度收益 & 净值 ----------

def _build_factor_return_and_nav(
    trade_date: str,
    ten_year_bond_index_code: str,
    gold_index_code: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    构造因子日度收益矩阵 df_ret 和因子净值矩阵 df_nav。

    - MKT/SMB/HML/QMJ: 从 FactorDataReader.read_daily_factors 获取日度收益，
      然后 (1+ret).cumprod() 得到净值。
    - 10YBOND: 从 CSIIndexDataFetcher.get_data_by_code_and_date 获取 'close'，
      直接视为净值；日度收益为 close.pct_change()。
    - GOLD:   从 CSIIndexDataFetcher.get_data_by_code_and_date 获取 'close'，
      直接视为净值；日度收益为 close.pct_change()。
    """
    factor_data_reader = FactorDataReader()
    csi_index_data_fetcher = CSIIndexDataFetcher()

    # 先拿到截至 trade_date 的因子日度收益（MKT/SMB/HML/QMJ）
    df_factors = factor_data_reader.read_daily_factors(
        start=None,
        end=trade_date,
    )[["MKT", "SMB", "HML", "QMJ"]]

    df_factors.index = pd.to_datetime(df_factors.index)
    df_factors = df_factors.sort_index()
    df_factors = df_factors.dropna(how="all")

    # 10YBOND 净值
    df_bond = csi_index_data_fetcher.get_data_by_code_and_date(
        code=ten_year_bond_index_code,
        end=trade_date,
    )
    df_bond = df_bond[["date", "close"]].dropna().set_index("date").sort_index()
    df_bond.index = pd.to_datetime(df_bond.index)

    # 10YBOND 日度收益
    df_bond_ret = df_bond["close"].pct_change().rename("10YBOND")

    # GOLD 净值（使用上金所 Au99.99.SGE）
    try:
        df_gold = csi_index_data_fetcher.get_data_by_code_and_date(
            code=gold_index_code,
            end=trade_date,
        )
        df_gold = df_gold[["date", "close"]].dropna().set_index("date").sort_index()
        df_gold.index = pd.to_datetime(df_gold.index)
        df_gold_ret = df_gold["close"].pct_change().rename("GOLD")
    except Exception as e:
        logger.warning("获取 GOLD(Au99.99.SGE) 数据失败，将不在本次 BL 中使用 GOLD 因子: %s", e)
        df_gold = None
        df_gold_ret = None

    # 对齐日期：MKT/SMB/HML/QMJ + 10YBOND +（可选）GOLD
    ret_parts = [df_factors, df_bond_ret]
    if df_gold_ret is not None:
        ret_parts.append(df_gold_ret)

    df_ret = pd.concat(ret_parts, axis=1).sort_index()

    # 净值矩阵：四个风格因子用 (1+r).cumprod()，利率和黄金用价格本身
    df_nav = (1.0 + df_ret[["MKT", "SMB", "HML", "QMJ"]]).cumprod()

    # 对齐 10YBOND、GOLD 的净值到 df_nav 的索引
    df_nav["10YBOND"] = df_bond["close"].reindex(df_nav.index)

    if df_gold is not None:
        df_nav["GOLD"] = df_gold["close"].reindex(df_nav.index)

    # 再次按列顺序统一并去掉全空行
    desired_cols = [c for c in ["MKT", "SMB", "HML", "QMJ", "10YBOND", "GOLD"] if c in df_ret.columns]
    df_ret = df_ret[desired_cols].dropna(how="all")

    desired_cols_nav = [c for c in ["MKT", "SMB", "HML", "QMJ", "10YBOND", "GOLD"] if c in df_nav.columns]
    df_nav = df_nav[desired_cols_nav].dropna(how="all")

    return df_ret, df_nav


def _compute_factor_future_rets(
    df_ret: pd.DataFrame,
    horizon_days: int = 20,
) -> Dict[str, pd.Series]:
    """
    根据日度收益 df_ret 构造「滚动最近 horizon_days 日收益」序列。

    注意：这里是 *过去* horizon_days 的滚动收益，
    而不是 “未来 20 日” 的 forward ret。
    """
    factor_future_ret_map: Dict[str, pd.Series] = {}

    for col in df_ret.columns:
        # 简单用复利：(1+r).prod - 1
        rolling_prod = (1.0 + df_ret[col]).rolling(horizon_days).apply(
            lambda x: float(np.prod(x) - 1.0),
            raw=False,
        )
        factor_future_ret_map[col] = rolling_prod.dropna()

    return factor_future_ret_map


def omega_scale_from_val_loss(
    val_loss: float,
    history: Sequence[float],
    s_min: float = 0.5,
    s_max: float = 5.0,
    min_history: int = 12,
    k: float = 2.4,
) -> float:
    """
    根据当前 val_loss 与历史 val_loss 序列的相对位置（分位数 p）计算
    Black–Litterman 观点方差的缩放因子 s。

    设计目标：
    - 用“相对历史表现”而不是绝对 loss 作为模型置信度度量；
    - p <= 0.2：模型处于历史相对较好的 20% → 轻微增强观点（减小 ω，s ∈ [s_min, 1] 线性变化）；
    - 0.2 < p < 0.8：模型表现中性 → s ≈ 1（不动）；
    - p >= 0.8：模型处于历史较差的 20% → 指数放大 ω（s >= 1），
      特别是 p > 0.9 时 s 很大，让 BL 结果几乎只受先验影响。

    参数
    ----
    val_loss : float
        当前模型在最近验证集上的 logloss。
    history : Sequence[float]
        历史 val_loss 序列（建议按时间顺序或乱序都可以，只要是同一任务的历史记录）。
    s_min : float
        缩放下限，在 p 很小时最多把 omega 缩到 s_min 倍（避免过度自信）。
    s_max : float
        缩放上限，在 p 很大时最多把 omega 放到 s_max 倍（避免数值爆炸）。
    min_history : int
        计算分位数所需的最小历史样本数，少于该值则认为历史不足，返回 s=1。
    k : float
        指数放大段的斜率参数。默认 k=2.4：
        - p=0.8 → s=1
        - p≈0.9 → s≈exp(k * ((0.9-0.8)/0.2)) ≈ exp(4 * 1/3) ≈ 3.8
        - p更高时 s 很快接近 s_max，从而让 BL 接近先验。

    返回
    ----
    float
        omega 缩放因子 s，用于：
            omega_new = s * omega_raw
    """
    import numpy as np
    import math

    # 1. 基本兜底：val_loss 或 history 无效时，直接不缩放
    if val_loss is None or not np.isfinite(val_loss):
        return 1.0

    hist = np.asarray(history, dtype=float)
    hist = hist[np.isfinite(hist)]

    if hist.size < min_history:
        # 历史样本太少，分位数不可靠 → 暂时不做缩放
        return 1.0

    # 2. 计算当前 val_loss 在历史中的经验分位数 p ∈ (0, 1)
    # 使用右侧 searchsorted，避免大量相等值时 p 过小
    hist_sorted = np.sort(hist)
    rank = np.searchsorted(hist_sorted, val_loss, side="right")
    # 简单经验分位数：rank / (N + 1)，保证在 (0, 1) 内
    p = rank / (hist_sorted.size + 1.0)

    # 3. 分段映射 p -> s
    # 3.1 p <= 0.2：线性从 s_min -> 1
    if p <= 0.2:
        # p=0   -> s = s_min
        # p=0.2 -> s = 1
        t = (0.2 - p) / 0.2  # t ∈ [0, 1]
        s = 1.0 - (1.0 - s_min) * t

    # 3.2 0.2 < p < 0.8：不动，s ≈ 1
    elif p < 0.8:
        s = 1.0

    # 3.3 p >= 0.8：指数放大，p>=0.9 时 s 已经比较大
    else:
        # 归一化到 [0,1]：p=0.8 -> g=0, p=1.0 -> g=1
        g = (p - 0.8) / 0.2
        g = max(0.0, min(g, 1.0))
        try:
            s = math.exp(k * g)
        except OverflowError:
            s = s_max

    # 4. 裁剪到 [s_min, s_max]
    s = max(min(s, s_max), s_min)
    return float(s)

# ---------- Rolling + ML 因子视图计算 ----------

def compute_factor_views(
    trade_date: str,
    dataset_builder,
    ten_year_bond_index_code: str,
    gold_index_code: str,
    horizon_days: int = 20,
    alpha_10ybond: float = 0.3,
) -> Tuple[FactorViewResult, pd.DataFrame, pd.DataFrame]:
    """
    核心入口：给定 trade_date，计算 5 个因子的 Rolling+ML 视图。

    返回：
    - FactorViewResult：包含 er_final / er_stat / er_ml / label_to_ret / softprob_final
    - df_ret: 因子日度收益矩阵
    - df_nav: 因子净值矩阵（用于后面 compute_prior_mu_sigma）
    """
    as_of = pd.Timestamp(trade_date)

    # 1. 构造因子日度收益 & 净值
    df_ret, df_nav = _build_factor_return_and_nav(
        trade_date=trade_date,
        ten_year_bond_index_code=ten_year_bond_index_code,
        gold_index_code=gold_index_code,
    )

    # 2. 构造 “滚动 20 日收益” future_ret
    factor_future_ret_map = _compute_factor_future_rets(
        df_ret=df_ret,
        horizon_days=horizon_days,
    )

    # 3. Rolling 统计
    stats: Dict[str, ThreeClassStats] = {}
    er_stat: Dict[str, float] = {}
    label_to_ret: Dict[str, np.ndarray] = {}
    softprob_stat: Dict[str, np.ndarray] = {}

    for factor, cfg in ROLLING_CFG.items():
        if factor not in factor_future_ret_map:
            continue

        future_ret = factor_future_ret_map[factor]
        res = compute_three_class_stats_for_date(
            future_ret=future_ret,
            as_of=as_of,
            cfg=cfg,
        )

        if res is None:
            # 样本不足：ER_stat 视为 0，softprob 用均匀分布
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

    # 4. ML softprob（MKT & 10YBOND）
    softprob_ml_all = get_softprob_dict(
        trade_date,
        dataset_builder=dataset_builder,
        horizon_days=10,  # 这里沿用你现有逻辑即可
    )
    er_ml: Dict[str, float] = {}

    # MKT：纯 ML softprob × rolling payoff
    if "MKT" in softprob_ml_all and "MKT" in label_to_ret:
        sp_mkt = np.asarray(softprob_ml_all["MKT"], dtype=float)
        lt_mkt = label_to_ret["MKT"]
        er_ml["MKT"] = float(np.dot(sp_mkt, lt_mkt))

    # 10YBOND：softprob_ml × rolling payoff
    if "10YBOND" in softprob_ml_all and "10YBOND" in label_to_ret:
        sp_bond = np.asarray(softprob_ml_all["10YBOND"], dtype=float)
        lt_bond = label_to_ret["10YBOND"]
        er_ml["10YBOND"] = float(np.dot(sp_bond, lt_bond))

    # 5. 根据 FACTOR_SOURCE 选择最终 ER / softprob（MKT/SMB/HML/QMJ/10YBOND）
    er_final: Dict[str, float] = {}
    softprob_final: Dict[str, np.ndarray] = {}

    for factor, source in FACTOR_SOURCE.items():
        if factor not in label_to_ret:
            continue

        lt = label_to_ret[factor]

        if source == "rolling":
            sp = softprob_stat[factor]
            softprob_final[factor] = sp
            er_final[factor] = er_stat[factor]

        elif source == "ml":
            sp_ml = softprob_ml_all.get(factor)
            if sp_ml is None:
                sp_ml = softprob_stat[factor]
            sp_ml = np.asarray(sp_ml, dtype=float)
            softprob_final[factor] = sp_ml
            er_final[factor] = float(np.dot(sp_ml, lt))
            er_ml.setdefault(factor, er_final[factor])

        elif source == "mix":
            sp_stat = softprob_stat[factor]
            er_s = er_stat[factor]
            sp_ml = softprob_ml_all.get(factor)

            if sp_ml is None or factor not in er_ml:
                # ML 不可用 → 退化成纯 rolling
                softprob_final[factor] = sp_stat
                er_final[factor] = er_s
            else:
                sp_ml = np.asarray(sp_ml, dtype=float)
                softprob_final[factor] = sp_ml  # softprob 给 BL 用 ML 的
                er_m = er_ml[factor]
                er_final[factor] = (1.0 - alpha_10ybond) * er_s + alpha_10ybond * er_m

        else:
            # "none" 或其它，当前不处理
            continue

    # 6. GOLD：使用 LLM 观点，不走 FACTOR_SOURCE
    if "GOLD" in df_nav.columns:
        try:
            gv = GoldViewLLM()
            res = gv.generate_view(gold_index_code, as_of.date())
        except Exception as e:
            logger.warning("调用 GoldViewLLM 失败，将 GOLD 视图设为 None: %s", e)
            res = None

        er_gold: float | None = None
        if res is not None:
            er_candidate = getattr(res, "expected_return", None)
            view_flag = getattr(res, "view", None)
            if er_candidate is not None and view_flag != "no_view":
                er_gold = float(er_candidate)

        if er_gold is not None:
            er_final["GOLD"] = er_gold
        else:
            # 显式标记 GOLD 有 key 但无观点，便于后续 BL 中按 None 退化为先验
            er_final["GOLD"] = None  # type: ignore

    fv = FactorViewResult(
        stats=stats,
        er_final=er_final,
        er_stat=er_stat,
        er_ml=er_ml,
        softprob_final=softprob_final,
        label_to_ret=label_to_ret,
    )

    return fv, df_ret, df_nav


# ---------- 因子空间 Black–Litterman ----------

def _factor_bl_posterior(
    mu_prior: np.ndarray,
    Sigma_prior: np.ndarray,
    factor_list: List[str],
    er_view: Dict[str, float],
    tau: float,
    variance_raw: np.ndarray,
    view_var_scale: float,
    prior_mix: float,
    horizon_days: int,
    omega_view_scale: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于因子层先验 (mu_prior, Sigma_prior) 和视图 (er_view, variance_raw)
    计算 Black–Litterman 的因子后验 (mu_post, Sigma_post)。

    设计要点：
    - 视图使用绝对期望收益（abs ER），即 er_view[f] 表示因子 f 在 horizon_days 内的期望收益。
    - GOLD 特殊处理：
        * 若 er_view["GOLD"] 不为 None，则将 GOLD 的先验改为固定 3%/年（折算为 horizon_days）
          并以此先验参与 BL；
        * 若 er_view["GOLD"] 为 None，则不改 mu_prior 对应维度，等价于该因子“仅使用统计先验”。
    - Omega 的构造为逐因子 loop：
        omega_i = view_part_i + prior_part_i
        其中：
            view_part_i  = view_var_scale * variance_raw[i]
            prior_part_i = prior_mix * sigma_prior_diag[i]
        若提供 omega_view_scale 且包含该因子：
            view_part_i *= omega_view_scale[f]
        即：模型置信度缩放只作用于“视图部分”，不影响先验方差部分。
    """
    n = len(factor_list)
    mu_prior = mu_prior.reshape(-1, 1)  # 列向量
    Sigma_prior = np.asarray(Sigma_prior, dtype=float)

    if variance_raw.shape[0] != n:
        raise ValueError(
            f"variance_raw length {variance_raw.shape[0]} "
            f"does not match factor_list length {n}."
        )

    # 1. P, q 组装（每个因子一条 view：P = I）
    P = np.eye(n, dtype=float)
    q = np.zeros((n, 1), dtype=float)

    # GOLD 的长期先验：3%/年 → horizon_days 日收益
    annual_prior_gold = 0.03
    h_days = float(horizon_days)
    prior_gold_h = annual_prior_gold * (h_days / 252.0)

    for i, f in enumerate(factor_list):
        view_val = er_view.get(f, None)

        if f == "GOLD":
            if view_val is None:
                # 无 LLM 观点：直接使用 compute_prior_mu_sigma 得到的先验均值
                q[i, 0] = float(mu_prior[i, 0])
            else:
                # 有 LLM 观点：把 GOLD 的先验强制重置为固定 3%/年对应的 horizon_days 收益
                mu_prior[i, 0] = prior_gold_h
                q[i, 0] = float(view_val)
        else:
            # 其它因子：保持原逻辑，无观点时退化为先验
            if view_val is None:
                q[i, 0] = float(mu_prior[i, 0])
            else:
                q[i, 0] = float(view_val)

    # 2. Omega：逐因子构造 view_part + prior_part，并支持按因子缩放视图部分
    sigma_prior_diag = np.diag(Sigma_prior).astype(float)
    variance_raw = np.asarray(variance_raw, dtype=float)

    if omega_view_scale is None:
        omega_view_scale = {}

    omega_diag = np.zeros(n, dtype=float)

    for i, f in enumerate(factor_list):
        # 视图部分：rolling 方差 * 系数
        view_part = view_var_scale * variance_raw[i]

        # 模型置信度缩放（仅作用于视图部分）
        s = float(omega_view_scale.get(f, 1.0))
        view_part *= s

        # 先验部分：先验方差 * 混合系数（不受模型置信度影响）
        prior_part = prior_mix * sigma_prior_diag[i]

        omega_i = view_part + prior_part
        omega_diag[i] = max(float(omega_i), 1e-10)  # 数值安全

    Omega = np.diag(omega_diag)
    Omega_inv = np.diag(1.0 / omega_diag)

    # 3. BL 后验（保留 tau 控制先验强度）
    #   Sigma_post = inv( inv(tau*Sigma_prior) + P^T Omega^{-1} P )
    #   mu_post    = Sigma_post ( inv(tau*Sigma_prior) mu_prior + P^T Omega^{-1} q )
    tauSigma = tau * Sigma_prior
    tauSigma_inv = np.linalg.inv(tauSigma)

    middle = P.T @ Omega_inv @ P
    Sigma_post = np.linalg.inv(tauSigma_inv + middle)
    mu_post = Sigma_post @ (tauSigma_inv @ mu_prior + P.T @ Omega_inv @ q)

    return mu_post.flatten(), Sigma_post


def _compute_factor_view_variance_raw(
    df_ret: pd.DataFrame,
    factor_list: List[str],
    as_of: pd.Timestamp,
    horizon_days: int,
) -> np.ndarray:
    """
    在因子空间估计每条 view 的原始不确定性 variance_raw（20 日尺度）。

    关键点：
    - 先从日度收益 df_ret 构造“滚动 horizon_days 日收益”（近似 20 日收益）：
        R_t(h) = prod_{k=0..h-1} (1 + r_{t-k}) - 1
      这样 variance_raw 和 Sigma_prior 的量纲都是“h 日收益”的方差。
    - 然后在这一条 20 日收益序列上，和 rolling_three_class 一样：
        * 取最近 window_days 个“交易日”样本；
        * 按 half_life_days 构造时间衰减权重；
        * 计算加权方差，作为 variance_raw_f。
    - 样本不足时，用该因子全历史 20 日收益的普通方差 × penalty_scale。
    """
    as_of = pd.Timestamp(as_of)
    variance_raw = np.zeros(len(factor_list), dtype=float)

    penalty_scale = 5.0  # 样本不足时的惩罚系数

    for i, factor in enumerate(factor_list):
        cfg = ROLLING_CFG.get(factor)
        if cfg is None:
            variance_raw[i] = 0.0
            continue

        if factor not in df_ret.columns:
            variance_raw[i] = 0.0
            continue

        # 该因子完整的日度收益（交易日索引）
        ret_daily = df_ret[factor].dropna()
        if ret_daily.empty:
            variance_raw[i] = 0.0
            continue

        # === 1) 从日度收益构造“滚动 horizon_days 日收益” ===
        # 这里用 (1+r).rolling(h).apply(prod)-1，和你其它地方的 20 日收益口径保持一致（线性收益）
        one_plus = (1.0 + ret_daily).astype(float)
        ret_h = one_plus.rolling(horizon_days).apply(
            lambda x: float(np.prod(x) - 1.0),
            raw=True,
        ).dropna()

        if ret_h.empty:
            variance_raw[i] = 0.0
            continue

        # === 2) 截到 as_of 之前的所有“h 日收益”，再取最近 window_days 个（按交易日计数）===
        past_all = ret_h.loc[:as_of].dropna()
        if past_all.empty:
            variance_raw[i] = 0.0
            continue

        ret_payoff = past_all.iloc[-cfg.window_days:]
        n = ret_payoff.shape[0]

        if n >= cfg.min_samples:
            # 样本充足：正常按“加权方差 / 有效样本数”估计 variance_raw
            age_days = (as_of - ret_payoff.index).days.astype(float)
            age_days = np.maximum(age_days, 0.0)

            w = np.exp(-np.log(2.0) * age_days / float(cfg.half_life_days))
            w = w.to_numpy().astype(float)
            w_sum = float(w.sum())
            if w_sum <= 0:
                variance_raw[i] = 0.0
                continue

            w_norm = w / w_sum
            vals = ret_payoff.values.astype(float)

            m = float((w_norm * vals).sum())
            var_w = float((w_norm * (vals - m) ** 2).sum())
            variance_raw[i] = var_w

        else:
            # 样本不足：提高这条 view 的不确定性，让 BL 更靠近因子先验
            hist_vals = ret_h.loc[ret_h.index < as_of].values.astype(float)

            if hist_vals.size <= 1:
                variance_raw[i] = 0.0
            else:
                var_all = float(np.var(hist_vals, ddof=1))
                variance_raw[i] = penalty_scale * var_all

    return variance_raw


def compute_factor_bl_posterior(
    trade_date: str,
    dataset_builder,
    ten_year_bond_index_code: str,
    gold_index_code: str,
    horizon_days: int = 20,
    lookback_years: float = 8.0,
    tau: float = 1.0,
    alpha_10ybond: float = 0.3,
    view_var_scale: float = 0.7,
    prior_mix: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, List[str], FactorViewResult]:
    """
    综合因子 Rolling+ML 视图 + 因子先验，在因子空间做 BL。

    参数：
    - trade_date: 调仓日（字符串）
    - dataset_builder: DatasetBuilder 实例
    - ten_year_bond_index_code: 10 年期国债指数代码
    - horizon_days: 视图 horizon（天数），如 20 日
    - lookback_years: 先验窗口长度（年）
    - tau: BL 中的 tau
    - alpha_10ybond: 10YBOND 中 ML 观点的权重
    - view_var_scale: 放大“视图自身不确定性 variance_raw”的系数
    - prior_mix: 将先验方差混入 Omega 的比例

    返回：
    - mu_factor_post: (n_factors,)
    - Sigma_factor_post: (n_factors, n_factors)
    - factor_list: ['MKT','SMB','HML','QMJ','10YBOND', 'GOLD']（视数据而定）
    - factor_views: FactorViewResult（便于 debug）
    """
    trade_date_str = pd.Timestamp(trade_date).strftime("%Y-%m-%d")

    # 1. Rolling + ML 视图（因子层 ER / softprob）
    factor_views, df_ret, df_nav = compute_factor_views(
        trade_date=trade_date_str,
        dataset_builder=dataset_builder,
        ten_year_bond_index_code=ten_year_bond_index_code,
        gold_index_code=gold_index_code,
        horizon_days=horizon_days,
        alpha_10ybond=alpha_10ybond,
    )

    # 2. 因子先验（8 年等），这里的 mu_prior 是“统计先验”
    mu_prior, Sigma_prior, factor_list = compute_prior_mu_sigma(
        price_df=df_nav,
        horizon_days=horizon_days,
        lookback_years=lookback_years,
        method="linear",
    )

    # 3. 视图 ER（只对有 prior 的因子做 BL），使用“绝对期望收益 abs ER”
    er_view: Dict[str, float] = {
        f: factor_views.er_final[f]
        for f in factor_list
        if f in factor_views.er_final
    }

    # 4. 视图自身不确定性 variance_raw（因子空间）
    as_of = pd.Timestamp(trade_date_str)
    variance_raw = _compute_factor_view_variance_raw(
        df_ret=df_ret,
        factor_list=factor_list,
        as_of=as_of,
        horizon_days=horizon_days,
    )

    # 5. 视图模型置信度缩放（MKT / 10YBOND） → omega_view_scale
    omega_view_scale: Dict[str, float] = {}
    try:
        # get_model_val_loss 返回 {task_name: val_loss}
        val_loss_dict = get_model_val_loss(trade_date_str)

        # 因子名 → 训练任务名 的映射
        factor_task_map = {
            "MKT": "mkt_tri",
            "10YBOND": "10Ybond_tri",
        }

        for f, task in factor_task_map.items():
            if f not in factor_list:
                continue

            val_loss = None
            if isinstance(val_loss_dict, dict):
                # 优先按 task 名取，如果你实际实现是按因子名存的，也可以自行调整这里的 key
                val_loss = val_loss_dict.get(task) or val_loss_dict.get(f)

            if val_loss is None:
                continue

            val_hist = load_val_history_for_task(task)
            if val_hist is None or len(val_hist) == 0:
                continue

            s = omega_scale_from_val_loss(val_loss=float(val_loss), history=val_hist)
            omega_view_scale[f] = float(s)
    except Exception as e:
        logger.warning("计算 omega_view_scale 失败，将不对视图方差做模型置信度缩放: %s", e)
        omega_view_scale = {}

    # 6. 因子 BL 后验（使用新的 Omega 逻辑 + omega_view_scale）
    mu_factor_post, Sigma_factor_post = _factor_bl_posterior(
        mu_prior=mu_prior,
        Sigma_prior=Sigma_prior,
        factor_list=factor_list,
        er_view=er_view,
        tau=tau,
        variance_raw=variance_raw,
        view_var_scale=view_var_scale,
        prior_mix=prior_mix,
        horizon_days=horizon_days,
        omega_view_scale=omega_view_scale,
    )

    # 7. GOLD Regime 权重：保留 3%/年结构性收益，只对“周期溢价”部分加权
    if "GOLD" in factor_list:
        try:
            as_of_date = pd.Timestamp(trade_date_str).date()

            # 使用 CSIIndexDataFetcher 作为黄金价格数据源
            gold_price_fetcher = CSIIndexDataFetcher()
            regime_model = GoldRegimeModel(gold_price_fetcher=gold_price_fetcher)

            w_regime = float(regime_model.get_weight(as_of_date))
        except Exception as e:
            logger.warning(
                "GoldRegimeModel 计算失败，将不对 GOLD 周期溢价做缩放: %s", e
            )
            w_regime = 1.0

        # 数值兜底
        if not np.isfinite(w_regime) or w_regime < 0.0:
            w_regime = 1.0

        idx_gold = factor_list.index("GOLD")

        # 固定 GOLD 长期结构性先验：3%/年 → horizon_days 日收益
        annual_struct_gold = 0.03
        prior_struct_h = annual_struct_gold * (float(horizon_days) / 252.0)

        # 当前 BL 得到的 GOLD 后验视图（包含结构 + 周期）
        mu_g = float(mu_factor_post[idx_gold])

        # 将 BL 后验拆成：结构性 + 周期性偏移
        mu_cycle = mu_g - prior_struct_h

        # 只让周期性那一部分乘以 w_regime，结构性 3%/年保持不动
        mu_factor_post[idx_gold] = prior_struct_h + w_regime * mu_cycle

        logger.info(
            "GOLD regime adjust: as_of=%s, w_regime=%.4f, mu_before=%.6f, "
            "mu_struct=%.6f, mu_after=%.6f",
            as_of_date,
            w_regime,
            mu_g,
            prior_struct_h,
            mu_factor_post[idx_gold],
        )

    # 8. 因子风格拥挤（SMB/HML/QMJ）risk_scaler：在后验 μ_f 上做缩放
    try:
        style_scalers: Dict[str, float] = {}
        scaler_eq = EquityStyleCrowdingScaler()
        as_of_date = pd.Timestamp(trade_date_str).date()
        style_scalers = scaler_eq.compute_for_date(as_of_date)  # 预期返回 {'SMB': s_smb, 'HML': s_hml, 'QMJ': s_qmj}
    except Exception as e:
        logger.warning("EquityStyleCrowdingScaler 计算失败，将不对 SMB/HML/QMJ 做缩放: %s", e)
        style_scalers = {}

    for f in ("SMB", "HML", "QMJ"):
        if f not in factor_list:
            continue
        s = float(style_scalers.get(f, 1.0))
        if s == 1.0:
            continue
        idx = factor_list.index(f)
        mu_factor_post[idx] *= s

    return mu_factor_post, Sigma_factor_post, factor_list, factor_views


def compute_asset_mu_from_factor_bl(
    asset_source_map: Dict[str],
    code_factors_map: Dict[str, List[str]],
    asset_codes: List[str],
    trade_date: str,
    dataset_builder,
    ten_year_bond_index_code: str,
    gold_index_code: str,
    horizon_days: int = 20,
    lookback_years: float = 8.0,
    tau: float = 1.0,
    alpha_10ybond: float = 0.3,
    beta_lookback_days: int = 250,
    view_var_scale: float = 0.7,
    prior_mix: float = 0.3,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """
    对给定 asset_codes，基于“因子层 BL → β 映射”给出资产层 μ_post。

    返回：
    - mu_asset_post: np.ndarray, shape (n_assets_kept,)
    - code_list_kept: 与 mu_asset_post 对齐的资产代码列表
    - mu_factor_post_dict: {factor: mu_post}（便于 debug）
    """
    trade_date_str = pd.Timestamp(trade_date).strftime("%Y-%m-%d")

    # 1. 因子层 BL 后验
    mu_factor_post, Sigma_factor_post, factor_list, factor_views = compute_factor_bl_posterior(
        trade_date=trade_date_str,
        dataset_builder=dataset_builder,
        ten_year_bond_index_code=ten_year_bond_index_code,
        gold_index_code=gold_index_code,
        horizon_days=horizon_days,
        lookback_years=lookback_years,
        tau=tau,
        alpha_10ybond=alpha_10ybond,
        view_var_scale=view_var_scale,
        prior_mix=prior_mix,
    )

    # 组一个方便看的 dict
    mu_factor_post_dict = {
        f: float(mu_factor_post[i]) for i, f in enumerate(factor_list)
    }

    factor_type_keys = [k for k, v in asset_source_map.items() if v == "factor"]
    df_betas_fund = load_fund_betas(
        codes=factor_type_keys,
        trade_date=trade_date_str,
        lookback_days=beta_lookback_days,
    )
    # 收集所有出现过的因子，并排序统一列顺序
    all_factors = sorted({f for fs in code_factors_map.values() for f in fs})
    beta_records = []
    for code, fs in code_factors_map.items():
        asset_type = asset_source_map.get(code)
        if not asset_type:
            raise ValueError(f"Asset {code} must specify asset type in asset_source_map")
        if asset_type != "factor" and len(fs) != 1:
            raise ValueError(f"Non-factor asset {code} must have exactly one factor, got {fs}")
        
        beta_row = {"code": code}
        if asset_type == "factor":
            row = df_betas_fund.loc[code]
            if row.empty:
                raise ValueError(f"Missing beta data for {code}")
            for f in fs:
                beta_row[f] = row[f]
        else:
            beta_row[fs[0]] = 1.0

        beta_records.append(beta_row)


    # 构造完整 beta DataFrame，未指定的因子填 0
    df_beta = pd.DataFrame(beta_records).fillna(0.0)
    df_beta = df_beta[["code"] + all_factors]

    # 期望 df_betas 至少有 ['code','MKT','SMB','HML','QMJ'] 这些列
    df_beta = df_beta.set_index("code", drop=True)

    # 取因子交集，防止某些因子在 β 里不存在
    factor_common = [f for f in factor_list if f in df_beta.columns]
    if not factor_common:
        raise ValueError("load_fund_betas 返回的列里没有任何因子列，无法映射 μ_factor_post 到资产层。")

    B = df_beta.loc[df_beta.index.intersection(asset_codes), factor_common]

    code_list_kept = list(B.index)
    if not code_list_kept:
        raise ValueError("给定 asset_codes 在 beta 表里都不存在。")

    # 对齐 μ_factor_post
    idx_map = {f: i for i, f in enumerate(factor_list)}
    mu_factor_vec = np.array(
        [mu_factor_post[idx_map[f]] for f in factor_common],
        dtype=float,
    )

    # 3. 映射到资产层：μ_asset_post = B @ μ_factor_post
    mu_asset_post = B.values @ mu_factor_vec  # shape (n_assets_kept,)

    return mu_asset_post, code_list_kept, mu_factor_post_dict
