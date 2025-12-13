# app/models/scaler_models/crowding_equity_scaler.py

import logging
from dataclasses import dataclass
import math
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from app.data_fetcher.factor_data_reader import FactorDataReader

logger = logging.getLogger(__name__)


@dataclass
class EquityStyleCrowdingConfig:
    """
    因子风格拥挤度（SMB/HML/QMJ）风险缩放器配置。

    参数
    ----
    z0 : float
        触发“极端风格 / 拥挤”判断的 z-score 阈值（使用绝对值 |z|）。
        当 |z_f| <= z0 时，不进行缩放（s_f = 1.0）。
    s_min : float
        缩放因子的下界，用于防止过度惩罚。
        例如 s_min = 0.5 表示最多将该因子观点缩到 50%。
    lookback_days_long : int
        用于计算滚动区间收益 z-score 的最长历史窗口（以交易日数量计）。
        例如 3000 ≈ 约 12 年历史（按 250 交易日/年粗略估算）。
    lookback_days_min : int
        允许启用拥挤度判断的最少有效交易日数量。
        若历史长度 < lookback_days_min，则对该因子不启用拥挤度缩放（s_f = 1.0）。
    window_length : int
        计算滚动区间收益 R_f(t; L) 的窗口长度（以交易日数量计），
        例如 252 ≈ 1 年。
    """

    z0: float = 1.5
    s_min: float = 0.5
    lookback_days_long: int = 3000
    lookback_days_min: int = 750
    window_length: int = 252  # 约 12 个月
    alpha: float = 0.8


class EquityStyleCrowdingScaler:
    """
    因子风格拥挤（SMB/HML/QMJ）风险缩放器。

    新逻辑要点
    ----------
    1. 使用因子“日收益率”构造滚动区间收益 R_f(t; L)，不再用 NAV 水平：
         R_f(t; L) = ∏_{τ=t-L+1}^t (1 + r_f(τ)) - 1

    2. 在最近 lookback_days_long 个交易日内，对所有可用的 R_f(t; L) 计算均值和标准差，
       并对当前调仓日对应的 R_f^* 计算 z-score：

         z_f = (R_f^* - μ_R,f) / σ_R,f

    3. 使用 |z_f| 与阈值 z0 做比较，得到因子级缩放系数 s_f：

         if |z_f| <= z0:  s_f = 1.0
         else:            s_f = max(1 / (1 + (|z_f| - z0)), s_min)

       解释为：极端风格 regime 下，降低对该因子长期 payoff/stat ER 的置信度，
       将 ER_f 绝对值往 0 收缩。

    4. 本类只输出 {factor: s_f}，具体如何作用到 ER_f 由因子视图构造模块处理：

         ER_f^view = s_f * ER_f^base
    """

    FACTORS: List[str] = ["SMB", "HML", "QMJ"]

    def __init__(self, config: Optional[EquityStyleCrowdingConfig] = None):
        self.config = config or EquityStyleCrowdingConfig()
        if self.config.s_min <= 0 or self.config.s_min > 1:
            logger.warning(
                "EquityStyleCrowdingScaler: s_min=%.4f 不在 (0, 1] 范围内，"
                "建议设置为 0 < s_min <= 1（当前值可能导致异常缩放）。",
                self.config.s_min,
            )

    # ---------- 主接口：给定调仓日，直接算 SMB/HML/QMJ 的缩放因子 ----------

    def compute_for_date(
        self,
        trade_date: str,
        factor_data_reader: Optional[FactorDataReader] = None,
    ) -> Dict[str, float]:
        """
        根据指定调仓日 trade_date，计算因子级风格拥挤缩放因子。

        参数
        ----
        trade_date : str
            调仓日，例如 "2025-12-31"。
            内部会以此作为 read_daily_factors 的 end 参数。
        factor_data_reader : FactorDataReader, 可选
            若为空，则在内部创建 FactorDataReader 实例。
            若上游已有实例，可传入以节省资源。

        返回
        ----
        Dict[str, float]
            形如：
                {
                    "SMB": s_smb,
                    "HML": s_hml,
                    "QMJ": s_qmj,
                }
            若数据不足或异常，对应因子返回 1.0 并记录 warning。
        """
        fdr = factor_data_reader or FactorDataReader()

        # 1) 读取全历史日收益率（到 trade_date 为止）
        df_factors = fdr.read_daily_factors(
            start=None,
            end=trade_date,
        )[["MKT", "SMB", "HML", "QMJ"]]

        df_factors.index = pd.to_datetime(df_factors.index)
        df_factors = df_factors.sort_index()
        df_factors = df_factors.dropna(how="all")

        if df_factors.empty:
            logger.warning(
                "EquityStyleCrowdingScaler.compute_for_date: 截止 %s 未获取到任何因子日收益数据，"
                "所有因子缩放因子退化为 1.0。",
                trade_date,
            )
            return {f: 1.0 for f in self.FACTORS}

        # 2) 仅保留最近 lookback_days_long 个交易日
        if len(df_factors) > self.config.lookback_days_long:
            df_factors = df_factors.iloc[-self.config.lookback_days_long :]

        # 3) 对每个因子分别计算 s_f（基于滚动区间收益的 z-score）
        scalers: Dict[str, float] = {}
        for factor in self.FACTORS:
            scalers[factor] = self._compute_single_factor_scaler(
                factor=factor,
                df_returns=df_factors,
                trade_date=trade_date,
            )

        return scalers

    # ---------- 内部：单个因子的缩放计算（滚动区间收益版） ----------

    def _compute_single_factor_scaler(
        self,
        factor: str,
        df_returns: pd.DataFrame,
        trade_date: str,
    ) -> float:
        """
        对某个因子（SMB/HML/QMJ）根据“滚动区间收益”的 z-score 计算 s_f。

        拥挤度指标：
            - 使用 window_length=L（默认 252 交易日）计算滚动区间收益：
                R_f(t; L) = ∏(1 + r_t) - 1
            - 在最近 lookback_days_long 窗口内的所有 R_f(t; L) 上计算 μ_R, σ_R
            - 当前调仓日对应 R_f^*，得到 z_f = (R_f^* - μ_R) / σ_R
        """
        if factor not in df_returns.columns:
            logger.warning(
                "EquityStyleCrowdingScaler._compute_single_factor_scaler: "
                "因子 '%s' 不在数据列中，缩放因子退化为 1.0。",
                factor,
            )
            return 1.0

        series_ret = df_returns[factor].dropna()
        if series_ret.empty:
            logger.warning(
                "EquityStyleCrowdingScaler._compute_single_factor_scaler: "
                "因子 '%s' 在窗口内无有效日收益数据，缩放因子退化为 1.0。",
                factor,
            )
            return 1.0

        # 至少需要 lookback_days_min 个交易日才启用拥挤度识别
        if len(series_ret) < self.config.lookback_days_min:
            logger.warning(
                "EquityStyleCrowdingScaler._compute_single_factor_scaler: "
                "因子 '%s' 有效历史样本数 %d < 最小要求 %d，"
                "不启用风格拥挤缩放，s_%s=1.0。",
                factor,
                len(series_ret),
                self.config.lookback_days_min,
                factor,
            )
            return 1.0

        as_of_ts = pd.to_datetime(trade_date)
        series_ret = series_ret[series_ret.index <= as_of_ts]
        if series_ret.empty:
            logger.warning(
                "EquityStyleCrowdingScaler._compute_single_factor_scaler: "
                "因子 '%s' 在 %s 之前没有有效数据，s_%s=1.0。",
                factor,
                trade_date,
                factor,
            )
            return 1.0

        if len(series_ret) < self.config.lookback_days_min:
            logger.warning(
                "EquityStyleCrowdingScaler._compute_single_factor_scaler: "
                "因子 '%s' 在 %s 之前有效样本数 %d < 最小要求 %d，"
                "不启用风格拥挤缩放，s_%s=1.0。",
                factor,
                trade_date,
                len(series_ret),
                self.config.lookback_days_min,
                factor,
            )
            return 1.0

        # 4) 计算滚动窗口收益 R_f(t; L)
        L = self.config.window_length

        if L <= 0:
            logger.warning(
                "EquityStyleCrowdingScaler._compute_single_factor_scaler: "
                "window_length=%d 非法，已退化为不缩放，s_%s=1.0。",
                L,
                factor,
            )
            return 1.0

        # 用简单收益构造累计收益：R = ∏(1 + r) - 1
        # rolling.apply 会把窗口对齐到窗口末端日期
        roll_ret = (1.0 + series_ret.astype(float)).rolling(
            window=L, min_periods=L
        ).apply(lambda x: float(np.prod(x) - 1.0), raw=False)

        roll_ret = roll_ret.dropna()
        if roll_ret.empty:
            logger.warning(
                "EquityStyleCrowdingScaler._compute_single_factor_scaler: "
                "因子 '%s' 在窗口长度 L=%d 下无法构造任何滚动区间收益，s_%s=1.0。",
                factor,
                L,
                factor,
            )
            return 1.0

        # 截止 trade_date 的所有 R_f(t; L)
        roll_ret = roll_ret[roll_ret.index <= as_of_ts]
        if roll_ret.empty:
            logger.warning(
                "EquityStyleCrowdingScaler._compute_single_factor_scaler: "
                "因子 '%s' 的滚动收益在 %s 之前均无有效值，s_%s=1.0。",
                factor,
                trade_date,
                factor,
            )
            return 1.0

        if len(roll_ret) < max(5, int(self.config.lookback_days_min / max(L, 1))):
            # 粗略要求：至少有若干个窗口样本才有意义
            logger.warning(
                "EquityStyleCrowdingScaler._compute_single_factor_scaler: "
                "因子 '%s' 的滚动收益样本数 %d 过少，s_%s=1.0。",
                factor,
                len(roll_ret),
                factor,
            )
            return 1.0

        R_current = roll_ret.iloc[-1]
        mu_R = roll_ret.mean()
        sigma_R = roll_ret.std(ddof=0)

        if not np.isfinite(mu_R) or not np.isfinite(sigma_R) or sigma_R <= 0:
            logger.warning(
                "EquityStyleCrowdingScaler._compute_single_factor_scaler: "
                "因子 '%s' 的滚动收益统计量异常（mu_R=%.6f, sigma_R=%.6f），s_%s=1.0。",
                factor,
                mu_R,
                sigma_R,
                factor,
            )
            return 1.0

        z = (R_current - mu_R) / sigma_R
        abs_z = float(abs(z))

        scaler = self._scaler_from_abs_z(abs_z=abs_z)

        logger.info(
            "EquityStyleCrowdingScaler: factor=%s, trade_date=%s, abs_z=%.4f, scaler=%.4f",
            factor,
            trade_date,
            abs_z,
            scaler,
        )

        return scaler

    # ---------- 内部：从 |z| 计算缩放因子 ----------

    def _scaler_from_abs_z(self, abs_z: float) -> float:
        """
        给定 |z|，按配置的 z0, s_min 计算缩放因子 s_f。

        公式：
            if abs_z <= z0:
                s = 1.0
            else:
                s = max( 1 / (1 + (abs_z - z0)), s_min )
        """
        z0 = self.config.z0
        s_min = self.config.s_min
        alpha = getattr(self.config, "alpha", 1.0)

        if not np.isfinite(abs_z):
            logger.warning(
                "EquityStyleCrowdingScaler._scaler_from_abs_z: abs_z 非有限值 (%.6f)，缩放因子退化为 1.0。",
                abs_z,
            )
            return 1.0

        if abs_z <= z0:
            return 1.0

        # abs_z > z0
        delta = abs_z - z0
        s = math.exp(-alpha * delta)  # 越大惩罚越狠

        # 理论上 s 应该在 (0, 1]，这里保持通用裁剪
        s = max(s_min, min(1.0, s))

        if s == 0.0:
            # 理论上不应为 0，这里避免极端情况完全杀死该风格
            logger.warning(
                "EquityStyleCrowdingScaler._scaler_from_abs_z: 计算得到 s=0，已强制调整为 s_min=%.4f。",
                s_min,
            )
            s = s_min

        return s
