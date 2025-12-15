# app/models/scaler_models/crowding_gold_scaler.py

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class GoldRiskScalerConfig:
    """
    黄金拥挤度 risk scaler 的配置参数。

    参数
    ----
    q80 : float
        拥挤度经验 80% 分位数（正常偏高但尚可接受）。
    q95 : float
        拥挤度经验 95% 分位数（极端拥挤区域阈值）。
    lam : float
        最大惩罚强度：
            - 当 crowding <= q80 时，scaler = 1
            - 当 crowding >= q95 时，scaler = 1 + lam
            - 中间区间线性插值
    """

    q80: float = 0.29
    q95: float = 0.35
    lam: float = 1.0


class GoldRiskScaler:
    """
    黄金拥挤度风险缩放器（基于 CFTC 数据）。

    逻辑说明
    --------
    1. 拥挤度定义：
        crowding = noncomm_net_all / open_interest_all

    2. 经验阈值：
        - q80 = 0.30
        - q95 = 0.36
      这两个值基于黄金市场长期经验分布，用于 v3.0 初版。

    3. 风险缩放函数：
        若 crowding <= q80:
            scaler = 1
        若 q80 < crowding <= q95:
            scaler = 1 + lam * (crowding - q80) / (q95 - q80)
        若 crowding > q95:
            scaler = 1 + lam

      通常 lam = 1，对应最大罚力度 2 倍（实际用于 ER 为 μ / scaler）。

    4. 无数据时：
        如果无法从 CFTC 获取 crowding（没有记录或数据无效），
        返回 1.0，并打印 warning。
    """

    def __init__(self, config: Optional[GoldRiskScalerConfig] = None):
        self.config = config or GoldRiskScalerConfig()
        if self.config.q95 <= self.config.q80:
            logger.warning(
                "GoldRiskScaler: 配置参数 q95 <= q80（q80=%.4f, q95=%.4f），"
                "将导致中间区间退化为阶跃。请检查配置。",
                self.config.q80,
                self.config.q95,
            )

    # --------- 对外主接口：从 CFTC 直接算 scaler ---------

    def compute_from_cftc(
        self,
        as_of: str,
        db: Optional[Session] = None,
    ) -> float:
        """
        给定调仓日 as_of，从 CFTC 数据中获取最近一期黄金拥挤度，
        并计算 risk scaler。

        参数
        ----
        as_of : str
            调仓日期，如 "2025-12-31"。
        db : Session, 可选
            可选的 SQLAlchemy Session；若为 None，将由 CftcGoldReader 管理。

        返回
        ----
        float
            risk scaler，>= 1.0。
            若 CFTC 数据缺失或无效，则返回 1.0，并打印 warning 日志。
        """
        crowding = CftcGoldReader.get_latest_crowding_before(as_of=as_of, db=db)

        if crowding is None:
            logger.warning(
                "GoldRiskScaler.compute_from_cftc: 在 %s 之前未获取到有效的黄金 CFTC 拥挤度数据，"
                "risk_scaler 退化为 1.0（不做风险惩罚）。",
                as_of,
            )
            return 1.0

        try:
            scaler = self.compute_from_crowding(crowding)
        except Exception as e:
            logger.warning(
                "GoldRiskScaler.compute_from_cftc: 计算 risk_scaler 失败，"
                "crowding=%.6f，错误：%s。risk_scaler 退化为 1.0。",
                crowding,
                e,
            )
            return 1.0

        return scaler

    # --------- 辅助接口：给定 crowding 直接算 scaler ---------

    def compute_from_crowding(self, crowding: float) -> float:
        """
        已知拥挤度 crowding，直接计算黄金 risk scaler。

        参数
        ----
        crowding : float
            拥挤度 noncomm_net_all / open_interest_all。
            理论上位于 [-1, 1] 附近，但本函数不强行裁剪。

        返回
        ----
        float
            risk scaler，>= 1.0。
        """
        if crowding is None:
            logger.warning(
                "GoldRiskScaler.compute_from_crowding: crowding 为 None，"
                "risk_scaler 退化为 1.0。"
            )
            return 1.0

        q80 = self.config.q80
        q95 = self.config.q95
        lam = self.config.lam

        # 若 q95 <= q80，则视为退化配置：仅用 crowding > q80 决定最大惩罚
        if q95 <= q80:
            if crowding <= q80:
                return 1.0
            else:
                scaler = 1.0 + max(lam, 0.0)
                return float(max(scaler, 1.0))

        # 正常情况：分段线性
        if crowding <= q80:
            return 1.0

        if crowding >= q95:
            scaler = 1.0 + max(lam, 0.0)
            return float(max(scaler, 1.0))

        # q80 < crowding < q95: 线性插值
        ratio = (crowding - q80) / (q95 - q80)
        ratio = max(min(ratio, 1.0), 0.0)  # 数值安全

        scaler = 1.0 + max(lam, 0.0) * ratio

        # 理论上 scaler >= 1.0，这里再做一次保险裁剪
        return float(max(scaler, 1.0))
