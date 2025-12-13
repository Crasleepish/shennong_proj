# app/models/regime/gold_regime_model.py

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GoldRegimeConfig:
    """
    黄金周期溢价 w_regime 的配置参数。

    整体思想：
    - trend_score: 黄金自身价格趋势（只惩罚明显走坏的情形）
    - usd_trend: 美元强势程度（只惩罚显著强美元右尾）
    - vuln_score: CFTC + 价格位置 → 脆弱度（拥挤、易踩踏）

    最终：
        w_regime = w_max * trend_score * usd_trend * (1 - vuln_beta * vuln_score)
    """

    # ---- 黄金价格相关 ----
    gold_index_code: str = "Au99.99.SGE"
    gold_lookback_days: int = 252 * 8     # 取历史窗口（最多回看 8 年）
    gold_trend_window_long: int = 120     # 趋势窗口：120 日
    gold_vol_window: int = 60             # 波动率窗口：60 日（年化）

    # TS → trend_score 单边指数衰减参数
    ts_threshold: float = 0.0             # TS >= ts_threshold 视作“中性及以上”，不惩罚
    ts_alpha: float = 0.8                 # 指数衰减速度：越大惩罚越快
    ts_floor: float = 0.02                # 最小 trend_score，避免变成 0

    # ---- 美元指数相关 ----
    usd_lookback_years: int = 8           # 统计 raw 分布用的历史年数
    usd_lookback_min_years: int = 3       # 至少有多少年数据才启用
    usd_ma_short: int = 20                # 短期均线（天）
    usd_ma_long: int = 60                 # 长期均线（天）
    usd_vol_window: int = 60              # 美元波动率窗口（天）
    usd_z_clip: float = 3.0               # raw z-score 的截断上限（绝对值）
    usd_z_neutral_hi: float = 1.5         # z <= 1.5 视作中性，不惩罚
    usd_z_beta: float = 0.8               # 右尾指数惩罚强度：越大，强美元时削得越狠

    # ---- CFTC / 脆弱度相关 ----
    vuln_beta: float = 0.5                # 脆弱度对 w_regime 的影响权重 0~1

    # ---- 周期溢价权重上下限 ----
    w_min: float = 0.0
    w_max: float = 1.0                    # 例如最多只允许 100% 周期溢价进入


class GoldRegimeModel:
    """
    基于黄金自身趋势 + 美元 regime + CFTC 脆弱度
    对黄金“周期风险溢价”的权重 w_regime 做动态调整的模型。

    get_weight(as_of) 返回 w_regime ∈ [w_min, w_max]，
    可用于：
        mu_gold = mu_structural * w_regime + mu_cycle * w_regime  等场景
    """

    def __init__(
        self,
        cfg: Optional[GoldRegimeConfig] = None,
        gold_price_fetcher=None,
        usd_index_reader=None,
        cftc_reader=None,
    ) -> None:
        self.cfg = cfg or GoldRegimeConfig()

        # 依赖默认实现：允许外部注入，便于测试 / 替换
        if gold_price_fetcher is None:
            try:
                from app.data_fetcher.csi_index_data_fetcher import CSIIndexDataFetcher

                gold_price_fetcher = CSIIndexDataFetcher()
            except Exception as e:
                logger.warning(
                    "GoldRegimeModel: 无法导入 CSIIndexDataFetcher，请在初始化时显式传入 gold_price_fetcher: %s",
                    e,
                )
        if usd_index_reader is None:
            try:
                from app.data_fetcher.us_index_data_reader import USIndexDataReader

                usd_index_reader = USIndexDataReader()
            except Exception as e:
                logger.warning(
                    "GoldRegimeModel: 无法导入 USIndexReader，请在初始化时显式传入 usd_index_reader: %s",
                    e,
                )
        if cftc_reader is None:
            try:
                from app.data_fetcher.cftc_gold_reader import CftcGoldReader

                cftc_reader = CftcGoldReader()
            except Exception as e:
                logger.warning(
                    "GoldRegimeModel: 无法导入 CftcGoldReader，请在初始化时显式传入 cftc_reader: %s",
                    e,
                )

        self.gold_price_fetcher = gold_price_fetcher
        self.usd_index_reader = usd_index_reader
        self.cftc_reader = cftc_reader

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def get_weight(self, as_of: date) -> float:
        """
        返回黄金“周期风险溢价”权重 w_regime ∈ [w_min, w_max]。

        - trend_score(as_of): 黄金自身趋势，只惩罚明显走坏的情形
        - usd_trend(as_of): 美元 regime，只惩罚显著强美元右尾
        - vuln_score(as_of): CFTC 拥挤 / 价格位置 → 脆弱度
        """
        cfg = self.cfg

        trend_score = self._compute_trend_score(as_of)
        usd_trend = self._compute_usd_trend(as_of)
        vuln_score = self._compute_vuln_score(as_of)

        # 只惩罚坏消息：trend_score, usd_trend, (1 - vuln_beta * vuln_score) 都在 [0,1]
        base = trend_score * usd_trend * (1.0 - cfg.vuln_beta * vuln_score)
        w_regime = cfg.w_max * base

        # 裁剪到 [w_min, w_max]
        w_regime = float(
            max(cfg.w_min, min(cfg.w_max, w_regime))
        )

        logger.info(
            "GoldRegimeModel.get_weight: as_of=%s, trend_score=%.4f, "
            "usd_trend=%.4f, vuln_score=%.4f, w_regime=%.4f",
            as_of,
            trend_score,
            usd_trend,
            vuln_score,
            w_regime,
        )

        return w_regime

    # ------------------------------------------------------------------
    # 1) 黄金自身趋势单边惩罚：TS → trend_score
    # ------------------------------------------------------------------
    def _compute_trend_score(self, as_of: date) -> float:
        """
        基于黄金价格计算 TS (trend signal)，然后做“只惩罚坏消息”的单边指数衰减：

            TS = 120 日收益 / 年化 60 日波动率

        映射：

            TS >= ts_threshold → trend_score = 1.0  （中性及以上，不惩罚）
            TS <  ts_threshold → trend_score = exp( alpha * (TS - ts_threshold) )
                                 再裁剪到 [ts_floor, 1.0]

        这样：
        - 大部分正常 / 偏积极情形保持周期溢价不变；
        - 只有在 TS 明显走坏时才逐步削弱周期溢价。
        """
        cfg = self.cfg
        if self.gold_price_fetcher is None:
            logger.warning("GoldRegimeModel: gold_price_fetcher 未配置，trend_score 直接返回 1.0")
            return 1.0

        eps = 1e-12

        # 往前多取一些天数，确保滚动窗口够用
        extra_days = cfg.gold_trend_window_long + cfg.gold_vol_window + 20
        start_date = as_of - timedelta(days=cfg.gold_lookback_days + extra_days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = as_of.strftime("%Y-%m-%d")

        try:
            df = self.gold_price_fetcher.get_data_by_code_and_date(
                cfg.gold_index_code,
                start=start_str,
                end=end_str,
            )
        except Exception as e:
            logger.warning(
                "GoldRegimeModel: 获取黄金价格失败，trend_score=1.0: %s", e
            )
            return 1.0

        if df is None or df.empty:
            logger.warning("GoldRegimeModel: 黄金价格数据为空，trend_score=1.0")
            return 1.0

        # 统一日期索引 & close 列
        if "date" in df.columns:
            df = df.sort_values("date")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            df = df.sort_index()

        if "close" not in df.columns:
            logger.warning("GoldRegimeModel: 黄金数据中没有 'close' 列，trend_score=1.0")
            return 1.0

        close = df["close"].astype(float).dropna()
        if close.size < cfg.gold_trend_window_long + cfg.gold_vol_window + 5:
            logger.warning(
                "GoldRegimeModel: 黄金历史数据太短（n=%d），trend_score=1.0", close.size
            )
            return 1.0

        # 计算 120 日收益（信号）
        ret_long = close.pct_change(cfg.gold_trend_window_long).iloc[-1]

        # 计算 60 日年化波动率（噪声）
        ret_daily = close.pct_change().dropna()
        sigma_60_daily = ret_daily.rolling(cfg.gold_vol_window).std().iloc[-1]
        if pd.isna(sigma_60_daily) or sigma_60_daily <= 0:
            logger.warning(
                "GoldRegimeModel: 无法计算黄金 60 日波动率，trend_score=1.0"
            )
            return 1.0

        sigma_60 = float(sigma_60_daily * math.sqrt(252.0))

        TS = float(ret_long / (sigma_60 + eps))

        # 映射 TS → trend_score
        if TS >= cfg.ts_threshold:
            trend_score = 1.0
        else:
            # 单边指数衰减：TS 越低，trend_score 越接近 0
            trend_score = math.exp(cfg.ts_alpha * (TS - cfg.ts_threshold))
            trend_score = max(cfg.ts_floor, min(1.0, trend_score))

        logger.debug(
            "GoldRegimeModel._compute_trend_score: as_of=%s, ret_long=%.4f, "
            "sigma_60=%.4f, TS=%.4f, trend_score=%.4f",
            as_of,
            ret_long,
            sigma_60,
            TS,
            trend_score,
        )

        return float(trend_score)

    # ------------------------------------------------------------------
    # 2) 美元 regime 单边惩罚：z → usd_trend
    # ------------------------------------------------------------------
    def _compute_usd_trend(self, as_of: date) -> float:
        """
        美元 regime → usd_trend（只惩罚强美元右尾）。

        步骤：
        1. 基于美元指数构造 raw 序列：
               raw_t = (MA_short - MA_long) / vol_annual
           表示“短期相对长期多涨了几个年化波动率”。

        2. 在长期窗口内计算 raw 的 z-score，并截断：
               z = (raw_t - mean(raw_hist)) / std(raw_hist)
               z_clip ∈ [-usd_z_clip, usd_z_clip]

        3. 单边指数惩罚映射：
           - 若 z <= z_neutral_hi → usd_trend = 1.0 （中性/弱美元：不惩罚）
           - 若 z >  z_neutral_hi → usd_trend = exp(-β * (z - z_neutral_hi))
                                     再裁剪到 (0, 1]
        """
        cfg = self.cfg
        if self.usd_index_reader is None:
            logger.warning("GoldRegimeModel: usd_index_reader 未配置，usd_trend=1.0")
            return 1.0

        eps = 1e-12

        # 为了计算长期 raw 分布，需要往前取 usd_lookback_years + 1 年的数据
        lookback_days = int((cfg.usd_lookback_years + 1) * 365)
        start_date = as_of - timedelta(days=lookback_days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = as_of.strftime("%Y-%m-%d")

        try:
            df_usd = self.usd_index_reader.read_us_index(
                start=start_str,
                end=end_str,
            )
        except Exception as e:
            logger.warning(
                "GoldRegimeModel: 获取美元指数失败，usd_trend=1.0: %s", e
            )
            return 1.0

        if df_usd is None or df_usd.empty:
            logger.warning("GoldRegimeModel: 美元指数数据为空，usd_trend=1.0")
            return 1.0

        # 统一日期索引 & close 列（假定 'bid_close'）
        if "trade_date" in df_usd.columns:
            df_usd = df_usd.sort_values("trade_date")
            df_usd["trade_date"] = pd.to_datetime(df_usd["trade_date"])
            df_usd = df_usd.set_index("trade_date")
        else:
            df_usd = df_usd.sort_index()

        if "bid_close" not in df_usd.columns:
            logger.warning("GoldRegimeModel: 美元指数中没有 'bid_close' 列，usd_trend=1.0")
            return 1.0

        close = df_usd["bid_close"].astype(float).dropna()
        if close.size < max(cfg.usd_ma_long, cfg.usd_vol_window) + 20:
            logger.warning(
                "GoldRegimeModel: 美元指数历史数据太短（n=%d），usd_trend=1.0", close.size
            )
            return 1.0

        # 日收益 & 年化波动率
        ret_daily = close.pct_change().dropna()
        vol_daily = ret_daily.rolling(cfg.usd_vol_window).std()
        vol_annual = vol_daily * math.sqrt(252.0)

        ma_short = close.rolling(cfg.usd_ma_short).mean()
        ma_long = close.rolling(cfg.usd_ma_long).mean()
        rel_diff = (ma_long - ma_short) / ma_long

        raw = rel_diff / (vol_annual + eps)
        raw = raw.dropna()
        raw.index = pd.to_datetime(raw.index)

        if raw.empty:
            logger.warning("GoldRegimeModel: 无法构造 raw 序列，usd_trend=1.0")
            return 1.0

        # 限制历史窗口：最多 usd_lookback_years 年，且至少 usd_lookback_min_years 年
        raw = raw[raw.index <= pd.to_datetime(as_of)]
        if raw.empty:
            logger.warning("GoldRegimeModel: raw 序列在 as_of 之前为空，usd_trend=1.0")
            return 1.0

        # 转年数检查
        span_days = (raw.index[-1] - raw.index[0]).days
        span_years = span_days / 365.0
        if span_years < cfg.usd_lookback_min_years:
            logger.warning(
                "GoldRegimeModel: raw 历史不足 %.1f 年（只有 %.2f 年），usd_trend=1.0",
                cfg.usd_lookback_min_years,
                span_years,
            )
            return 1.0

        raw_t = float(raw.iloc[-1])
        mu_raw = float(raw.mean())
        sigma_raw = float(raw.std())

        if sigma_raw <= 0 or math.isnan(sigma_raw):
            logger.warning("GoldRegimeModel: raw 标准差为 0 或 NaN，usd_trend=1.0")
            return 1.0

        z = (raw_t - mu_raw) / (sigma_raw + eps)
        # 仅用于日志：完整 z
        z_full = z

        # 截断 z（避免极端值）
        z_clip = max(-cfg.usd_z_clip, min(cfg.usd_z_clip, z))

        # 单边指数惩罚：只对右尾 z > z_neutral_hi 进行惩罚
        if z_clip <= cfg.usd_z_neutral_hi:
            usd_trend = 1.0
        else:
            delta = z_clip - cfg.usd_z_neutral_hi
            usd_trend = math.exp(-cfg.usd_z_beta * delta)
            usd_trend = max(1e-4, min(1.0, usd_trend))

        logger.debug(
            "GoldRegimeModel._compute_usd_trend: as_of=%s, raw_t=%.4f, "
            "mu_raw=%.4f, sigma_raw=%.4f, z_full=%.4f, z_clip=%.4f, usd_trend=%.4f",
            as_of,
            raw_t,
            mu_raw,
            sigma_raw,
            z_full,
            z_clip,
            usd_trend,
        )

        return float(usd_trend)

    # ------------------------------------------------------------------
    # 3) CFTC / 价格位置 → 脆弱度 vuln_score ∈ [0,1]
    # ------------------------------------------------------------------
    def _compute_vuln_score(self, as_of: date) -> float:
        """
        这里给一个相对保守、简单的缺省实现：

        - 从 CFTC 数据中取 latest <= as_of 的一条：
              noncomm_net_all / open_interest_all → crowding ∈ [0, +∞)
        - 通过分段线性 / clip 映射为 [0,1]，作为脆弱度。

        注：你可以在后续根据自己的 CFTC 指标设计，把这块逻辑换成更精细的
        （比如结合 price_position、term structure 等），接口保持 vuln_score ∈ [0,1] 不变即可。
        """
        if self.cftc_reader is None:
            logger.warning("GoldRegimeModel: cftc_reader 未配置，vuln_score=0.0")
            return 0.0

        try:
            row = self.cftc_reader.read_latest_before(as_of)
        except Exception as e:
            logger.warning(
                "GoldRegimeModel: 读取 CFTC 数据失败，vuln_score=0.0: %s", e
            )
            return 0.0

        if row is None:
            logger.warning("GoldRegimeModel: CFTC 数据为空，vuln_score=0.0")
            return 0.0

        # 这里假设 row 是一个 dict 或 ORM 对象，含有 open_interest_all, noncomm_net_all
        try:
            oi = float(row["open_interest_all"]) if isinstance(row, dict) else float(row.open_interest_all)
            net = float(row["noncomm_net_all"]) if isinstance(row, dict) else float(row.noncomm_net_all)
        except Exception:
            logger.warning("GoldRegimeModel: CFTC 行结构不符合预期，vuln_score=0.0")
            return 0.0

        if oi <= 0:
            return 0.0

        crowding = net / oi  # 大致 0~0.4 之间

        # 分段映射为 [0,1]，你可以之后微调参数：
        #   crowding <= 0.15 → vuln≈0
        #   0.15~0.25        → vuln 线性上升
        #   crowding >= 0.25 → vuln≈1
        if crowding <= 0.15:
            vuln = 0.0
        elif crowding >= 0.25:
            vuln = 1.0
        else:
            vuln = (crowding - 0.15) / (0.25 - 0.15)

        vuln = float(max(0.0, min(1.0, vuln)))

        logger.debug(
            "GoldRegimeModel._compute_vuln_score: as_of=%s, crowding=%.4f, vuln=%.4f",
            as_of,
            crowding,
            vuln,
        )

        return vuln
