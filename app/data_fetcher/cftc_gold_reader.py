# app/data_fetcher/cftc_gold_reader.py

import logging
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.gold_models import GoldCFTCReport

logger = logging.getLogger(__name__)


class CftcGoldReader:
    """
    从数据库中读取黄金 CFTC 周度持仓数据，并提供拥挤度（crowding）时间序列。

    约定：
    - 使用 GoldCFTCReport.report_date 作为时间索引（每周一次）。
    - 拥挤度定义：
        crowding = noncomm_net_all / open_interest_all
      其中：
        - open_interest_all: 总未平仓量（Futures + Options Combined）
        - noncomm_net_all : 非商业净多头（管理基金 + 其他可报告 Long - Short）
    """

    # -------- 基础读取接口 --------
    @staticmethod
    def read_raw(
        start: Optional[str] = None,
        end: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> pd.DataFrame:
        """
        从 gold_cftc_report 表中读取原始 CFTC 数据。

        参数
        ----
        start : str, 可选
            起始日期（含），格式如 "2024-01-01" 或 "2024/01/01"。
            对应 GoldCFTCReport.report_date 的过滤下界。
        end : str, 可选
            结束日期（含），同上。
        db : Session, 可选
            传入已有的 SQLAlchemy Session；若为 None 则通过 get_db() 自动管理。

        返回
        ----
        pd.DataFrame
            index: DatetimeIndex（report_date）
            columns: 包含至少
                - open_interest_all
                - noncomm_net_all
            若无数据则返回空 DataFrame。
        """
        need_close = False
        if db is None:
            db_ctx = get_db()
            db = db_ctx.__enter__()
            need_close = True

        try:
            query = db.query(GoldCFTCReport)

            if start:
                start_ts = pd.to_datetime(start)
                query = query.filter(GoldCFTCReport.report_date >= start_ts)
            if end:
                end_ts = pd.to_datetime(end)
                query = query.filter(GoldCFTCReport.report_date <= end_ts)

            df = pd.read_sql(query.statement, db.bind)

            if df.empty:
                logger.warning(
                    "CftcGoldReader.read_raw: 在区间 [%s, %s] 内未读取到任何 CFTC 黄金数据。",
                    start,
                    end,
                )
                return pd.DataFrame()

            # 统一处理日期与排序
            if "report_date" not in df.columns:
                logger.error(
                    "CftcGoldReader.read_raw: 结果中不包含 report_date 字段，"
                    "请检查 GoldCFTCReport 模型定义。"
                )
                return pd.DataFrame()

            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values("report_date")

            # 只保留本模块需要用到的字段，避免暴露过多内部细节
            keep_cols = []
            for col in ["report_date", "open_interest_all", "noncomm_net_all"]:
                if col not in df.columns:
                    logger.error(
                        "CftcGoldReader.read_raw: 缺少必需字段 '%s'，请检查 GoldCFTCReport 模型和表结构。",
                        col,
                    )
                    return pd.DataFrame()
                keep_cols.append(col)

            df = df[keep_cols]

            # 去除日期为空的行
            df = df.dropna(subset=["report_date"])
            df = df.set_index("report_date")
            df.index.name = "date"

            return df

        finally:
            if need_close:
                db_ctx.__exit__(None, None, None)

    # -------- 拥挤度计算接口 --------
    @staticmethod
    def read_crowding(
        start: Optional[str] = None,
        end: Optional[str] = None,
        min_open_interest: int = 0,
        db: Optional[Session] = None,
    ) -> pd.DataFrame:
        """
        读取黄金 CFTC 数据并计算拥挤度 crowding 序列。

        拥挤度定义：
            crowding = noncomm_net_all / open_interest_all

        参数
        ----
        start : str, 可选
            起始日期（含），过滤 report_date >= start。
        end : str, 可选
            结束日期（含），过滤 report_date <= end。
        min_open_interest : int, 可选
            若 open_interest_all 小于该阈值，则认为数据不可靠并丢弃。
            默认 0 表示不过滤。
        db : Session, 可选
            传入已有的 SQLAlchemy Session；若为 None 则内部自动创建。

        返回
        ----
        pd.DataFrame
            index: DatetimeIndex（周度 report_date）
            columns:
                - crowding: 拥挤度（非商业净多头 / 总未平仓量），范围大致在 [-1, 1] 附近
                - open_interest_all
                - noncomm_net_all
            若无有效数据则返回空 DataFrame。
        """
        df = CftcGoldReader.read_raw(start=start, end=end, db=db)
        if df.empty:
            return df

        # 过滤 open_interest 太小或为 0 的情况，避免除零或噪声
        if min_open_interest > 0:
            before = len(df)
            df = df[df["open_interest_all"] >= min_open_interest]
            after = len(df)
            if after < before:
                logger.info(
                    "CftcGoldReader.read_crowding: 由于 open_interest_all < %d 剔除 %d 条记录。",
                    min_open_interest,
                    before - after,
                )

        # 避免 /0 和 NaN
        oi = df["open_interest_all"].replace(0, pd.NA)
        crowding = df["noncomm_net_all"] / oi

        df_out = df.copy()
        df_out["crowding"] = crowding

        # 丢弃 crowding 计算不出的行
        df_out = df_out.dropna(subset=["crowding"])

        # 确保输出顺序
        df_out = df_out[["crowding", "open_interest_all", "noncomm_net_all"]]

        logger.info(
            "CftcGoldReader.read_crowding: 生成黄金 CFTC 拥挤度序列，共 %d 期。",
            len(df_out),
        )

        return df_out

    # -------- 获取某日对应的最近一期 CFTC 拥挤度 --------
    @staticmethod
    def get_latest_crowding_before(
        as_of: str,
        db: Optional[Session] = None,
    ) -> Optional[float]:
        """
        给定某个日期 as_of，返回该日期之前（含当日）最近一次
        CFTC 报告的拥挤度 crowding（标量）。

        该接口方便在日度/调仓日逻辑中直接查询“当前最新 CFTC 拥挤度”。

        参数
        ----
        as_of : str
            查询日期，如 '2025-12-31'。
        db : Session, 可选
            传入已有的 SQLAlchemy Session；若为 None 则内部自动创建。

        返回
        ----
        float 或 None
            - crowding 值（非商业净多头 / 总未平仓量）
            - 若在 as_of 之前没有任何 CFTC 数据，则返回 None。
        """
        as_of_ts = pd.to_datetime(as_of)

        need_close = False
        if db is None:
            db_ctx = get_db()
            db = db_ctx.__enter__()
            need_close = True

        try:
            query = (
                db.query(GoldCFTCReport)
                .filter(GoldCFTCReport.report_date <= as_of_ts)
                .order_by(GoldCFTCReport.report_date.desc())
            )

            df = pd.read_sql(query.statement.limit(1), db.bind)
            if df.empty:
                logger.warning(
                    "CftcGoldReader.get_latest_crowding_before: 在 %s 之前无任何 CFTC 记录。",
                    as_of,
                )
                return None

            row = df.iloc[0]
            oi = row.get("open_interest_all")
            net = row.get("noncomm_net_all")

            if oi is None or oi == 0 or net is None:
                logger.warning(
                    "CftcGoldReader.get_latest_crowding_before: 最近一条记录 open_interest_all 或 noncomm_net_all 无效。"
                )
                return None

            return float(net) / float(oi)

        finally:
            if need_close:
                db_ctx.__exit__(None, None, None)
