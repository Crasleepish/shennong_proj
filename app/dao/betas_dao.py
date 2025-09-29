# app/dao/betas_dao.py

import pandas as pd
import numpy as np
import json
from sqlalchemy import select, func, desc, and_, literal_column, literal
from sqlalchemy.orm import aliased
from app.database import get_db
from app.models.fund_models import FundBeta
from app.models.fund_models import FundInfo
from app.models.etf_model import EtfInfo
from app.utils.cov_packer import pack_covariance, unpack_covariance
import logging

logger = logging.getLogger(__name__)


class FundBetaDao:

    @staticmethod
    def select_by_code_date(code: str, date: str = None) -> pd.DataFrame:
        with get_db() as db:
            query = db.query(FundBeta).filter(FundBeta.code == code)
            if date:
                query = query.filter(FundBeta.date == pd.to_datetime(date))
            df = pd.read_sql(query.statement, db.bind)
        return df

    @staticmethod
    def select_latest_by_code(code: str) -> pd.DataFrame:
        with get_db() as db:
            subq = (
                select(
                    FundBeta.code,
                    func.max(FundBeta.date).label("max_date")
                )
                .where(FundBeta.code == code)
                .group_by(FundBeta.code)
                .subquery()
            )

            stmt = (
                select(FundBeta)
                .join(subq, and_(
                    FundBeta.code == subq.c.code,
                    FundBeta.date == subq.c.max_date
                ))
            )
            df = pd.read_sql(stmt, db.bind)
        return df
 
    @staticmethod
    def select_all_by_code_date(code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        with get_db() as db:
            query = db.query(FundBeta).filter(FundBeta.code == code)
            if start_date:
                query = query.filter(FundBeta.date >= pd.to_datetime(start_date))
            if end_date:
                query = query.filter(FundBeta.date <= pd.to_datetime(end_date))
            df = pd.read_sql(query.statement, db.bind)
        return df

    @staticmethod
    def get_latest_fund_betas(
        fund_type_list=None,
        invest_type_list=None,
        found_date_limit: str = None,
        as_of_date: str = None,
    ) -> pd.DataFrame:
        with get_db() as db:
            fi = aliased(FundInfo)
            fb = aliased(FundBeta)

            # 子查询：每个基金最新的一条 beta 记录
            latest_beta_subq = (
                select(fb)
                .where(
                    fb.code == fi.fund_code,
                    fb.date <= pd.to_datetime(as_of_date) if as_of_date else literal(True)
                )
                .order_by(desc(fb.date))
                .limit(1)
                .lateral()
            )

            # 主查询
            stmt = (
                select(
                    fi.fund_code,
                    fi.fund_name,
                    fi.fund_type,
                    fi.invest_type,
                    fi.found_date,
                    literal_column("fb.code").label("code"),
                    literal_column("fb.\"date\"").label("date"),
                    literal_column("fb.\"MKT\"").label("MKT"),
                    literal_column("fb.\"SMB\"").label("SMB"),
                    literal_column("fb.\"HML\"").label("HML"),
                    literal_column("fb.\"QMJ\"").label("QMJ"),
                    literal_column("fb.const").label("const"),
                )
                .select_from(fi)
                .join(latest_beta_subq.alias("fb"), literal_column("true"), isouter=False)
            )

            # 过滤条件
            if fund_type_list:
                stmt = stmt.where(fi.fund_type.in_(fund_type_list))
            if invest_type_list:
                stmt = stmt.where(fi.invest_type.in_(invest_type_list))
            if found_date_limit:
                stmt = stmt.where(fi.found_date <= pd.to_datetime(found_date_limit))

            # 查询为 DataFrame
            df = pd.read_sql(stmt, db.bind)
        return df
    
    @staticmethod
    def get_all_betas_by_type_cond(
        fund_type_list=None,
        invest_type_list=None,
        found_date_limit: str = None,
        as_of_date: str = None,
        start_date: str = None,
    ) -> pd.DataFrame:
        with get_db() as db:
            fi = aliased(FundInfo)
            fb = aliased(FundBeta)

            # 基础查询：按基金基本信息筛选，再取其所有（至 as_of_date 之前）的 beta 记录
            stmt = (
                select(
                    fi.fund_code,
                    fb.code.label("code"),
                    fb.date.label("date"),
                    fb.MKT.label("MKT"),
                    fb.SMB.label("SMB"),
                    fb.HML.label("HML"),
                    fb.QMJ.label("QMJ"),
                    fb.const.label("const"),
                    fb.P_bin.label("P_bin")
                )
                .select_from(fi)
                .join(fb, fb.code == fi.fund_code, isouter=False)
            )

            # 基于基金属性的过滤
            if fund_type_list:
                stmt = stmt.where(fi.fund_type.in_(fund_type_list))
            if invest_type_list:
                stmt = stmt.where(fi.invest_type.in_(invest_type_list))
            if found_date_limit:
                stmt = stmt.where(fi.found_date <= pd.to_datetime(found_date_limit))

            # 截止日期过滤（仅取 as_of_date 及之前的 beta 记录）
            if as_of_date:
                stmt = stmt.where(fb.date <= pd.to_datetime(as_of_date))
            
            if start_date:
                stmt = stmt.where(fb.date >= pd.to_datetime(start_date))

            # 排序：先按代码，再按日期
            stmt = stmt.order_by(fb.code.asc(), fb.date.asc())

            df = pd.read_sql(stmt, db.bind)
        return df

    @staticmethod
    def get_latest_etf_betas(
        fund_type_list=None,
        invest_type_list=None,
        found_date_limit: str = None
    ) -> pd.DataFrame:
        with get_db() as db:
            ei = aliased(EtfInfo)
            fb = aliased(FundBeta)

            subq = (
                select(
                    fb.code,
                    func.max(fb.date).label("max_date")
                )
                .group_by(fb.code)
                .subquery()
            )

            stmt = (
                select(fb)
                .join(ei, ei.etf_code == fb.code)
                .join(subq, and_(
                    fb.code == subq.c.code,
                    fb.date == subq.c.max_date
                ))
            )

            if fund_type_list:
                stmt = stmt.where(ei.fund_type.in_(fund_type_list))
            if invest_type_list:
                stmt = stmt.where(ei.invest_type.in_(invest_type_list))
            if found_date_limit:
                stmt = stmt.where(ei.found_date <= pd.to_datetime(found_date_limit))

            df = pd.read_sql(stmt, db.bind)
        return df

    @staticmethod
    def upsert_one(code: str, date: str, betas: dict, P: np.ndarray = None, log_nav_true: float = None, log_nav_fit: float = None, output_p_json: bool = False):
        data = {
            "code": code,
            "date": pd.to_datetime(date),
            **betas
        }
        if P is not None:
            P_binary, _ = pack_covariance(P)
            data["P_bin"] = P_binary
            if output_p_json:
                data["P_json"] = json.dumps(P.tolist())
        
        if log_nav_true is not None:
            data["log_nav_true"] = log_nav_true
        
        if log_nav_fit is not None:
            data["log_nav_fit"] = log_nav_fit
            
        obj = FundBeta(**data)
        with get_db() as db:
            db.merge(obj)
            db.commit()


    @staticmethod
    def select_const_by_code(codes: list, date: str = None) -> pd.DataFrame:
        """
        输入: codes - List[str]，资产代码列表
        输出: DataFrame，index=日期(date)，columns=code，values=FundBeta.const
        """
        if not codes:
            return pd.DataFrame()

        with get_db() as db:
            query = (
                db.query(
                    FundBeta.date.label("date"),
                    FundBeta.code.label("code"),
                    FundBeta.const.label("const"),
                )
                .filter(FundBeta.code.in_(codes), FundBeta.date <= pd.to_datetime(date))
            )
            df = pd.read_sql(query.statement, db.bind)

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        # 同一(date, code)若有多条记录，取最后一条（按读取顺序的“last”）
        df_pivot = (
            df.pivot_table(index="date", columns="code", values="const", aggfunc="last")
              .sort_index()
        )
        return df_pivot