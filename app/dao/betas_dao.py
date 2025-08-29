# app/dao/betas_dao.py

import pandas as pd
import numpy as np
import json
from sqlalchemy import select, func, desc, and_, literal_column
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

            subq = (
                select(
                    fb.code,
                    func.max(fb.date).label("max_date")
                )
                .where(
                    fb.date <= pd.to_datetime(as_of_date) if as_of_date is not None else True
                )
                .group_by(fb.code)
                .subquery()
            )

            stmt = (
                select(fb)
                .join(fi, fi.fund_code == fb.code)
                .join(subq, and_(
                    fb.code == subq.c.code,
                    fb.date == subq.c.max_date
                ))
            )

            if fund_type_list:
                stmt = stmt.where(fi.fund_type.in_(fund_type_list))
            if invest_type_list:
                stmt = stmt.where(fi.invest_type.in_(invest_type_list))
            if found_date_limit:
                stmt = stmt.where(fi.found_date <= pd.to_datetime(found_date_limit))

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
    def upsert_one(code: str, date: str, betas: dict, P: np.ndarray = None):
        data = {
            "code": code,
            "date": pd.to_datetime(date),
            **betas
        }
        if P is not None:
            P_binary, _ = pack_covariance(P)
            data["P_bin"] = P_binary
            
        obj = FundBeta(**data)
        with get_db() as db:
            db.merge(obj)
            db.commit()

    @staticmethod
    def upsert_batch(df: pd.DataFrame):
        if df.empty:
            logger.warning("空 DataFrame，跳过批量 upsert")
            return
        with get_db() as db:
            count = 0
            for _, row in df.iterrows():
                clean_row = row.dropna().to_dict()
                obj = FundBeta(**clean_row)
                db.merge(obj)
                count += 1
            db.commit()
            logger.info("批量 upsert FundBeta 共 %d 条", count)
