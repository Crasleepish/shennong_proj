# app/data_process/fundamental_filler.py

import pandas as pd
from typing import List
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.stock_models import FundamentalData
from sqlalchemy import and_, or_, asc
import logging

logger = logging.getLogger(__name__)


def safe_value(val):
    return None if pd.isna(val) else val

def safe_num(val):
    return 0 if pd.isna(val) else val

def safe_ffill_float(series: pd.Series) -> pd.Series:
    return pd.Series(series.values, dtype="float64").ffill()


def calc_operating_profit_ttm(df: pd.DataFrame, overwrite: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values("report_date")
    if "operating_profit_ttm" not in df.columns:
        df["operating_profit_ttm"] = None

    for i, row in df.iterrows():
        if not overwrite and pd.notna(row.get("operating_profit_ttm")):
            continue

        cur_date = row["report_date"]
        cur_op = row["operating_profit"]
        if pd.isna(cur_op):
            continue

        prev_annual = df[df["report_date"] == pd.Timestamp(cur_date.year - 1, 12, 31)]
        prev_same_period = df[
            (df["report_date"].dt.year == cur_date.year - 1) &
            (df["report_date"].dt.month == cur_date.month)
        ].sort_values("report_date", ascending=False)

        if prev_annual.empty or prev_same_period.empty:
            continue

        last_annual = prev_annual.iloc[0]["operating_profit"]
        last_period = prev_same_period.iloc[0]["operating_profit"]
        if pd.isna(last_annual) or pd.isna(last_period):
            continue

        val = cur_op + last_annual - last_period
        df.at[i, "operating_profit_ttm"] = val

    df["operating_profit_ttm"] = safe_ffill_float(df["operating_profit_ttm"])
    return df


def calc_total_liabilities(df: pd.DataFrame, overwrite: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values("report_date")
    if "total_liabilities" not in df.columns:
        df["total_liabilities"] = None

    for i, row in df.iterrows():
        if not overwrite and pd.notna(row.get("total_liabilities")):
            continue

        cl = row["current_liabilities"]
        ncl = row["noncurrent_liabilities"]
        tcl = row["total_liabilities"]
        if pd.isna(cl) and pd.isna(ncl) and pd.isna(tcl):
            continue

        df.at[i, "total_liabilities"] = max(safe_num(cl) + safe_num(ncl), safe_num(tcl))

    df["total_liabilities"] = safe_ffill_float(df["total_liabilities"])
    return df


def fill_fundamental_fields(codes: List[str] = None, asof_date: str = None, overwrite: bool = False):
    try:
        with get_db() as db:
            if codes is None:
                codes = db.query(FundamentalData.stock_code).distinct().all()
                codes = [c[0] for c in codes]

            for code in codes:
                sql = (
                    db.query(FundamentalData)
                    .filter(FundamentalData.stock_code == code)
                )

                if asof_date:
                    asof_date = pd.to_datetime(asof_date, format="%Y%m%d")
                    sql = sql.filter(FundamentalData.report_date <= asof_date)

                sql = sql.order_by(asc(FundamentalData.report_date))
                df = pd.read_sql(sql.statement, db.bind)
                df = df.sort_values("report_date")

                modified_flag = False

                if overwrite or pd.isna(df["operating_profit_ttm"].iloc[-1]):
                    df = calc_operating_profit_ttm(df, overwrite)
                    modified_flag = True

                if overwrite or pd.isna(df["total_liabilities"].iloc[-1]):
                    df = calc_total_liabilities(df, overwrite)
                    modified_flag = True

                if modified_flag:
                    for _, row in df.iterrows():
                        db.merge(FundamentalData(
                            stock_code=row["stock_code"],
                            report_date=row["report_date"],
                            operating_profit_ttm=safe_value(row.get("operating_profit_ttm")),
                            total_liabilities=safe_value(row.get("total_liabilities"))
                        ))

            db.commit()
            logger.info("✅ 所有报告期字段填充完成")

    except Exception as e:
        logger.error(f"❌ 批量填充失败: {e}")
        db.rollback()


def fill_specific_report_date(report_date: str, overwrite: bool = False):
    try:
        report_date = pd.to_datetime(report_date)

        with get_db() as db:
            rows = (
                db.query(FundamentalData)
                .filter(FundamentalData.report_date <= report_date)
                .order_by(FundamentalData.stock_code, FundamentalData.report_date)
                .all()
            )
            if not rows:
                logger.warning("⚠️ 未找到任何匹配的记录")
                return

            df = pd.DataFrame([r.__dict__ for r in rows])
            df = df.drop(columns=["_sa_instance_state"])

            for code in df["stock_code"].unique():
                df_code = df[df["stock_code"] == code].copy()
                df_code = df_code.sort_values("report_date")

                if overwrite or df_code["operating_profit_ttm"].isna().any():
                    df_code = calc_operating_profit_ttm(df_code, overwrite)

                if overwrite or df_code["total_liabilities"].isna().any():
                    df_code = calc_total_liabilities(df_code, overwrite)

                # 提取目标日期那一行
                row = df_code[df_code["report_date"] == report_date]
                if row.empty:
                    continue
                row = row.iloc[0]

                db.merge(FundamentalData(
                    stock_code=row["stock_code"],
                    report_date=row["report_date"],
                    operating_profit_ttm=safe_value(row.get("operating_profit_ttm")),
                    total_liabilities=safe_value(row.get("total_liabilities"))
                ))

            db.commit()
            logger.info(f"✅ 已完成 {report_date.date()} 当期字段填充")

    except Exception as e:
        logger.error(f"❌ 指定报告期填充失败: {e}")
        db.rollback()