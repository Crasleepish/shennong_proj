# tests/test_fund_beta_dao.py

import pytest
import pandas as pd
from datetime import datetime
from app.dao.betas_dao import FundBetaDao
from app.database import get_db
from app.models.fund_models import FundBeta, FundInfo
from app.models.etf_model import EtfInfo
from app import create_app
from app.config import TestConfig, Config
from app.utils.cov_packer import pack_covariance, unpack_covariance
import json
import numpy as np
from sqlalchemy import select


@pytest.fixture
def app():
    app = create_app(config_class=Config)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app


@pytest.fixture(scope="function")
def setup_test_data(app):
    """
    初始化测试数据：插入 ETF / 基金信息及 FundBeta 数据
    """
    with get_db() as db:
        db.query(FundBeta).filter(FundBeta.code.in_(["TEST_ETF", "TEST_FUND"])).delete()

        db.merge(EtfInfo(
            etf_code="TEST_ETF",
            etf_name="测试ETF",
            fund_type="指数型",
            invest_type="股票型",
            found_date=datetime(2020, 1, 1)
        ))

        db.merge(FundInfo(
            fund_code="TEST_FUND",
            fund_name="测试基金",
            fund_type="混合型",
            invest_type="债券型",
            found_date=datetime(2019, 6, 1),
            fee_rate=0.01,
            commission_rate=0.005,
            market="E"
        ))

        P_json_default = json.dumps([[1 if i == j else 0 for j in range(5)] for i in range(5)])

        db.merge(FundBeta(
            code="TEST_ETF",
            date=datetime(2023, 12, 31),
            MKT=1.1, SMB=0.2, HML=-0.1, QMJ=0.05, const=0.0002,
            P_json=P_json_default
        ))

        db.merge(FundBeta(
            code="TEST_ETF",
            date=datetime(2024, 12, 31),
            MKT=1.2, SMB=0.3, HML=-0.2, QMJ=0.06, const=0.0003,
            P_json=P_json_default
        ))

        db.merge(FundBeta(
            code="TEST_FUND",
            date=datetime(2024, 12, 30),
            MKT=1.3, SMB=0.1, HML=0.0, QMJ=0.04, const=0.0004,
            P_json=P_json_default
        ))

        db.commit()


def test_select_by_code_date(app, setup_test_data):
    df = FundBetaDao.select_by_code_date("TEST_ETF")
    assert not df.empty
    assert "MKT" in df.columns
    assert df["code"].nunique() == 1


def test_select_latest_by_code(app, setup_test_data):
    df = FundBetaDao.select_latest_by_code("TEST_ETF")
    assert len(df) == 1
    assert df.iloc[0]["date"] == pd.to_datetime("2024-12-31").date()


def test_get_latest_etf_betas(app, setup_test_data):
    df = FundBetaDao.get_latest_etf_betas(
        fund_type_list=["指数型"],
        invest_type_list=["股票型"],
        found_date_limit="2024-01-01"
    )
    assert not df.empty
    assert "code" in df.columns
    assert df["code"].isin(["TEST_ETF"]).all()


def test_get_latest_fund_betas(app, setup_test_data):
    df = FundBetaDao.get_latest_fund_betas(
        fund_type_list=["混合型"],
        invest_type_list=["债券型"],
        found_date_limit="2025-01-01"
    )
    assert not df.empty
    assert "code" in df.columns
    assert df["code"].isin(["TEST_FUND"]).all()


def test_upsert_one(app, setup_test_data):
    P = np.array([[1, 0.5, 0.2, 0.1, 0.05], [0.5, 1, 0.3, 0.2, 0.1], [0.2, 0.3, 1, 0.4, 0.2], [0.1, 0.2, 0.4, 1, 0.3], [0.05, 0.1, 0.2, 0.3, 1]])
    FundBetaDao.upsert_one("TEST_ETF", "2025-07-23", {
        "MKT": 0.88, "SMB": 0.22, "HML": -0.05, "QMJ": 0.07, "const": 0.0009
    }, P=P)
    df = FundBetaDao.select_by_code_date("TEST_ETF", "2025-07-23")
    assert not df.empty
    assert abs(df.iloc[0]["MKT"] - 0.88) < 1e-6


BATCH_SIZE = 1000

def test_migrate_pjson_to_pbin(app):
    """
    遍历 fund_beta 表中所有记录，将 P_json 字段解码为矩阵并压缩为 P_bin 存储
    批量处理 FundBeta 表，将 P_json 字段压缩为 P_bin
    """
    total_updated = 0
    offset = 0

    while True:
        with get_db() as db:
            # 分批查询，避免内存暴涨
            stmt = (
                select(FundBeta)
                .where(FundBeta.P_json != None)
                .offset(offset)
                .limit(BATCH_SIZE)
            )
            rows = db.execute(stmt).scalars().all()

            if not rows:
                break  # 结束

            batch_updated = 0
            for row in rows:
                try:
                    matrix_data = json.loads(row.P_json)
                    matrix = np.array(matrix_data, dtype=np.float32)
                    row.P_bin = pack_covariance(matrix)[0]
                    batch_updated += 1
                except Exception as e:
                    print(f"⚠️ 错误 @ {row.code} {row.date}: {e}")

            db.commit()
            total_updated += batch_updated
            print(f"✅ 处理批次 offset={offset} 更新={batch_updated}")
            offset += BATCH_SIZE

    print(f"✅ 全部完成：总更新 {total_updated} 条")