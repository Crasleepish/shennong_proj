# tests/test_dataset_builder.py

import pytest
from app.ml.dataset_builder import DatasetBuilder
from app import create_app
from app.config import TestConfig
import pandas as pd

@pytest.fixture
def app():
    app = create_app(config_class=TestConfig)
    yield app

@pytest.fixture
def builder(app):
    return DatasetBuilder()

def test_build_output_shape(builder):
    # 构建一个时间段的数据（你需要保证数据库里这个时间段有数据）
    start_date = "2018-01-01"
    end_date = "2020-12-31"

    tasks = [
        builder.build_mkt_volatility,
        builder.build_mkt_tri_class,
        builder.build_smb_hml_tri,
        builder.build_smb_qmj_tri,
        builder.build_hml_qmj_tri,
    ]

    for task in tasks:
        X, Y = task(start=start_date, end=end_date)

        # 1. 非空
        assert not X.empty, f"{task.__name__}: X is empty"
        assert not Y.empty, f"{task.__name__}: Y is empty"

        # 2. 对齐
        assert all(X.index == Y.index), f"{task.__name__}: Index mismatch"

        # 3. 没有NaN
        assert not X.isnull().values.any(), f"{task.__name__}: X has NaN"
        assert not Y.isnull().values.any(), f"{task.__name__}: Y has NaN"

        # 4. Y 是单列
        assert Y.shape[1] == 1, f"{task.__name__}: Y should be single-column"

        # 5. X 有足够特征列
        assert X.shape[1] >= 5, f"{task.__name__}: Too few features"

        print(f"{task.__name__} passed with shape X: {X.shape}, Y: {Y.shape}")
