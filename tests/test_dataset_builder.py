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
    # 构建一个小时间段的数据（确保数据库中有对应数据）
    X, Y = builder.build(start="2010-01-01", end="2016-01-01")

    # X 和 Y 不应为空
    assert not X.empty, "特征集 X 不应为空"
    assert not Y.empty, "标签集 Y 不应为空"

    # 样本数量应一致
    assert X.shape[0] == Y.shape[0], "X 和 Y 的行数应相同"

    # 标签应包含预期列（部分即可）
    expected_cols = ["MKT_20d_ret", "SMB_60d_ret"]
    for col in expected_cols:
        assert col in Y.columns, f"标签列缺失: {col}"

def test_vif_pca_features(builder):
    X, _ = builder.build(start="2015-01-01", end="2016-01-01")

    # VIF + PCA 后维度应合理（不超过原始维度）
    assert X.shape[1] < 100, "降维后的特征数量不应过大"

    # 检查是否为 DataFrame 且无缺失值
    assert isinstance(X, pd.DataFrame), "输出应为 DataFrame"
    assert not X.isnull().any().any(), "特征中不应包含缺失值"
