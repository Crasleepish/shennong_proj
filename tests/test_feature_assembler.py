import pytest
from app import create_app
from app.config import TestConfig
import pandas as pd
from app.features.feature_assembler import FeatureAssembler
from app.features.macro_feature_builder import MacroFeatureBuilder

@pytest.fixture
def app():
    app = create_app(config_class=TestConfig)
    yield app

def test_feature_assembler_output(app):
    macro_feature_plan = {
        "社融": [
            {"func": MacroFeatureBuilder.yoy_growth, "suffix": "yoy"},
            {"func": MacroFeatureBuilder.mom_growth, "suffix": "mom"},
            {"func": MacroFeatureBuilder.rolling_mean, "suffix": "avg3", "kwargs": {"window": 3}},
        ],
        "PMI": [
            {"func": MacroFeatureBuilder.value_minus, "suffix": "diff_50", "kwargs": {"series2": pd.Series(50, index=pd.date_range("2000-01", "2100-01", freq="M"))}},
            {"func": MacroFeatureBuilder.rolling_slope, "suffix": "trend3", "kwargs": {"window": 3}},
        ],
        "外储": [
            {"func": MacroFeatureBuilder.yoy_growth, "suffix": "yoy"},
            {"func": MacroFeatureBuilder.rolling_log_slope, "suffix": "trend3", "kwargs": {"window": 3}},
        ],
        "黄金": [
            {"func": MacroFeatureBuilder.yoy_growth, "suffix": "yoy"},
            {"func": MacroFeatureBuilder.rolling_log_slope, "suffix": "trend3", "kwargs": {"window": 3}},
        ],
        # --- 多指标函数特征 ---
        ("LPR1Y", "CPI"): [
            {"func": MacroFeatureBuilder.value_minus, "suffix": "real_rate"},  # 实际利率 = 名义利率 - 通胀
        ],
        ("M1YOY", "M2YOY"): [
            {"func": MacroFeatureBuilder.value_minus, "suffix": "diff"},  # YoY增速差
        ],
        ("BOND_10Y", "BOND_2Y"): [
            {"func": MacroFeatureBuilder.value_minus, "suffix": "diff"},  # 利差（10Y - 2Y）
        ],
        ("LPR1Y", "BOND_10Y"): [
            {"func": MacroFeatureBuilder.value_minus, "suffix": "diff"},  # 利差(LPR1Y - 10Y)
        ],
    }

    assembler = FeatureAssembler(macro_feature_plan)
    df_features = assembler.assemble_features(start="2020-01-01", end="2024-12-31")

    assert isinstance(df_features, pd.DataFrame)
    assert not df_features.empty
    assert df_features.index.name == "date"
    assert df_features.shape[1] > 0
