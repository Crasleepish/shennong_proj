# tests/test_macro_feature_pipeline.py

import pandas as pd
from app.features import MacroFeatureBuilder, MacroFeaturePipeline

def test_pipeline_transform():
    # 构造一个测试 DataFrame
    dates = pd.date_range("2020-01", periods=15, freq="M")
    df = pd.DataFrame({
        "社融": range(100, 115),
        "PMI": range(50, 65),
        "M1": range(10, 25),
        "M2": range(15, 30),
    }, index=dates)

    plan = {
        "社融": [
            {"func": MacroFeatureBuilder.yoy_growth, "suffix": "yoy"},
            {"func": MacroFeatureBuilder.rolling_slope, "suffix": "slope3", "kwargs": {"window": 3}},
        ],
        "PMI": [
            {"func": MacroFeatureBuilder.rolling_mean, "suffix": "avg3", "kwargs": {"window": 3}},
        ],
        "M1": [
            {"func": MacroFeatureBuilder.value_minus, "suffix": "minus_M2", "kwargs": {"series2": df["M2"]}}
        ]
    }

    pipeline = MacroFeaturePipeline(feature_plan=plan)
    features_df = pipeline.transform(df)

    assert isinstance(features_df, pd.DataFrame)
    assert "社融_yoy" in features_df.columns
    assert "PMI_avg3" in features_df.columns
    assert "M1_minus_M2" in features_df.columns
    assert not features_df.isna().all().any()  # 至少有部分数据不是 NaN
