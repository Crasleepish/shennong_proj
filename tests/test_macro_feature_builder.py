# tests/test_macro_feature_builder.py

import pandas as pd
import numpy as np
from app.features.macro_feature_builder import MacroFeatureBuilder


def test_yoy_growth():
    series = pd.Series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220])
    result = MacroFeatureBuilder.yoy_growth(series)
    assert np.isclose(result.iloc[-1], 1.2)  # (220 - 100...) / 100...

def test_mom_growth():
    series = pd.Series([100, 105, 110])
    result = MacroFeatureBuilder.mom_growth(series)
    assert np.isclose(result.iloc[-1], (110 - 105) / 105)

def test_z_score():
    series = pd.Series(range(1, 21))
    z = MacroFeatureBuilder.z_score(series, window=5)
    assert not z.isna().all()
    assert np.isclose(z.iloc[-1], 1.2649, atol=1e-3)

def test_rolling_mean():
    series = pd.Series([1, 2, 3, 4, 5])
    avg = MacroFeatureBuilder.rolling_mean(series, window=3)
    assert avg.iloc[-1] == 4.0

def test_rolling_slope_linear():
    series = pd.Series([1, 2, 3, 4, 5, 6])
    slope = MacroFeatureBuilder.rolling_slope(series, window=3)
    assert np.isclose(slope.iloc[-1], 1.0)

def test_value_minus():
    s1 = pd.Series([5, 6, 7])
    s2 = pd.Series([1, 2, 3])
    result = MacroFeatureBuilder.value_minus(s1, s2)
    pd.testing.assert_series_equal(result, pd.Series([4, 4, 4]))

def test_rolling_change():
    series = pd.Series([10, 11, 12, 13])
    change = MacroFeatureBuilder.rolling_change(series, window=2)
    assert change.iloc[-1] == 2
