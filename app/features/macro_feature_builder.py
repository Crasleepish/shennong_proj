# app/features/macro_feature_builder.py

import pandas as pd
import numpy as np
from typing import Optional

class MacroFeatureBuilder:
    """
    宏观指标特征构造器：支持同比、环比、z-score、滚动均值、斜率、差值等特征构建。
    所有方法都是静态方法，便于模块化组合调用。
    """

    @staticmethod
    def raw(series: pd.Series) -> pd.Series:
        """原始值"""
        return series

    @staticmethod
    def yoy_growth(series: pd.Series) -> pd.Series:
        """同比增长率"""
        return series.pct_change(periods=12)

    @staticmethod
    def mom_growth(series: pd.Series) -> pd.Series:
        """环比增长率"""
        return series.pct_change(periods=1)

    @staticmethod
    def z_score(series: pd.Series, window: int = 12) -> pd.Series:
        """滚动 Z-Score"""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        return (series - rolling_mean) / rolling_std

    @staticmethod
    def rolling_mean(series: pd.Series, window: int = 3) -> pd.Series:
        """滚动均值"""
        return series.rolling(window).mean()

    @staticmethod
    def rolling_slope(series: pd.Series, window: int = 3) -> pd.Series:
        """滚动斜率（线性趋势）"""
        def calc_slope(x):
            if x.isna().any(): return np.nan
            t = np.arange(len(x))
            A = np.vstack([t, np.ones(len(t))]).T
            m, _ = np.linalg.lstsq(A, x.values, rcond=None)[0]
            return m
        return series.rolling(window).apply(calc_slope, raw=False)
    
    @staticmethod
    def rolling_log_slope(series: pd.Series, window: int = 3) -> pd.Series:
        """对数变换后的滚动斜率，适用于长期趋势指标（如社融）"""
        log_series = np.log(series.replace(0, np.nan)).dropna()

        def calc_log_slope(x):
            if x.isna().any(): return np.nan
            t = np.arange(len(x))
            A = np.vstack([t, np.ones(len(t))]).T
            m, _ = np.linalg.lstsq(A, x.values, rcond=None)[0]
            return m

        return log_series.rolling(window).apply(calc_log_slope, raw=False)

    @staticmethod
    def value_minus(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """两个指标直接做差（如 LPR - CPI）"""
        return series1 - series2

    @staticmethod
    def rolling_change(series: pd.Series, window: int = 3) -> pd.Series:
        """滚动变动幅度"""
        return series.diff(periods=window)

    @staticmethod
    def rename_feature(series: pd.Series, name: str) -> pd.Series:
        """统一重命名，便于入库或训练时特征筛选"""
        series.name = name
        return series

    @staticmethod
    def yoy_diff(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        """两个序列的同比增速差 = YoY(A) - YoY(B)"""
        yoy_a = series_a.pct_change(periods=12)
        yoy_b = series_b.pct_change(periods=12)
        return yoy_a - yoy_b

    @staticmethod
    def diff_yoy(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        d = series_a - series_b
        return MacroFeatureBuilder.yoy_growth(d)