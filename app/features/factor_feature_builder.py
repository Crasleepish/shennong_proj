import pandas as pd
import numpy as np
from typing import Optional


class FactorFeatureBuilder:
    """
    构建因子特征（基于因子每日收益率）
    输入应为日度收益率时间序列（index=date, columns=factor_name）
    输出为特征 DataFrame（多列）
    """

    @staticmethod
    def calc_cum_return(df: pd.DataFrame) -> pd.DataFrame:
        return (1 + df).cumprod()

    @staticmethod
    def rolling_zscore(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        """
        计算滚动 Z-Score。
        若未指定 min_periods，则设为 max(5, 1)：至少需要 5 个数据点才计算 Z-score。
        """
        if min_periods is None:
            min_periods = max(5, 1)
        else:
            min_periods = max(5, min_periods)

        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()
        return (series - rolling_mean) / rolling_std

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain, index=series.index).rolling(window).mean()
        roll_down = pd.Series(loss, index=series.index).rolling(window).mean()
        rs = roll_up / roll_down
        return 100 - (100 / (1 + rs))

    @staticmethod
    def volatility(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).std()

    @staticmethod
    def mean_change(series: pd.Series, period: int) -> pd.Series:
        """
        计算短期均线与长期均线的变化率：
        均线值为 [1.0, 1.1, 1.21]，变化率为 [NaN, 0.1, 0.1]
        """
        mean_line = series.rolling(period).mean()
        return mean_line.pct_change()

    @staticmethod
    def regression_slope(series: pd.Series, window: int) -> pd.Series:
        def _slope(x):
            y = x.values
            x_vals = np.arange(len(y))
            if np.isnan(y).any():
                return np.nan
            slope = np.polyfit(x_vals, y, 1)[0]
            return slope

        return series.rolling(window).apply(_slope, raw=False)

    @staticmethod
    def regression_slope_diff(series: pd.Series, window: int) -> pd.Series:
        slope = FactorFeatureBuilder.regression_slope(series, window)
        return slope.diff()

    @staticmethod
    def up_mean_vs_down_max(series: pd.Series, up_days: int = 5, down_days: int = 20) -> pd.Series:
        up_avg = series.rolling(up_days).mean()
        down_max = series.rolling(down_days).min()
        return up_avg / down_max

    @staticmethod
    def regression_slope_on_zscore(series: pd.Series, zscore_window: int, reg_window: int) -> pd.Series:
        z = FactorFeatureBuilder.rolling_zscore(series, zscore_window)
        return FactorFeatureBuilder.regression_slope(z, reg_window)

    @staticmethod
    def regression_slope_diff_on_zscore(series: pd.Series, zscore_window: int, reg_window: int) -> pd.Series:
        slope = FactorFeatureBuilder.regression_slope_on_zscore(series, zscore_window, reg_window)
        return slope.diff()

    @staticmethod
    def mean_diff_change_rate(series: pd.Series, short: int, long: int) -> pd.Series:
        """
        计算短期均线 - 长期均线的差值曲线，然后对该差值曲线做变化率
        """
        diff_curve = series.rolling(short).mean() - series.rolling(long).mean()
        return diff_curve.pct_change()
    
    @staticmethod
    def value_minus(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """两个指标直接做差（如 LPR - CPI）"""
        return series1 - series2
    
    @staticmethod
    def regression_slope_on_zscore_of_diff(series1: pd.Series, series2: pd.Series, zscore_window: int, reg_window: int) -> pd.Series:
        series = series1 - series2
        z = FactorFeatureBuilder.rolling_zscore(series, zscore_window)
        return FactorFeatureBuilder.regression_slope(z, reg_window)

    @staticmethod
    def days_since_rsi_extreme(series: pd.Series, window: int = 14, threshold: float = 70, mode: str = "gt") -> pd.Series:
        """
        计算距离上次 RSI 极端值的交易日数（返回 log(1 + days)）
        :param series: 输入累积收益（净值）序列
        :param window: RSI 计算窗口
        :param threshold: 极端值判断阈值
        :param mode: "gt" 表示 RSI > 阈值，"lt" 表示 RSI < 阈值
        :return: log(1 + days_since_extreme)
        """
        rsi_series = FactorFeatureBuilder.rsi(series, window=window)
        days = []
        last_extreme_index = None

        for i in range(len(rsi_series)):
            current_rsi = rsi_series.iloc[i]

            if pd.isna(current_rsi):
                days.append(np.nan)
                continue
            
            if (mode == "gt" and current_rsi > threshold) or (mode == "lt" and current_rsi < threshold):
                last_extreme_index = i
                days.append(0)
            else:
                if last_extreme_index is None:
                    days.append(np.nan)
                else:
                    days.append(i - last_extreme_index)

        days_series = pd.Series(days, index=series.index)
        return np.log1p(days_series)
    
    @staticmethod
    def backward_cum_return(series: pd.Series, days: int) -> pd.Series:
        """
        计算过去 days 天的复合收益率（从 t-days 到 t）
        :param series: 累计收益率序列（即净值序列）
        :param days: 向后的窗口长度
        :return: 复合收益率序列
        """
        return series / series.shift(days) - 1