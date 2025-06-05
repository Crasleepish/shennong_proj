import pandas as pd
import numpy as np
from typing import Optional


class FactorFeatureBuilder:
    """
    构建因子特征（基于因子每日净值）
    输入应为日度净值时间序列（index=date, columns=factor_name）
    输出为特征 DataFrame（多列）
    """

    @staticmethod
    def calc_cum_return(df: pd.DataFrame) -> pd.DataFrame:
        return (1 + df).cumprod()

    @staticmethod
    def rolling_zscore(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
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
    def bb_position(series: pd.Series, window: int, num_std: float = 2.0) -> pd.Series:
        ma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        return (series - lower) / (upper - lower)

    @staticmethod
    def bb_break_signal(series: pd.Series, window: int, num_std: float = 2.0) -> pd.Series:
        ma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        break_up = (series > upper).astype(int)
        break_down = (series < lower).astype(int)
        return break_up - break_down

    @staticmethod
    def ma_diff(series: pd.Series, window: int) -> pd.Series:
        ma = series.rolling(window).mean()
        return series - ma

    @staticmethod
    def ma_cross_signal(series: pd.Series, short_window: int, long_window: int) -> pd.Series:
        ma_short = series.rolling(short_window).mean()
        ma_long = series.rolling(long_window).mean()
        signal = (ma_short > ma_long).astype(int)
        return signal.diff().fillna(0)

    @staticmethod
    def backward_cum_return(series: pd.Series, days: int) -> pd.Series:
        return series / series.shift(days) - 1

    @staticmethod
    def momentum(series: pd.Series, window: int) -> pd.Series:
        return series.pct_change(periods=window)

    @staticmethod
    def slope_of_ma(series: pd.Series, ma_window: int, slope_window: int) -> pd.Series:
        ma = series.rolling(ma_window).mean()
        return FactorFeatureBuilder.regression_slope(ma, slope_window)
