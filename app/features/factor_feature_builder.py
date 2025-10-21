import pandas as pd
import numpy as np
from typing import Optional, Iterable

# 可选：计划交易日历读取（推荐）
try:
    from app.data_fetcher.trade_calender_reader import TradeCalendarReader  # 注意：你项目里文件名是 calender
    _HAS_TRADE_CAL_READER = True
except Exception:
    TradeCalendarReader = None
    _HAS_TRADE_CAL_READER = False


class FactorFeatureBuilder:
    """
    构建因子特征（基于因子每日净值）
    输入应为日度净值时间序列（index=date, columns=factor_name）
    输出为特征 DataFrame（多列）
    """

    @staticmethod
    def _as_datetime_index(index_like: Iterable) -> pd.DatetimeIndex:
        """
        工具：索引规范化（兼容 datetime.date / DatetimeIndex）
        将可能由 datetime.date、numpy datetime64[D] 或 object 组成的索引，规范化为 DatetimeIndex（ns 精度）。
        """
        idx = pd.Index(index_like)
        if isinstance(idx, pd.DatetimeIndex):
            return idx
        # 统一转为 DatetimeIndex
        dt_idx = pd.to_datetime(idx, errors="raise")
        if not isinstance(dt_idx, pd.DatetimeIndex):
            raise TypeError("无法把索引转换为 DatetimeIndex。")
        return dt_idx

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
        return (series - ma) / ma

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

    @staticmethod
    def cal_gap_days(series: pd.Series) -> pd.Series:
        """
        1) 距上一交易日的自然日间隔（gap）：
           gap_t = (t - t_prev).days；首日填 1。
        """
        orig_index = series.index
        idx = FactorFeatureBuilder._as_datetime_index(orig_index)
        gap = idx.to_series().diff().dt.days
        gap = gap.fillna(1).astype(int)
        gap.index = orig_index  # 保持与输入索引类型一致（可能是 datetime.date）
        return gap

    @staticmethod
    def cal_is_month_start_7d(series: pd.Series) -> pd.Series:
        """
        2) 月初交易日：自然日 1~7 号
        """
        orig_index = series.index
        idx = FactorFeatureBuilder._as_datetime_index(orig_index)
        s = pd.Series((idx.day <= 7).astype(int), index=orig_index)
        return s

    @staticmethod
    def cal_is_month_end_7d(series: pd.Series) -> pd.Series:
        """
        3) 月末交易日：该月最后 7 个自然日
        """
        orig_index = series.index
        idx = FactorFeatureBuilder._as_datetime_index(orig_index)
        dim = idx.days_in_month
        s = pd.Series((idx.day >= (dim - 6)).astype(int), index=orig_index)
        return s

    @staticmethod
    def cal_is_pre_holiday(
        series: pd.Series,
        min_gap: int = 4,
        calendar: Optional[Iterable] = None,
        use_fallback_if_no_calendar: bool = True,
    ) -> pd.Series:
        """
        4) 节前（PIT）：基于“计划交易日历”判定：
           若下一计划交易日与当日相差 >= min_gap 个自然日，则为 1，否则 0。
           ——不依赖真实成交的‘下一交易日’，避免信息泄露。
        
        参数
        ----
        series : pd.Series
            任意日度时间序列，仅使用其 index。
        min_gap : int
            判定阈值（默认 4）。
        calendar : Optional[Iterable]
            可传入 list / Index / DatetimeIndex / 含 datetime.date 的序列，作为计划交易日历。
        use_fallback_if_no_calendar : bool
            若无法获得计划日历，是否降级为基于 index 的 next-day 间隔近似（默认 True）。
        """
        orig_index = series.index
        idx = FactorFeatureBuilder._as_datetime_index(orig_index)

        # 1) 获取计划交易日历 DatetimeIndex（优先使用传入的 calendar）
        cal_idx: Optional[pd.DatetimeIndex] = None
        if calendar is not None:
            cal_idx = FactorFeatureBuilder._as_datetime_index(calendar)

        # 2) 若未显式传入且有 Reader，则尝试读取更宽窗口的计划日历
        if cal_idx is None and _HAS_TRADE_CAL_READER:
            try:
                # 尝试以 idx 的边界扩 370 天，避免边界溢出
                start = (idx.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                end = (idx.max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                # 根据你们的实现自行调整签名（常见有 start/end 或仅 end）
                try:
                    cal_str = TradeCalendarReader.get_trade_dates(start=start, end=end)
                except TypeError:
                    cal_str = TradeCalendarReader.get_trade_dates(end=end)
                cal_idx = FactorFeatureBuilder._as_datetime_index(pd.to_datetime(cal_str))
            except Exception:
                cal_idx = None  # 读取失败则考虑 fallback

        # 3) 基于计划日历做判断
        if cal_idx is not None and len(cal_idx) > 0:
            cal_values = cal_idx.values  # numpy datetime64[ns]
            # 每个 t 在计划日历中的插入位置
            pos = np.searchsorted(cal_values, idx.values)
            next_pos = pos + 1
            valid = next_pos < len(cal_values)

            next_dates = np.empty(len(idx), dtype="datetime64[ns]")
            next_dates[~valid] = np.datetime64("NaT")
            next_dates[valid] = cal_values[next_pos[valid]]

            next_gap_days = (next_dates - idx.values).astype("timedelta64[D]").astype("float")
            is_pre = pd.Series((next_gap_days >= float(min_gap)), index=orig_index).fillna(False).astype(int)
            return is_pre

        # 4) fallback（可关闭）：用实际索引的“下一交易日”间隔近似
        if use_fallback_if_no_calendar:
            idx_series = pd.Series(idx, index=orig_index)  # 保留原索引
            next_day = idx_series.shift(-1)
            next_gap = (next_day - pd.to_datetime(orig_index)).dt.days
            return (next_gap >= int(min_gap)).fillna(False).astype(int)

        # 5) 无可用计划日历且禁用 fallback：返回全 0
        return pd.Series(0, index=orig_index, dtype=int)