import pandas as pd
from typing import List, Optional
import logging


logger = logging.getLogger(__name__)

from app.data_fetcher.calender_fetcher import CalendarFetcher

class RebalanceDateGenerator:
    def __init__(self, freq: str, anchor: str = 'start', custom_months: Optional[List[int]] = None):
        self.freq = freq  # daily / weekly / monthly / yearly / custom_months
        self.anchor = anchor
        self.custom_months = custom_months
        self.calendar_fetcher = CalendarFetcher()

    def get_dates_from_range(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        trade_dates_str = self.calendar_fetcher.get_trade_date(
            start=start_date.replace("-", ""), end=end_date.replace("-", ""), format="%Y-%m-%d"
        )
        trade_dates = pd.to_datetime(trade_dates_str)

        if self.freq == 'daily':
            return trade_dates

        if self.freq == 'custom_months':
            # 先筛出指定月份
            trade_dates = trade_dates[trade_dates.month.isin(self.custom_months)]
            group = trade_dates.to_series().groupby(lambda x: (x.year, x.month))
        elif self.freq == 'monthly':
            group = trade_dates.to_series().groupby(lambda x: (x.year, x.month))
        elif self.freq == 'yearly':
            group = trade_dates.to_series().groupby(lambda x: x.year)
        else:
            raise ValueError("Unsupported frequency")

        dates = []
        for _, g in group:
            if g is not None and len(g) > 0:
                selected = g.index[0] if self.anchor == 'start' else g.index[-1]
                if start_date <= selected.strftime("%Y-%m-%d") <= end_date:
                    dates.append(selected)

        return pd.DatetimeIndex(sorted(set(dates)))
    
    def get_prev_balance_date(self, current_date: str) -> Optional[pd.Timestamp]:
        start = (pd.to_datetime(current_date) - pd.Timedelta(days=366)).strftime("%Y-%m-%d")
        trade_dates_str = self.calendar_fetcher.get_trade_date(start=start.replace("-", ""), end=current_date.replace("-", ""), format="%Y-%m-%d")
        trade_dates = pd.to_datetime(trade_dates_str)
        rebalance_dates = self.get_dates_from_range(trade_dates[0].strftime("%Y-%m-%d"), current_date)
        rebalance_dates = rebalance_dates[rebalance_dates < pd.to_datetime(current_date)]
        if len(rebalance_dates) == 0:
            return None
        return rebalance_dates.max()