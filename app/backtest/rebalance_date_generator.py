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
        trade_dates_str = self.calendar_fetcher.get_trade_date(start=start_date.replace("-", ""), end=end_date.replace("-", ""), format="%Y-%m-%d")
        trade_dates = pd.to_datetime(trade_dates_str)

        if self.freq == 'daily':
            return trade_dates
        if self.freq == 'monthly':
            group = trade_dates.to_series().groupby(lambda x: (x.year, x.month))
        elif self.freq == 'yearly':
            group = trade_dates.to_series().groupby(lambda x: x.year)
        elif self.freq == 'custom_months':
            group = trade_dates.to_series().groupby(lambda x: (x.year, x.month) if x.month in self.custom_months else None)
        else:
            raise ValueError("Unsupported frequency")

        dates = []
        for _, g in group:
            if g is not None and len(g) > 0:
                if self.anchor == 'start':
                    dates.append(g.index[0])
                else:
                    dates.append(g.index[-1])
        return pd.DatetimeIndex(sorted(set(dates)))
