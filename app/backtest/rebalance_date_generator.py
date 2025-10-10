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
        """
        返回 [start_date, end_date] 区间内的再平衡日：
        1) 先确定区间内的“再平衡月”（或年 / 自定义月份）
        2) 对每个再平衡月（或年）独立获取该月（或年）全量交易日，再选首/末个交易日
        3) 再用 [start_date, end_date] 进行最终过滤
        """
        s = pd.to_datetime(start_date)
        e = pd.to_datetime(end_date)

        # 辅助：取某段自然日内全部交易日（闭区间）
        def _trade_dates(d1: pd.Timestamp, d2: pd.Timestamp) -> pd.DatetimeIndex:
            ts = self.calendar_fetcher.get_trade_date(
                start=d1.strftime("%Y%m%d"),
                end=d2.strftime("%Y%m%d"),
                format="%Y-%m-%d"
            )
            return pd.to_datetime(ts)

        if self.freq == 'daily':
            # 直接取区间交易日
            return _trade_dates(s, e)

        dates: list[pd.Timestamp] = []

        if self.freq in ('monthly', 'custom_months'):
            # 计算 [s, e] 覆盖到的所有自然月
            months = pd.period_range(s.to_period('M'), e.to_period('M'), freq='M')
            if self.freq == 'custom_months':
                # 仅保留指定月份
                months = [p for p in months if p.month in (self.custom_months or [])]

            for p in months:
                m_start = pd.Timestamp(p.start_time.date())
                # 月末自然日（含闰月），用下一月首日-1天获得
                m_end = (p + 1).start_time - pd.Timedelta(days=1)

                # 为防止“end_date 在当月中途”导致漏取当月末交易日，这里对整月取交易日
                td = _trade_dates(m_start, m_end)
                if len(td) == 0:
                    continue

                selected = td[0] if self.anchor == 'start' else td[-1]
                dates.append(selected)

        elif self.freq == 'yearly':
            # 计算 [s, e] 覆盖到的所有年份
            years = range(s.year, e.year + 1)
            for y in years:
                y_start = pd.Timestamp(f"{y}-01-01")
                y_end = pd.Timestamp(f"{y}-12-31")
                td = _trade_dates(y_start, y_end)
                if len(td) == 0:
                    continue
                selected = td[0] if self.anchor == 'start' else td[-1]
                dates.append(selected)

        else:
            raise ValueError("Unsupported frequency")

        # 第3步：按 [s, e] 做最终过滤（闭区间）
        dates = [d for d in dates if (d >= s and d <= e)]
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