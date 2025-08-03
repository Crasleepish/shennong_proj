from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)

def get_latest_available_row(df: pd.DataFrame, asof_date: pd.Timestamp) -> pd.DataFrame:
    # 筛选截至asof_date之前的所有报告
    df = df.loc[df.index.get_level_values("report_date") <= asof_date]

    if df.empty:
        return pd.DataFrame()

    # 排序确保每个股票最后一条是最新的
    df = df.sort_index(level="report_date")

    # 保留每个股票最新一条记录
    latest = df.groupby("stock_code").tail(1)
    latest.index = latest.index.get_level_values("stock_code")
    return latest

class Selector(ABC):
    def __init__(self, parents: Optional[List['Selector']] = None):
        self.parents = parents or []

    def select(self, universe: Optional[Set[str]] = None) -> List[str]:
        for p in self.parents:
            universe = set(p.select(universe))
            if not universe:
                return []
        return self._select(universe)

    @abstractmethod
    def _select(self, universe: Optional[Set[str]]) -> List[str]:
        pass


class AmountSelector(Selector):
    # 根据成交量筛选，剃除最低的一部分的股票，剃除的比例为percentile
    def __init__(self, amount: pd.DataFrame, asof_date: pd.Timestamp, percentile: float, parents: List[Selector] = None):
        super().__init__(parents)
        self.amount = amount
        self.asof_date = asof_date
        self.percentile = percentile

    def _select(self, universe):
        if self.amount is None or self.asof_date not in self.amount.index:
            return []
        amount_series = self.amount.loc[self.asof_date].dropna()
        amount_series = amount_series[amount_series.index.isin(universe)]
        if amount_series.empty:
            return []
        amount_series = amount_series.sort_values(ascending=True)
        n = len(amount_series)
        cutoff = int(n * self.percentile)
        return amount_series.iloc[cutoff:].index.tolist()


class HSExchangeSelector(Selector):
    def __init__(self, stock_info: pd.DataFrame, parents: Optional[List[Selector]] = None):
        super().__init__(parents)
        self.stock_info = stock_info

    def _select(self, universe):
        hs_exchange_stocks = set(self.stock_info[self.stock_info["exchange"].isin(["SSE", "SZSE"])].index)
        return list(hs_exchange_stocks.intersection(universe))

class ListedMoreThanOneYearSelector(Selector):
    def __init__(self, stock_info: pd.DataFrame, asof_date: pd.Timestamp, parents: Optional[List[Selector]] = None):
        super().__init__(parents)
        self.stock_info = stock_info
        self.asof_date = asof_date

    def _select(self, universe):
        listed = self.stock_info[self.stock_info['listing_date'] <= self.asof_date - pd.DateOffset(years=1)].index
        return list(listed.intersection(universe))

class HasPriceSelector(Selector):
    def __init__(self, price_df: pd.DataFrame, asof_date: pd.Timestamp, parents: Optional[List[Selector]] = None):
        super().__init__(parents)
        self.price_df = price_df
        self.asof_date = asof_date

    def _select(self, universe):
        if self.price_df is None or self.asof_date not in self.price_df.index:
            return []
        available = self.price_df.columns[self.price_df.loc[self.asof_date].notna()]
        return list(set(available).intersection(universe))

class ExcludeBlacklistSelector(Selector):
    def __init__(self, blacklist: List[str], parents: Optional[List[Selector]] = None):
        super().__init__(parents)
        self.blacklist = set(blacklist)

    def _select(self, universe):
        if universe is None:
            return []
        return list(set(universe) - self.blacklist)

class BasicSelector(Selector):
    def __init__(self, stock_info: pd.DataFrame, blacklist: List[str], price_df: pd.DataFrame, asof_date: pd.Timestamp):
        super().__init__(parents=[
            ListedMoreThanOneYearSelector(stock_info, asof_date),
            HSExchangeSelector(stock_info),
            HasPriceSelector(price_df, asof_date),
            ExcludeBlacklistSelector(blacklist)
        ])

    def _select(self, universe):
        return list(universe) if universe else []

class MktCapPercentileSelector(Selector):
    def __init__(self, mkt_cap_df: pd.DataFrame, asof_date: pd.Timestamp, percentile: tuple, parents: Optional[List[Selector]] = None):
        super().__init__(parents)
        self.mkt_cap_df = mkt_cap_df
        self.asof_date = asof_date
        self.percentile = percentile

    def _select(self, universe):
        if self.mkt_cap_df is None or self.asof_date not in self.mkt_cap_df.index:
            return []
        cap_series = self.mkt_cap_df.loc[self.asof_date].dropna()
        if universe is not None:
            cap_series = cap_series[cap_series.index.isin(universe)]
        sorted_caps = cap_series.sort_values(ascending=True)
        n = len(sorted_caps)
        if n == 0:
            return []
        lower, upper = int(n * self.percentile[0]), int(n * self.percentile[1])
        return sorted_caps.iloc[lower:upper].index.tolist()

class QualityScoreSelector(Selector):
    def __init__(self, stock_info: pd.DataFrame, fundamental_df: pd.DataFrame, asof_date: pd.Timestamp,
                 score_percentile: tuple = (0.7, 1.0), parents: Optional[List[Selector]] = None):
        super().__init__(parents)
        self.stock_info = stock_info
        self.fundamental_df = fundamental_df
        self.asof_date = asof_date
        self.score_percentile = score_percentile

    def _select(self, universe):
        if self.fundamental_df is None:
            return []

        lookback_date = self.asof_date - pd.Timedelta(days=120)
        fundamentals = get_latest_available_row(self.fundamental_df, lookback_date)
        if fundamentals.empty:
            return []

        # if universe is not None:
            # fundamentals = fundamentals.loc[fundamentals.index.intersection(universe)]
        fundamentals = fundamentals.dropna(subset=[
            "operating_profit_ttm", "total_equity",
            "net_cash_from_operating", "net_profit",
            "total_assets", "total_liabilities"])
        if fundamentals.empty:
            return []

        fundamentals["profit"] = fundamentals["operating_profit_ttm"] / fundamentals["total_equity"]
        fundamentals["cfq"] = fundamentals["net_cash_from_operating"] / fundamentals["net_profit"]
        fundamentals["lev"] = fundamentals["total_liabilities"] / fundamentals["total_assets"]

        fundamentals["industry"] = self.stock_info.reindex(fundamentals.index)["industry"]
        zscore_df = fundamentals.groupby("industry")[["profit", "cfq", "lev"]].transform(lambda x: (x - x.mean()) / x.std())
        if universe is not None:
            zscore_df = zscore_df.loc[zscore_df.index.intersection(universe)]
        score = zscore_df["profit"] + zscore_df["cfq"] - zscore_df["lev"]
        score = score.dropna().sort_values(ascending=True)
        n = len(score)
        if n < 10:
            return []
        lower, upper = int(n * self.score_percentile[0]), int(n * self.score_percentile[1])
        return score.iloc[lower:upper].index.tolist()

class BMScoreSelector(Selector):
    def __init__(self, fundamental_df: pd.DataFrame, mkt_cap_df: pd.DataFrame, 
                 asof_date: pd.Timestamp, bm_percentile: tuple = (0.7, 1.0), parents: Optional[List[Selector]] = None):
        super().__init__(parents)
        self.fundamental_df = fundamental_df
        self.mkt_cap_df = mkt_cap_df
        self.asof_date = asof_date
        self.bm_percentile = bm_percentile

    def _select(self, universe):
        if self.fundamental_df is None or self.mkt_cap_df is None or self.asof_date not in self.mkt_cap_df.index:
            return []

        lookback_date = self.asof_date - pd.Timedelta(days=120)
        fundamentals = get_latest_available_row(self.fundamental_df, lookback_date)
        if fundamentals.empty:
            return []

        if universe is not None:
            fundamentals = fundamentals.loc[fundamentals.index.intersection(universe)]
        fundamentals = fundamentals.dropna(subset=["total_equity"])
        fundamentals = fundamentals[fundamentals["total_equity"] > 0]
        if fundamentals.empty:
            return []

        cap_series = self.mkt_cap_df.loc[self.asof_date].reindex(fundamentals.index)
        cap_series = cap_series[cap_series > 1e-6]
        fundamentals = fundamentals.loc[cap_series.index]

        bm = fundamentals["total_equity"] / cap_series
        bm = bm.dropna().sort_values(ascending=True)

        n = len(bm)
        if n < 10:
            return []
        lower, upper = int(n * self.bm_percentile[0]), int(n * self.bm_percentile[1])
        return bm.iloc[lower:upper].index.tolist()
