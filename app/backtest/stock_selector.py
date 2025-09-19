from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Set, Optional
import logging
import math

logger = logging.getLogger(__name__)

def get_latest_available_row(
    df: pd.DataFrame,
    asof_date: pd.Timestamp,
    check_fields: List[str] = None,
) -> pd.DataFrame:
    """
    获取截至 asof_date 前（含）一年以内、且在 check_fields 指定的字段上都不为空的最新报告行。

    :param df: 原始 DataFrame，索引需包含名为 "report_date" 的日期级别，或可通过 df.index.get_level_values("report_date") 取到报告日期。
    :param asof_date: 查询日期，如 pd.Timestamp("2025-08-06")
    :param check_fields: 需要验证非空的列名列表
    :return: 每个股票（或每个分组）最新的一行记录，索引设为股票代码（索引级别 "stock_code" 或对应列名）
    """
    # 1. 时间范围过滤：报告日期 ≤ asof_date，且 ≥ asof_date - 1 年
    one_year_ago = asof_date - pd.DateOffset(years=1)
    mask_date = (
        (df.index.get_level_values("report_date") <= asof_date) &
        (df.index.get_level_values("report_date") >= one_year_ago)
    )
    df = df.loc[mask_date]
    if df.empty:
        return pd.DataFrame()

    # 2. 指定字段均非空（当且仅当 check_fields 非 None 且长度 > 0）
    if check_fields:
        # 若需同时把空字符串视为缺失，可先做：df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna(subset=check_fields, how="any")
        if df.empty:
            return pd.DataFrame()

    # 3. 按报告日期升序排序，保证最新的排在后面
    df = df.sort_index(level="report_date")

    # 4. 分组取每组最后一条（最新的）记录
    #    如果股票代码是索引级别，用 level；若是列，请改为 df.groupby("stock_code")
    latest = df.groupby(level="stock_code").tail(1)

    # 5. 重设索引为股票代码
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
        cutoff = int(math.ceil(n * self.percentile))
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
                 score_percentile: tuple = (0.7, 1.0), lookback_range: int = 90, parents: Optional[List[Selector]] = None):
        super().__init__(parents)
        self.stock_info = stock_info
        self.fundamental_df = fundamental_df
        self.asof_date = asof_date
        self.score_percentile = score_percentile
        self.lookback_range = lookback_range

    def _select(self, universe):
        if self.fundamental_df is None:
            return []

        lookback_date = self.asof_date - pd.Timedelta(days=self.lookback_range)
        fundamentals = get_latest_available_row(self.fundamental_df, lookback_date, ["operating_profit_ttm", "total_equity", "net_cash_from_operating", "net_profit", "total_assets", "total_liabilities"])
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

        def zscore_trimmed(group: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
            result = pd.DataFrame(index=group.index, columns=cols)
            for col in cols:
                series = group[col]
                if series.count() <= 2:
                    mu = series.mean()
                    sigma = series.std()
                else:
                    # 固定丢弃一个最小值和一个最大值（保留其余）
                    drop_idx = [series.idxmin(), series.idxmax()]
                    trimmed = series.drop(index=drop_idx)
                    if trimmed.count() < 1:
                        mu = series.mean()
                        sigma = series.std()
                    else:
                        mu = trimmed.mean()
                        sigma = trimmed.std()

                if sigma == 0 or pd.isna(sigma):
                    result[col] = 0.0
                else:
                    result[col] = (series - mu) / sigma
            return result

        cols = ["profit", "cfq", "lev"]
        zscore_df = fundamentals.groupby("industry").apply(lambda g: zscore_trimmed(g, cols)).reset_index(level=0, drop=True)
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
                 asof_date: pd.Timestamp, bm_percentile: tuple = (0.7, 1.0), lookback_range: int = 90, parents: Optional[List[Selector]] = None):
        super().__init__(parents)
        self.fundamental_df = fundamental_df
        self.mkt_cap_df = mkt_cap_df
        self.asof_date = asof_date
        self.bm_percentile = bm_percentile
        self.lookback_range = lookback_range

    def _select(self, universe):
        if self.fundamental_df is None or self.mkt_cap_df is None or self.asof_date not in self.mkt_cap_df.index:
            return []

        lookback_date = self.asof_date - pd.Timedelta(days=self.lookback_range)
        fundamentals = get_latest_available_row(self.fundamental_df, lookback_date, ["total_equity"])
        if fundamentals.empty:
            return []

        if universe is not None:
            fundamentals = fundamentals.loc[fundamentals.index.intersection(universe)]
        fundamentals = fundamentals.dropna(subset=["total_equity"])
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
