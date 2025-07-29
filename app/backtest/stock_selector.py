from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def get_latest_available_row(df: pd.DataFrame, asof_date: pd.Timestamp) -> pd.Series:
    """
    从 DataFrame 中获取不晚于 asof_date 的最新一行数据（按 index 作为日期处理）。
    如果没有满足条件的数据，返回空 Series。
    """
    valid_dates = df.index[df.index <= asof_date]
    if valid_dates.empty:
        return pd.Series(dtype='float32')
    latest_date = valid_dates.max()
    return df.loc[latest_date]

class StockSelector(ABC):
    def __init__(self, stock_info_df: pd.DataFrame, blacklist: List[str]):
        self.stock_info = stock_info_df
        self.blacklist = blacklist

    @abstractmethod
    def select(self, data: Dict[str, pd.DataFrame], asof_date: pd.Timestamp, blacklist: List[str]) -> List[str]:
        pass


class BasicStockSelector(StockSelector):
    def select(self, data: Dict[str, pd.DataFrame], asof_date: pd.Timestamp) -> List[str]:
        # 1. 过滤已上市
        listed = self.stock_info[self.stock_info['listing_date'] <= asof_date].index.tolist()

        # 2. 过滤当日有价格数据的股票
        price_df = data.get('price')
        if price_df is None or asof_date not in price_df.index:
            return []
        available = price_df.columns[price_df.loc[asof_date].notna()].tolist()

        # 3. 排除黑名单
        result = list(set(listed).intersection(available).difference(self.blacklist))
        return sorted(result)
    

# === QualitySelector：质量三因子行业中性化选股 ===
class QualitySelector(StockSelector):
    def __init__(self, stock_info_df: pd.DataFrame, blacklist: List[str], size_percentile: tuple = (0.5, 1.0), score_percentile: tuple = (0.7, 1.0)):
        super().__init__(stock_info_df, blacklist)
        self.size_percentile = size_percentile  # e.g. (0.25, 0.75)
        self.score_percentile = score_percentile  # e.g. (0.7, 1.0)

    def select(self, data: Dict[str, pd.DataFrame], asof_date: pd.Timestamp) -> List[str]:
        # 1. 初筛：上市满一年 & 有价格
        listed = self.stock_info[self.stock_info['listing_date'] <= asof_date - pd.DateOffset(years=1)]
        price_df = data.get("price")
        if price_df is None or asof_date not in price_df.index:
            return []
        available = price_df.columns[price_df.loc[asof_date].notna()]
        universe = set(listed.index) & set(available) - set(self.blacklist)

        # 2. 市值分组（按百分位）
        mkt_cap = data.get("mkt_cap")
        if mkt_cap is None or asof_date not in mkt_cap.index:
            return []
        cap_series = mkt_cap.loc[asof_date].dropna()
        cap_universe = list(universe & set(cap_series.index))
        if not cap_universe:
            return []
        sorted_caps = cap_series[cap_universe].sort_values()
        n_cap = len(sorted_caps)
        lower_idx = int(n_cap * self.size_percentile[0])
        upper_idx = int(n_cap * self.size_percentile[1])
        cap_group = sorted_caps.iloc[lower_idx:upper_idx].index

        # 3. 获取基本面数据
        fundamental_df = data.get("fundamental")
        if fundamental_df is None:
            return []
        lookback_date = asof_date - pd.Timedelta(days=120)
        sub_df = get_latest_available_row(fundamental_df, lookback_date)
        if sub_df.empty:
            return []

        sub_df = sub_df.loc[cap_group.intersection(sub_df.index)].copy()
        sub_df = sub_df.dropna(subset=[
            "operating_profit_ttm", "total_equity",
            "net_cash_from_operating", "net_profit",
            "total_assets", "total_liabilities"
        ])
        if sub_df.empty:
            return []

        # 4. 构建因子
        sub_df["profit"] = sub_df["operating_profit_ttm"] / sub_df["total_equity"]
        sub_df["cfq"] = sub_df["net_cash_from_operating"] / sub_df["net_profit"]
        sub_df["lev"] = sub_df["total_liabilities"] / sub_df["total_assets"]

        # 5. 行业中性化 Z 分数
        ind = self.stock_info["industry"].reindex(sub_df.index)
        sub_df["industry"] = ind
        zscore_df = sub_df.groupby("industry")[["profit", "cfq", "lev"]].transform(lambda x: (x - x.mean()) / x.std())
        score = zscore_df["profit"] + zscore_df["cfq"] - zscore_df["lev"]
        score = score.dropna().sort_values(ascending=False)

        # 6. 百分位筛选
        n = len(score)
        if n < 10:
            return []
        lower, upper = int(n * self.score_percentile[0]), int(n * self.score_percentile[1])
        return score.iloc[lower:upper].index.tolist()


# === BMSelector：账面市值比因子选股 ===
class BMSelector(StockSelector):
    def __init__(self, stock_info_df: pd.DataFrame, blacklist: List[str], size_percentile: tuple = (0.5, 1.0), bm_percentile: tuple = (0.7, 1.0)):
        super().__init__(stock_info_df, blacklist)
        self.size_percentile = size_percentile
        self.bm_percentile = bm_percentile

    def select(self, data: Dict[str, pd.DataFrame], asof_date: pd.Timestamp) -> List[str]:
        # 1. 初筛
        listed = self.stock_info[self.stock_info['listing_date'] <= asof_date - pd.DateOffset(years=1)]
        price_df = data.get("price")
        if price_df is None or asof_date not in price_df.index:
            return []
        available = price_df.columns[price_df.loc[asof_date].notna()]
        universe = set(listed.index) & set(available) - set(self.blacklist)

        # 2. 市值过滤
        mkt_cap = data.get("mkt_cap")
        if mkt_cap is None or asof_date not in mkt_cap.index:
            return []
        cap_series = mkt_cap.loc[asof_date].dropna()
        cap_universe = list(universe & set(cap_series.index))
        if not cap_universe:
            return []
        sorted_caps = cap_series[cap_universe].sort_values()
        n_cap = len(sorted_caps)
        lower_idx = int(n_cap * self.size_percentile[0])
        upper_idx = int(n_cap * self.size_percentile[1])
        cap_group = sorted_caps.iloc[lower_idx:upper_idx].index

        # 3. 获取基本面
        fundamental_df = data.get("fundamental")
        if fundamental_df is None:
            return []
        lookback_date = asof_date - pd.Timedelta(days=120)
        sub_df = get_latest_available_row(fundamental_df, lookback_date)
        if sub_df.empty:
            return []

        sub_df = sub_df.loc[cap_group.intersection(sub_df.index)].copy()
        sub_df = sub_df.dropna(subset=["total_equity"])
        sub_df = sub_df[sub_df["total_equity"] > 0]
        if sub_df.empty:
            return []

        # 4. 计算 BM 值
        cap_vals = cap_series[sub_df.index]
        cap_vals = cap_vals[cap_vals > 1e-6]  # 避免除以零或极小值
        sub_df = sub_df.loc[cap_vals.index]
        bm = sub_df["total_equity"] / cap_vals
        bm = bm.dropna().sort_values(ascending=False)

        # 5. 分位筛选
        n = len(bm)
        if n < 10:
            return []
        lower, upper = int(n * self.bm_percentile[0]), int(n * self.bm_percentile[1])
        return bm.iloc[lower:upper].index.tolist()
