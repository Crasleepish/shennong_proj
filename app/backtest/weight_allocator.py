from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class WeightAllocator(ABC):
    @abstractmethod
    def allocate(self, stock_list: List[str], data: Dict[str, pd.DataFrame], asof_date: pd.Timestamp) -> Dict[str, float]:
        pass

class EqualWeightAllocator(WeightAllocator):
    def allocate(self, stock_list: List[str], data: Dict[str, pd.DataFrame], asof_date: pd.Timestamp) -> Dict[str, float]:
        n = len(stock_list)
        return {s: 1.0 / n for s in stock_list} if n > 0 else {}

class MktCapWeightAllocator(WeightAllocator):
    def allocate(self, stock_list: List[str], data: Dict[str, pd.DataFrame], asof_date: pd.Timestamp) -> Dict[str, float]:
        mkt_cap_df = data.get("mkt_cap")
        if mkt_cap_df is None or asof_date not in mkt_cap_df.index:
            return {}

        cap_series = mkt_cap_df.loc[asof_date][stock_list].dropna()
        total = cap_series.sum()
        if total == 0:
            return {}
        return (cap_series / total).to_dict()

