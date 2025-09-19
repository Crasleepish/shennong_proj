""" 
Universal vectorbt backtest helper
=================================

A lightweight wrapper around **vectorbt** that lets you back‑test a whole‑portfolio
*target‑weight* schedule in one call.

Inputs
------
* **weights** – `pd.DataFrame` (time × asset) of desired fractional weights (0‑1).
* **close**   – matching price matrix used for valuation and execution.

Outputs (dict)
--------------
* `nav`             – net‑asset‑value curve (Series)
* `returns`         – daily returns (Series)
* `actual_weights`  – realised post‑trade weights (DataFrame)
* `weight_error`    – difference between realised and target weights (DataFrame)
* `pf`              – the underlying `vectorbt.Portfolio` instance
* `stats`           – key performance metrics (DataFrame)

Quick‑start
-----------
>>> from universal_backtest import run_backtest, BacktestConfig
>>> out = run_backtest(weights_df, price_df, BacktestConfig())
>>> out['stats'].loc[['annualized_return', 'sharpe_ratio', 'total_fees_paid']]

Installation
------------
$ pip install vectorbt pandas numpy numba
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import logging
from numba import njit
import vectorbt as vbt
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import SizeType, Direction, SegmentContext

__all__ = ["BacktestConfig", "run_backtest"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Execution‑assumption container
# ---------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """Centralised trade‑execution assumptions."""

    buy_fee: float = 0.0005        # % of notional when *increasing* exposure
    sell_fee: float = 0.0005       # % of notional when *decreasing* exposure
    slippage: float = 0.0002       # symmetric price slippage (fraction of price)
    init_cash: float = 1_000_000.0 # starting capital
    cash_sharing: bool = True      # single cash account for all assets
    freq: str = "D"               # bar frequency for annualisation

# ---------------------------------------------------------------------
# Build mapping of price_index -> weight_index
# ---------------------------------------------------------------------
def build_sparse_index_mapping(weight_index: pd.Index, price_index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """
    构建稀疏索引映射表：将 price 中的 bar 行号映射到 weights 中的有效行号。
    用于实现 broadcast 参数在时间维度上不对齐时的映射。

    Returns:
        keys:    np.ndarray[int] of price index positions
        values:  np.ndarray[int] of weights index positions
    """
    keys, values = [], []
    weight_pos = {dt: j for j, dt in enumerate(weight_index)}
    for i, dt in enumerate(price_index):
        if dt in weight_pos:
            keys.append(i)
            values.append(weight_pos[dt])
    return np.array(keys, dtype=np.int32), np.array(values, dtype=np.int32)

@njit(cache=True)
def lookup_index(i: int, keys: np.ndarray, values: np.ndarray) -> int:
    """在 keys 中查找 i 所在位置对应的 weights 索引值"""
    for k, v in zip(keys, values):
        if k == i:
            return v
    return -1

# ---------------------------------------------------------------------
# Pre-group function: allocate order_value_out buffer
# ---------------------------------------------------------------------
@njit(cache=True)
def _pre_group_func_nb(c):
    # 分配用于排序调用顺序的缓存数组
    order_value_out = np.empty(c.group_len, dtype=np.float64)
    return (order_value_out,)

# ---------------------------------------------------------------------
# Pre-segment function for sorting call sequence (sell before buy)
# ---------------------------------------------------------------------

@njit(cache=True)
def sparse_sort_call_seq_nb(c: SegmentContext, order_value_out, target_w, direction, size_type, price_idx_arr, weight_idx_arr):
    group_value_now = nb.get_group_value_ctx_nb(c)

    row_in_w = lookup_index(c.i, price_idx_arr, weight_idx_arr)
    if row_in_w == -1:
        return  # 当前 bar 不在调仓日中，跳过排序

    for k in range(c.from_col, c.to_col):
        col = k
        w = target_w[row_in_w, col]
        # 若权重为NaN，则置为0
        if np.isnan(w):
            w = 0.0

        cash_now = c.last_cash[c.group] if c.cash_sharing else c.last_cash[col]
        free_cash_now = c.last_free_cash[c.group] if c.cash_sharing else c.last_free_cash[col]
        val_price = c.last_val_price[col]

        order_value = nb.approx_order_value_nb(
            w, size_type, direction,
            cash_now, c.last_position[col], free_cash_now,
            val_price, group_value_now
        )
        order_value_out[col - c.from_col] = order_value

    nb.insert_argsort_nb(order_value_out, c.call_seq_now)

@njit(cache=True)
def _pre_segment_func_nb(c, order_value_out, target_w, price, size_type, direction, price_idx_arr, weight_idx_arr):
    # 更新每个资产的 last_val_price 以便估值
    for col in range(c.from_col, c.to_col):
        c.last_val_price[col] = nb.get_col_elem_nb(c, col, price)
    sparse_sort_call_seq_nb(c, order_value_out, target_w, direction, size_type, price_idx_arr, weight_idx_arr)
    return ()

# ---------------------------------------------------------------------
# Order factory (Numba JIT)
# ---------------------------------------------------------------------
@njit(cache=True)
def _order_func_nb(c, target_w, price, buy_fee, sell_fee, slippage, price_idx_arr, weight_idx_arr):
    """在当前 bar 上为每个资产生成一个 TargetPercent 类型订单"""
    row_in_w = lookup_index(c.i, price_idx_arr, weight_idx_arr)
    if row_in_w == -1:
        return nb.order_nothing_nb()

    w = target_w[row_in_w, c.col]
    if np.isnan(w):
        return nb.order_nothing_nb()

    fee = buy_fee if w > c.position_now else sell_fee

    return nb.order_nb(
        size=w,
        size_type=SizeType.TargetPercent,
        price=price[c.i, c.col],
        fees=fee,
        slippage=slippage,
        direction=Direction.LongOnly,
    )


# ---------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------

def run_backtest(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    cfg: BacktestConfig | None = None,
) -> Dict[str, object]:
    """Run a weight‑schedule back‑test and return key artefacts."""

    if cfg is None:
        cfg = BacktestConfig()

	
    # 1) Align matrices --------------------------------------------------
															
    close = close.ffill().copy()
    weights = weights.fillna(0.0).copy()
    weights, close = weights.align(close, join="left", axis=1)
    if weights.empty or close.empty:
        raise ValueError("After alignment, weights/close share no common index or columns.")

    if weights.isna().all().all():
        raise ValueError("Weights are all NaN – nothing to do.")

    # Build sparse mapping: price row i -> weights row j
    price_idx_arr, weight_idx_arr = build_sparse_index_mapping(weights.index, close.index)
    # price_idx_arr: 表示 close 的第几个位置是调仓日
    # weight_idx_arr: 表示这些调仓日对应 weights 中的哪一行

    # 2) Convert to ndarray for Numba -----------------------------------
    w_arr     = weights.to_numpy(dtype=np.float64)
    price_arr = close.to_numpy(dtype=np.float64)

    # 3) Construct segment mask from weight index -----------------------
    seg_mask = np.isin(close.index, weights.index)[:, None]

    # 4) Vectorbt simulation --------------------------------------------
    pf = vbt.Portfolio.from_order_func(
        close,
        _order_func_nb,
        w_arr,
        price_arr,
        cfg.buy_fee,
        cfg.sell_fee,
        cfg.slippage,
        price_idx_arr,
        weight_idx_arr,
        init_cash=cfg.init_cash,
        cash_sharing=cfg.cash_sharing,
        group_by=True,
        use_numba=True,
        freq=cfg.freq,
        segment_mask=seg_mask,
        pre_group_func_nb=_pre_group_func_nb,
        pre_segment_func_nb=_pre_segment_func_nb,
        pre_segment_args=(
            w_arr,
            price_arr,
            SizeType.TargetPercent,
            Direction.LongOnly,
            price_idx_arr,
            weight_idx_arr,
        ),
        ffill_val_price=True,
    )

    # 5) Assemble outputs and print order records ----------------------
    nav = pf.value()
    rets = pf.returns()
    aw = pf.asset_value(group_by=False).div(pf.value(), axis=0)
    err = aw.reindex_like(weights) - weights

    logging.info("\n===== Order Records =====")
    order_rec = pf.orders.records.copy()
    order_rec['timestamp'] = pf.wrapper.index[order_rec['idx']]
    order_rec['asset'] = pf.wrapper.columns[order_rec['col']]
    logging.info(order_rec[['timestamp', 'asset', 'size', 'price', 'fees', 'side']])

    stats = pf.returns_stats(defaults=dict(freq=cfg.freq))

    return {
        "pf": pf,
        "nav": nav,
        "returns": rets,
        "actual_weights": aw,
        "weight_error": err,
        "stats": stats,
    }

