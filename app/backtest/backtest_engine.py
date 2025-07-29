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
from numba import njit
import vectorbt as vbt
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import SizeType, Direction

__all__ = ["BacktestConfig", "run_backtest"]

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
# Order factory (Numba JIT)
# ---------------------------------------------------------------------
@njit
def _order_func_nb(c, target_w, price, buy_fee, sell_fee, slippage):
    """Generate **one** TargetPercent order toward *target_w* for this element."""
    w = target_w[c.i, c.col]
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
# Pre-segment function for sorting call sequence (sell before buy)
# ---------------------------------------------------------------------
@njit
def _pre_segment_func_nb(c, size, price, size_type, direction):
    # 更新每个资产的最后有效价格
    for col in range(c.from_col, c.to_col):
        c.last_val_price[col] = nb.get_col_elem_nb(c, col, price)
    order_value_out = np.empty(c.to_col - c.from_col, dtype=np.float64)
    nb.sort_call_seq_nb(c, size, size_type, direction, order_value_out)
    return ()

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
    weights, close = weights.align(close, join="inner", axis=0)
    weights, close = weights.align(close, join="inner", axis=1)
    if weights.empty or close.empty:
        raise ValueError("After alignment, weights/close share no common index or columns.")

    if weights.isna().all().all():
        raise ValueError("Weights are all NaN – nothing to do.")

    # 2) Convert to ndarray for Numba -----------------------------------
    w_arr     = weights.to_numpy(dtype=np.float64)
    price_arr = close.to_numpy(dtype=np.float64)

    # 3) Construct segment mask from weight index -----------------------
    all_dates = close.index
    seg_mask = np.isin(all_dates, weights.index)[:, None]

    # 4) Vectorbt simulation --------------------------------------------
    pf = vbt.Portfolio.from_order_func(
        close,
        _order_func_nb,
        w_arr,
        price_arr,
        cfg.buy_fee,
        cfg.sell_fee,
        cfg.slippage,
        init_cash=cfg.init_cash,
        cash_sharing=cfg.cash_sharing,
        group_by=True,
        use_numba=True,
        freq=cfg.freq,
        segment_mask=seg_mask,
        pre_segment_func_nb=_pre_segment_func_nb,
        pre_segment_args=(
            w_arr,
            price_arr,
            np.full_like(w_arr, SizeType.TargetPercent),
            np.full_like(w_arr, Direction.LongOnly),
        ),
    )

    # 5) Assemble outputs and print order records ----------------------
    nav = pf.value()
    rets = pf.returns()
    aw = pf.asset_value(group_by=False).div(pf.value(), axis=0)
    err = aw.reindex_like(weights) - weights

    print("\n===== Order Records =====")
    order_rec = pf.orders.records.copy()
    order_rec['timestamp'] = pf.wrapper.index[order_rec['idx']]
    order_rec['asset'] = pf.wrapper.columns[order_rec['col']]
    print(order_rec[['timestamp', 'asset', 'size', 'price', 'fees', 'side']])

    stats = pf.returns_stats(defaults=dict(freq=cfg.freq))

    return {
        "pf": pf,
        "nav": nav,
        "returns": rets,
        "actual_weights": aw,
        "weight_error": err,
        "stats": stats,
    }
