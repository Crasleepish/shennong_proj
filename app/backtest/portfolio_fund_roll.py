'''

### 本组合为场外基金轮动组合
基金池
"000218", "003376", "005561", "240016"
'''
from app.data.helper import *
import os
import pandas as pd
import numpy as np
import vectorbt as vbt
import logging
from numba import njit
from vectorbt.portfolio.enums import Direction, OrderStatus, NoOrder, CallSeqType, SizeType
from vectorbt.portfolio import nb
from scripts.fund_regression import annualized_return, max_drawdown, sharpe_ratio

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.INFO)

slippage_rate = 0.0
output_dir = r"./result"
output_prefix = "portfolio_fund"
selected_funds = ["100032", "004253", "003376"]
weights_of_assets = {"100032": 0.40, "004253": 0.12, "003376": 0.48}

def get_rebalance_dates(prices: pd.DataFrame, start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    根据价格数据的交易日期，返回每年3月6月9月和12月最后一个交易日作为再平衡日，
    仅在指定日期区间内。
    """
    dates = prices.loc[start_date:end_date].index
    rebalance_dates = []
    for year in sorted(dates.year.unique()):
        for month in [3, 6, 9, 12]:
            month_dates = dates[(dates.year == year) & (dates.month == month)]
            if not month_dates.empty:
                rebalance_dates.append(month_dates[-1])
    return pd.DatetimeIndex(sorted(rebalance_dates))

def safe_value(val):
    """将 NaN 转换为 None"""
    return None if pd.isna(val) else val


# ---------------------------
# 策略回测实现（使用 from_orders）
# ---------------------------
def backtest_strategy(start_date: str, end_date: str):
    logger.info("Starting backtest strategy from %s to %s", start_date, end_date)
    
    # 加载数据
    prices = get_fund_prices_by_code_list(selected_funds, start_date, end_date).loc[start_date:end_date]
    fees_rate_dict = get_fund_fees_by_code_list(selected_funds)

    # 获取再平衡日期
    rb_dates = get_rebalance_dates(prices, start_date, end_date)
    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)  # 目标持仓矩阵
    logger.info("Identified %d rebalance dates.", len(rb_dates))
    
    # 初始化订单矩阵
    init_cash = 100000
    order_sizes = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # 初始化动态费率矩阵
    fee_rate_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    # 维护动态持仓状态（用于计算订单）
    current_positions = pd.Series(0.0, index=prices.columns)  # 当前持仓数量
    segment_mask = np.full((prices.shape[0], 1), False, dtype=bool)
    current_cash = init_cash  # 当前现金
    
    # 填充目标持仓矩阵和调仓日标记
    for i, date in enumerate(prices.index):
        if date in rb_dates:
            rb_date = date
            segment_mask[i, 0] = True
            logger.info("Processing rebalance date: %s", rb_date.strftime("%Y-%m-%d"))
        
            # 确保调仓日的价格无缺失或异常
            if prices.loc[rb_date, selected_funds].isnull().any():
                logger.warning("Missing price data for selected assets on %s; skipping.", rb_date.strftime("%Y-%m-%d"))
                return
            logger.info("Selected assets count on %s: %d", rb_date.strftime("%Y-%m-%d"), len(selected_funds))
            # 填充目标持仓（示例：市值加权）
            target_weights.loc[rb_date, selected_funds] = weights_of_assets

            # 填充动态费率
            fee_rate_df.loc[rb_date, selected_funds] = fees_rate_dict

            current_prices = prices.loc[rb_date]
            holdings_value = (current_positions * current_prices).sum()
            total_asset = holdings_value + current_cash

            # 计算目标持仓
            target_positions = np.floor((target_weights.loc[rb_date] * total_asset) / current_prices)
            target_positions = target_positions.fillna(0)
            diff_positions = target_positions - current_positions
            diff_value = diff_positions * current_prices
            assumpt_left_balance = -1 * (diff_value[diff_value < 0] * (1 - fee_rate_df.loc[rb_date] - slippage_rate)).sum() + current_cash
            assumpt_need_amount = (diff_value[diff_value > 0] * (1 + fee_rate_df.loc[rb_date] + slippage_rate)).sum()
            while assumpt_left_balance < assumpt_need_amount:
                scale_factor = assumpt_left_balance / assumpt_need_amount
                target_positions = np.floor(target_positions * scale_factor)
                diff_positions = target_positions - current_positions
                diff_value = diff_positions * current_prices
                assumpt_left_balance = -1 * (diff_value[diff_value < 0] * (1 - fee_rate_df.loc[rb_date] - slippage_rate)).sum() + current_cash
                assumpt_need_amount = (diff_value[diff_value > 0] * (1 + fee_rate_df.loc[rb_date] + slippage_rate)).sum()
            order_sizes.loc[rb_date] = diff_positions
            current_positions = target_positions
            current_cash = assumpt_left_balance - assumpt_need_amount


    # 输出当日持仓详情到 CSV
    target_weights.loc[rb_dates].to_csv(os.path.join(output_dir, output_prefix + "_portfolio.csv"))

    # -------------------------------
    # 定义回测所需的回调函数
    # -------------------------------

    # 组级预处理：为当前组初始化排序数组
    @njit(cache=True)
    def pre_group_func_nb(c):
        order_value_out = np.empty(c.group_len, dtype=np.float64)
        return (order_value_out,)

    # 段级预处理：更新每个资产的最新估值，并根据目标订单参数排序订单调用顺序
    @njit(cache=True)
    def pre_segment_func_nb(c, order_value_out, size, price, size_type, direction):
        # 更新每个资产的最后有效价格
        for col in range(c.from_col, c.to_col):
            c.last_val_price[col] = nb.get_col_elem_nb(c, col, price)
        # 根据 size、size_type、direction 和当前状态计算订单“价值”，并排序 call sequence
        nb.sort_call_seq_nb(c, size, size_type, direction, order_value_out)
        return ()

    # 订单生成函数：从广播的参数中提取当前资产的数值，生成订单
    @njit(cache=True)
    def order_func_nb(c, size, price, size_type, direction, fees, slippage):
        if nb.get_elem_nb(c, size) == 0:
            return nb.NoOrder
        print(">>>generate order: idx=", c.i, 
              ", col=", c.col, 
              ", size=", nb.get_elem_nb(c, size),
              ", price=", nb.get_elem_nb(c, price), 
              ", size_type=", nb.get_elem_nb(c, size_type),
              ", direction=", nb.get_elem_nb(c, direction),
              ", fees=", nb.get_elem_nb(c, fees),
              ", slippage=", nb.get_elem_nb(c, slippage))
        return nb.order_nb(
            size=nb.get_elem_nb(c, size),
            price=nb.get_elem_nb(c, price),
            size_type=nb.get_elem_nb(c, size_type),
            direction=nb.get_elem_nb(c, direction),
            fees=nb.get_elem_nb(c, fees),
            slippage=nb.get_elem_nb(c, slippage)
        )

    # -------------------------------
    # 构造 Portfolio
    # -------------------------------

    # 利用占位符Rep传递参数，并利用broadcast_named_args完成广播
    pf = vbt.Portfolio.from_order_func(
        prices,
        order_func_nb,
        # 订单生成函数的参数：首先传入订单 size（此处用目标权重），然后price, size_type, direction, fees, slippage
        vbt.Rep('size'),
        vbt.Rep('price'),
        vbt.Rep('size_type'),
        vbt.Rep('direction'),
        vbt.Rep('fees'),
        vbt.Rep('slippage'),
        # 每次调仓开始为一个segment开始（在调仓日更新订单排序）
        segment_mask=segment_mask,
        pre_group_func_nb=pre_group_func_nb,
        pre_segment_func_nb=pre_segment_func_nb,
        # pre_segment_func_nb 的附加参数（注意：这里传入的 'size' 实际为目标权重）
        pre_segment_args=(vbt.Rep('size'), vbt.Rep('price'), vbt.Rep('size_type'), vbt.Rep('direction')),
        broadcast_named_args=dict(
            price=prices,                # 订单价格使用收盘价数据
            size=order_sizes,         # 目标权重矩阵作为订单 size（用百分比表示）
            size_type=SizeType.Amount,   # 订单类型： 按数量
            direction=Direction.Both, # 订单方向：双向
            fees=fee_rate_df,                  # 交易费用（动态费用也可以传入Series或DataFrame）
            freq='B',
            slippage=slippage_rate               # 滑点
        ),
        cash_sharing=True,  # 同一组内共享现金
        group_by=True,      # 所有资产构成一个组
        init_cash=init_cash
    )
    
    # 输出结果（与原始代码相同）
    logger.info("Order detail:")
    logger.info(pf.orders.records_readable)
    total_value = pf.value()
    daily_returns = pf.returns()
    value_df = total_value.to_frame("value").reset_index(drop=False)
    # logger.info(pf.stats())
    total_value.to_csv(os.path.join(output_dir, output_prefix + "_total_value.csv"))
    daily_returns.to_csv(os.path.join(output_dir, output_prefix + "_daily_returns.csv"))
    logger.info("Latest Value: %s", pf.final_value())
    logger.info("Total Return: %s", pf.total_return())
    logger.info("annualized return: %s", annualized_return(value_df, value_col='value'))
    logger.info("max_drawdown: %s", max_drawdown(value_df, value_col='value'))
    logger.info("sharpe ratio: %s", sharpe_ratio(value_df, value_col='value', risk_free_rate=0))
    # 可视化（可选）
    fig = pf.value(group_by=True).vbt.plot(title="Portfolio Value Curve")
    fig.write_html(os.path.join(output_dir, output_prefix + "_value_plot.html"))
    return pf
