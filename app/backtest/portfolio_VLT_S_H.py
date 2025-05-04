'''
---

### **1. 分组步骤**
1. **按市值分组**：
   - 使用全市场股票市值的中位数作为分界点，将所有股票分为**小市值（S）**和**大市值（B）**两组。
2. **按VLT分组**：
   - 在**小市值（S）**组内，按VLT的30%和70%分位数将股票分为**低VLT（L）**、**中VLT（M）**和**高VLT（H）**三组。
   - 在**大市值（B）**组内，同样按VLT的30%和70%分位数将股票分为**低VLT（L）**、**中VLT（M）**和**高VLT（H）**三组。

---

### **2. 构建组合**
- 经过上述分组后，会形成 **2（市值） × 3（VLT） = 6个组合**：
  - 小市值 + 低VLT（S/L）
  - 小市值 + 中VLT（S/M）
  - 小市值 + 高VLT（S/H）
  - 大市值 + 低VLT（B/L）
  - 大市值 + 中VLT（B/M）
  - 大市值 + 高VLT（B/H）

---

### 本组合为小市值 + 高VLT（S/H）
'''
from app.data.helper import *
import os
import pandas as pd
import numpy as np
import vectorbt as vbt
import logging
from datetime import datetime, timedelta
from numba import njit
from vectorbt.portfolio.enums import Direction, OrderStatus, NoOrder, CallSeqType, SizeType
from vectorbt.portfolio import nb
from .compute_asset_growth import compute_and_cache_asset_growth, compute_and_cache_industry_avg_asset_growth
from app.utils.data_utils import calculate_column_volatility
from app.utils.data_utils import format_date

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.INFO)

fee_rate = 0.0003
slippage_rate = 0.0001
output_dir = r"./bt_result"
output_prefix = "portfolio_VLT_S_H"

def get_rebalance_dates(prices: pd.DataFrame, start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    根据价格数据的交易日期，返回每年6月和12月最后一个交易日作为再平衡日，
    仅在指定日期区间内。
    """
    dates = prices.loc[start_date:end_date].index
    rebalance_dates = []
    for year in sorted(dates.year.unique()):
        for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            month_dates = dates[(dates.year == year) & (dates.month == month)]
            if not month_dates.empty:
                rebalance_dates.append(month_dates[-1])
    return pd.DatetimeIndex(sorted(rebalance_dates))

def filter_universe(prices: pd.DataFrame, volumes: pd.DataFrame, mkt_cap: pd.DataFrame,
                    stock_info: pd.DataFrame, suspend_df: pd.DataFrame, list_status: pd.DataFrame, 
                    rb_date: pd.Timestamp) -> list:
    """
    在给定再平衡日 rb_date 下，过滤股票：
      1. 剔除成交量在当日最低 1% 的股票；
      2. 剔除上市未满 1 年的股票；
      3. 剔除停牌或过去 6 个月内发生过停牌的股票；
      4. 剔除已退市的股票（list_status != 'L'）。
    返回符合条件的股票代码列表。
    """
    # a. 剔除退市股票（只保留 list_status == 'L' 的股票）
    valid_listed = set(list_status.index[list_status['list_status'] == 'L'])

    # b. 成交量过滤：当日成交量
    vol = volumes.loc[rb_date]
    threshold = vol.quantile(0.01)
    valid_volume = set(vol[vol > threshold].index)
    
    # c. 上市时间过滤：上市时间至少在 rb_date 前 1 年
    valid_listing = set(stock_info.index[stock_info["listing_date"] <= (rb_date - pd.DateOffset(years=1))])
    
    # d. 停牌过滤：假设 suspend_df 包含所有停牌记录，排除过去 6 个月内（包括当日）停牌且未复牌的股票
    six_months_ago = rb_date - pd.DateOffset(months=6)
    recent_suspend = suspend_df[(suspend_df["suspend_date"] >= six_months_ago) & (suspend_df["suspend_date"] <= rb_date)]
    suspended_stocks = recent_suspend[
        (recent_suspend["resume_date"].isnull()) | (recent_suspend["resume_date"] > rb_date)
    ]["stock_code"].unique()
    valid_suspension = set(prices.columns) - set(suspended_stocks)
    
    valid = valid_listed & valid_volume & valid_listing & valid_suspension
    return list(valid)

def get_latest_fundamental(stock_code: str, rb_date: pd.Timestamp, fundamental_df: pd.DataFrame) -> pd.Series:
    """
    对于给定股票和再平衡日期，从 fundamental_df 中选取报告期不超过 rb_date 的90天前的最新记录，
    返回一行 Series，包含 total_equity 等基本面数据；若不存在则返回 None。
    """
    latest_report_date = rb_date - pd.Timedelta(days=120)
    df = fundamental_df[fundamental_df["stock_code"] == stock_code]
    df = df[df["report_date"] <= latest_report_date]
    if df.empty:
        return None
    return df.sort_values("report_date").iloc[-1]

def safe_value(val):
    """将 NaN 转换为 None"""
    return None if pd.isna(val) else val


# ---------------------------
# 策略回测实现（使用 from_orders）
# ---------------------------
def backtest_strategy(start_date: str, end_date: str):
    logger.info("Starting backtest strategy from %s to %s", start_date, end_date)
    
    # 加载数据
    refresh_holders()
    # 为保证数据的完整性，加载180天前的数据
    data_start_date = datetime.strptime(start_date, "%Y-%m-%d").date() - timedelta(days=180)
    data_start_date = data_start_date.strftime("%Y-%m-%d")
    prices = get_prices_df(data_start_date, end_date)
    volumes = get_volume_df(data_start_date, end_date)
    mkt_cap = get_mkt_cap_df(data_start_date, end_date)
    stock_info = get_stock_info_df()
    fundamental_df = get_fundamental_df()
    suspend_df = get_suspend_df()
    list_status_of_stocks = get_stock_status_map()

    # 获取再平衡日期
    rb_dates = get_rebalance_dates(prices, start_date, end_date)
    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)  # 目标持仓矩阵
    logger.info("Identified %d rebalance dates.", len(rb_dates))
    
    # 初始化订单矩阵
    init_cash = 1000000
    order_sizes = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
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
        
            # ---------------------------
            # 筛选逻辑
            # ---------------------------

            valid_stocks = filter_universe(prices, volumes, mkt_cap, stock_info, suspend_df, list_status_of_stocks, rb_date)
            logger.info("Valid stocks count on %s: %d", rb_date.strftime("%Y-%m-%d"), len(valid_stocks))
            if not valid_stocks:
                continue
        
            try:
                mkt_cap_on_date = mkt_cap.loc[rb_date, valid_stocks]
            except Exception as e:
                logger.warning("No market cap data on %s; skipping.", rb_date.strftime("%Y-%m-%d"))
                continue
        
            # 市值组筛选
            mkt_cap_sorted = mkt_cap_on_date.sort_values(ascending=True)
            num_threslold = int(len(mkt_cap_sorted) * 0.5)
            cap_selected_stocks = mkt_cap_sorted.index[:num_threslold]
            logger.info("Small cap group count on %s: %d", rb_date.strftime("%Y-%m-%d"), len(cap_selected_stocks))
        
            # VLT筛选
            # 先做行业中性化处理
            # 计算中性化后的盈利能力
            # calculate_volatility(prices, code, start_date, end_date)
            one_year_ago = rb_date - pd.DateOffset(years=1)
            price_series = prices.loc[one_year_ago:rb_date, cap_selected_stocks]
            volatility_series = price_series.apply(lambda col: calculate_column_volatility(col))
            volatility_series = volatility_series.dropna()
            if volatility_series.empty:
                logger.info("No volatility_dict on %s.", rb_date.strftime("%Y-%m-%d"))
                continue
            vlt_sorted = volatility_series.sort_values(ascending=True)
            num_selected = int(len(vlt_sorted) * 0.8)
            selected_stocks = vlt_sorted.index[num_selected:]
            # 确保调仓日的价格无缺失或异常
            if prices.loc[rb_date, selected_stocks].isnull().any():
                logger.warning("Missing price data for selected stocks on %s; skipping.", rb_date.strftime("%Y-%m-%d"))
                invalid_stocks = selected_stocks[prices.loc[rb_date, selected_stocks].isnull()]
                # 从selected_stocks中删除invalid_stocks
                selected_stocks = selected_stocks.drop(invalid_stocks)
            logger.info("Selected stocks count on %s: %d", rb_date.strftime("%Y-%m-%d"), len(selected_stocks))
            # 填充目标持仓（示例：市值加权）
            target_weights.loc[rb_date, selected_stocks] = mkt_cap.loc[rb_date, selected_stocks] / mkt_cap.loc[rb_date, selected_stocks].sum()

            current_prices = prices.loc[rb_date]
            holdings_value = (current_positions * current_prices).sum()
            total_asset = holdings_value + current_cash
            # 计算目标持仓
            target_positions = np.floor((target_weights.loc[rb_date] * total_asset) / current_prices)
            target_positions = target_positions.fillna(0)
            diff_positions = target_positions - current_positions
            diff_value = diff_positions * current_prices
            assumpt_left_balance = -1 * diff_value[diff_value < 0].sum() * (1 - fee_rate - slippage_rate) + current_cash
            assumpt_need_amount = diff_value[diff_value > 0].sum() * (1 + fee_rate + slippage_rate)
            while assumpt_left_balance < assumpt_need_amount:
                scale_factor = assumpt_left_balance / assumpt_need_amount
                target_positions = np.floor(target_positions * scale_factor)
                diff_positions = target_positions - current_positions
                diff_value = diff_positions * current_prices
                assumpt_left_balance = -1 * diff_value[diff_value < 0].sum() * (1 - fee_rate - slippage_rate) + current_cash
                assumpt_need_amount = diff_value[diff_value > 0].sum() * (1 + fee_rate + slippage_rate)
            order_sizes.loc[rb_date] = diff_positions
            current_positions = target_positions
            current_cash = assumpt_left_balance - assumpt_need_amount


    # 输出当日持仓详情到 CSV
    if not os.path.exists(os.path.join(output_dir, format_date(end_date))):
        os.makedirs(os.path.join(output_dir, format_date(end_date)))
    target_weights.loc[rb_dates].to_csv(os.path.join(output_dir, format_date(end_date), output_prefix + "_portfolio.csv"))

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
            fees=fee_rate,                  # 交易费用（动态费用也可以传入Series或DataFrame）
            slippage=slippage_rate               # 滑点
        ),
        cash_sharing=True,  # 同一组内共享现金
        group_by=True,      # 所有资产构成一个组
        init_cash=init_cash
    )
    
    # 输出结果（与原始代码相同）
    logger.info("Order detail:")
    logger.info(pf.orders.records_readable)
    total_value = pf.value().loc[rb_dates[0]:end_date]
    daily_returns = pf.returns().loc[rb_dates[0]:end_date]
    # logger.info(pf.stats())
    total_value.to_csv(os.path.join(output_dir, format_date(end_date), output_prefix + "_total_value.csv"))
    daily_returns.to_csv(os.path.join(output_dir, format_date(end_date), output_prefix + "_daily_returns.csv"))
    logger.info("Latest Value: %s", pf.final_value())
    logger.info("Total Return: %s", pf.total_return())
    # 可视化（可选）
    fig = pf.value(group_by=True).vbt.plot(title="Portfolio Value Curve")
    fig.write_html(os.path.join(output_dir, format_date(end_date), output_prefix + "_value_plot.html"))
    return pf
