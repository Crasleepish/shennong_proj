from app.data.helper import *
import pandas as pd
import numpy as np
import datetime
import vectorbt as vbt
import logging
from numba import njit
from vectorbt.portfolio.enums import Direction, OrderStatus, NoOrder, CallSeqType, SizeType
from vectorbt.portfolio import nb
from app.utils.data_utils import calculate_volatility

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.INFO)

fee_rate = 0.0003
slippage_rate = 0.0001

def get_rebalance_dates(prices: pd.DataFrame, start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    根据价格数据的交易日期，返回每年6月和12月最后一个交易日作为再平衡日，
    仅在指定日期区间内。
    """
    dates = prices.loc[start_date:end_date].index
    rebalance_dates = []
    for year in sorted(dates.year.unique()):
        for month in [6, 12]:
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
      3. 剔除过去 6 个月内发生过停牌的股票；
      4. 剔除已退市的股票（list_status != 'L'）。
    返回符合条件的股票代码列表。
    """
    # a. 剔除退市股票
    valid_listed = set(list_status.index[list_status['list_status'] == 'L'])

    # b. 成交量过滤：当日成交量大于1%分位点
    vol = volumes.loc[rb_date]
    threshold = vol.quantile(0.01)
    valid_volume = set(vol[vol > threshold].index)

    # c. 上市时间过滤：至少上市满1年
    valid_listing = set(stock_info.index[stock_info["listing_date"] <= (rb_date - pd.DateOffset(years=1))])

    # d. 停牌过滤：过去 6 个月内只要出现过停牌（suspend_type == 'S'）就剔除
    six_months_ago = rb_date - pd.DateOffset(months=6)
    recent_suspend = suspend_df[
        (suspend_df["trade_date"] >= six_months_ago) &
        (suspend_df["trade_date"] <= rb_date) &
        (suspend_df["suspend_type"] == "S")
    ]
    suspended_stocks = set(recent_suspend["stock_code"].unique())
    valid_suspension = set(prices.columns) - suspended_stocks

    # 综合过滤
    valid = valid_listed & valid_volume & valid_listing & valid_suspension
    return list(valid)

def get_latest_fundamental(stock_code: str, rb_date: pd.Timestamp, fundamental_df: pd.DataFrame) -> pd.Series:
    """
    对于给定股票和再平衡日期，从 fundamental_df 中选取报告期不超过 rb_date 的90天前的最新记录，
    返回一行 Series，包含 total_equity 等基本面数据；若不存在则返回 None。
    """
    latest_report_date = rb_date - pd.Timedelta(days=90)
    df = fundamental_df[fundamental_df["stock_code"] == stock_code]
    df = df[df["report_date"] <= latest_report_date]
    if df.empty:
        return None
    return df.sort_values("report_date").iloc[-1]

def get_ttm_value(stock_code: str, rb_date: pd.Timestamp, fundamental_df: pd.DataFrame, field: str) -> float:
    """
    1. 循环遍历满足日期条件的财报（从新到旧）
    2. 对每个财报尝试计算 TTM
    3. 若某个财报满足条件则返回，否则继续
    """
    # 获取所有满足日期条件的财报（按日期倒序排列）
    latest_report_date = rb_date - pd.Timedelta(days=120)
    candidate_reports = fundamental_df[
        (fundamental_df["stock_code"] == stock_code) &
        (fundamental_df["report_date"] <= latest_report_date)
    ].sort_values("report_date", ascending=False)
    
    if candidate_reports.empty:
        return None
    
    # 遍历所有候选财报
    for _, report in candidate_reports.iterrows():
        report_date = report["report_date"]
        
        # Case 1: 如果是 Q4 财报，直接返回
        if report_date.month == 12:
            return report[field]
        
        # Case 2: 非 Q4 财报，需要计算 TTM
        current_year = report_date.year
        current_month = report_date.month
        
        # 检查上一年年报是否存在
        last_annual_date = pd.Timestamp(year=current_year-1, month=12, day=31)
        last_annual_report = fundamental_df[
            (fundamental_df["stock_code"] == stock_code) &
            (fundamental_df["report_date"] == last_annual_date)
        ]
        if last_annual_report.empty:
            continue  # 缺失年报，跳过此财报
        
        # 检查去年同期累计值（可能需向前查找多个季度）
        same_period_last_year = pd.Timestamp(year=current_year-1, month=report_date.month + 1, day=1)
        same_period_reports = fundamental_df[
            (fundamental_df["stock_code"] == stock_code) &
            (fundamental_df["report_date"] < same_period_last_year) &
            (fundamental_df["report_date"].dt.month == current_month)
        ]
        
        if same_period_reports.empty:
            continue  # 缺失同期数据，跳过此财报
        
        # 取最近一期去年同期累计值
        last_period_value = same_period_reports.iloc[0][field]
        
        # 计算 TTM
        ttm_value = report[field] + last_annual_report.iloc[0][field] - last_period_value
        return ttm_value
    
    # 所有候选财报均不满足条件
    return None

def safe_value(val):
    """将 NaN 转换为 None"""
    return None if pd.isna(val) else val


# ---------------------------
# 策略回测实现（使用 from_orders）
# ---------------------------
def backtest_strategy(start_date: str, end_date: str):
    logger.info("Starting backtest strategy from %s to %s", start_date, end_date)
    
    # 加载数据
    prices = get_prices_df().loc[start_date:end_date]
    volumes = get_volume_df().loc[start_date:end_date]
    mkt_cap = get_mkt_cap_df().loc[start_date:end_date]
    stock_info = get_stock_info_df()
    fundamental_df = get_fundamental_df()
    suspend_df = get_suspend_df()

    # 获取再平衡日期
    rb_dates = get_rebalance_dates(prices, start_date, end_date)
    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)  # 目标持仓矩阵
    logger.info("Identified %d rebalance dates.", len(rb_dates))
    
    # 初始化订单矩阵（不再使用权重矩阵）
    # orders = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    init_cash = 1000000
    
    # 维护动态持仓状态（用于计算订单）
    order_sizes = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
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

            valid_stocks = filter_universe(prices, volumes, mkt_cap, stock_info, suspend_df, rb_date)
            logger.info("Valid stocks count on %s: %d", rb_date.strftime("%Y-%m-%d"), len(valid_stocks))
            if not valid_stocks:
                continue
        
            try:
                mkt_cap_on_date = mkt_cap.loc[rb_date, valid_stocks]
            except Exception as e:
                logger.warning("No market cap data on %s; skipping.", rb_date.strftime("%Y-%m-%d"))
                continue
        
            # 大市值组筛选
            mkt_cap_sorted = mkt_cap_on_date.sort_values(ascending=False)
            num_large = int(len(mkt_cap_sorted) * 0.5)
            large_cap_stocks = mkt_cap_sorted.index[:num_large]
            logger.info("Large cap group count on %s: %d", rb_date.strftime("%Y-%m-%d"), len(large_cap_stocks))
        
            # B/M筛选
            bm_ratios = {}
            for code in large_cap_stocks:
                current_cap = mkt_cap.loc[rb_date, code]
                fundamental = get_latest_fundamental(code, rb_date, fundamental_df)
                if fundamental is not None and current_cap and current_cap != 0:
                    bm = fundamental["total_equity"] / current_cap
                    bm_ratios[code] = bm
            if not bm_ratios:
                logger.info("No fundamental data for large cap group on %s.", rb_date.strftime("%Y-%m-%d"))
                continue
            bm_series = pd.Series(bm_ratios)
            bm_sorted = bm_series.sort_values(ascending=False)
            num_selected = int(len(bm_sorted) * 0.3)
            selected_stocks = bm_sorted.index[:num_selected]
            # 确保调仓日的价格无缺失或异常
            if prices.loc[rb_date, selected_stocks].isnull().any():
                logger.warning("Missing price data for selected stocks on %s; skipping.", rb_date.strftime("%Y-%m-%d"))
                return
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
    target_weights.loc[rb_dates].to_csv(f"result/portfolio.csv")

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
    total_value = pf.value()
    daily_returns = pf.returns()
    # logger.info(pf.stats())
    total_value.to_csv("result/portfolio_total_value.csv")
    daily_returns.to_csv("result/portfolio_daily_returns.csv")
    logger.info("Latest Value: %s", pf.final_value())
    logger.info("Total Return: %s", pf.total_return())
    # 可视化（可选）
    fig = pf.value(group_by=True).vbt.plot(title="Portfolio Value Curve")
    fig.write_html("result/portfolio_value.html")
    return pf
