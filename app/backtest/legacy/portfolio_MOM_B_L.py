'''
---

### **1. 分组步骤**
1. **按市值分组**：
   - 使用全市场股票市值的中位数作为分界点，将所有股票分为**小市值（S）**和**大市值（B）**两组。
2. **按MOM分组**：
   - 在**小市值（S）**组内，按MOM的30%和70%分位数将股票分为**低MOM（L）**、**中MOM（M）**和**高MOM（H）**三组。
   - 在**大市值（B）**组内，同样按MOM的30%和70%分位数将股票分为**低MOM（L）**、**中MOM（M）**和**高MOM（H）**三组。

---

### **2. 构建组合**
- 经过上述分组后，会形成 **2（市值） × 3（MOM） = 6个组合**：
  - 小市值 + 低MOM（S/L）
  - 小市值 + 中MOM（S/M）
  - 小市值 + 高MOM（S/H）
  - 大市值 + 低MOM（B/L）
  - 大市值 + 中MOM（B/M）
  - 大市值 + 高MOM（B/H）

---

### 本组合为大市值 + 低MOM（B/L）
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

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.INFO)

fee_rate = 0.0003
slippage_rate = 0.0001
output_dir = r"./result"
output_prefix = "portfolio_MOM_B_L"

def get_rebalance_dates(prices: pd.DataFrame, start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    根据价格数据的交易日期，返回每年6月和12月最后一个交易日作为再平衡日，
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

def filter_universe(prices: pd.DataFrame, volumes: pd.DataFrame, mkt_cap: pd.DataFrame,
                    stock_info: pd.DataFrame, suspend_df: pd.DataFrame, rb_date: pd.Timestamp) -> list:
    """
    在给定再平衡日 rb_date 下，过滤股票：
      1. 剔除成交量在当日最低 1% 的股票；
      2. 剔除上市未满 1 年的股票；
      3. 剔除停牌或过去 6 个月内发生过停牌且未复牌的股票；
      4. 剔除过去 3 个月内波动率（标准差）最高的 10% 股票。
      
    返回符合条件的股票代码列表。
    """
    # 1. 成交量过滤：使用当日成交量数据
    vol = volumes.loc[rb_date]
    threshold = vol.quantile(0.01)
    valid_volume = set(vol[vol > threshold].index)
    
    # 2. 上市时间过滤：股票上市时间至少在 rb_date 前 1 年
    valid_listing = set(stock_info.index[stock_info["listing_date"] <= (rb_date - pd.DateOffset(years=1))])
    
    # 3. 停牌过滤：排除过去 6 个月内（包括当日）停牌且未复牌的股票
    six_months_ago = rb_date - pd.DateOffset(months=6)
    recent_suspend = suspend_df[(suspend_df["suspend_date"] >= six_months_ago) & (suspend_df["suspend_date"] <= rb_date)]
    suspended_stocks = recent_suspend[
        (recent_suspend["resume_date"].isnull()) | (recent_suspend["resume_date"] > rb_date)
    ]["stock_code"].unique()
    valid_suspension = set(prices.columns) - set(suspended_stocks)
    
    # 4. 波动率过滤：计算过去3个月内的日收益率标准差（波动率），剔除波动率最高的10%
    start_vol = rb_date - pd.DateOffset(months=3)
    try:
        price_3m = prices.loc[start_vol:rb_date]
        returns_3m = price_3m.pct_change(fill_method=None).dropna(how='all')
        vol_series = returns_3m.std().dropna()  # 计算每只股票的日收益率标准差
        vol_threshold = vol_series.quantile(0.9)  # 取90分位数作为阈值
        valid_volatility = set(vol_series[vol_series <= vol_threshold].index)
    except Exception as e:
        logger.error("Error calculating volatility filter: %s", e)
        valid_volatility = set(prices.columns)
    
    # 综合所有条件
    valid = valid_volume & valid_listing & valid_suspension & valid_volatility
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

            valid_stocks = filter_universe(prices, volumes, mkt_cap, stock_info, suspend_df, rb_date)
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
            cap_selected_stocks = mkt_cap_sorted.index[num_threslold:]
            logger.info("Big cap group count on %s: %d", rb_date.strftime("%Y-%m-%d"), len(cap_selected_stocks))
        
            # MOM筛选
            # 计算三个月前的目标日期
            three_months_ago = rb_date - pd.DateOffset(months=3)
            target_year = three_months_ago.year
            target_month = three_months_ago.month

            # 找到目标月份最后一个交易日
            dates = prices.index
            month_dates = dates[(dates.year == target_year) & (dates.month == target_month)]
            if not month_dates.empty:
                start_date = month_dates[-1]
            else:
                raise ValueError(f"未能找到 {target_year}-{target_month} 的交易日")

            rb_price = prices.loc[rb_date, cap_selected_stocks]
            start_price = prices.loc[start_date, cap_selected_stocks]
            returns = (rb_price - start_price) / start_price
            returns.dropna(inplace=True)

            # 选择收益率最高的20%股票
            returns = returns.sort_values(ascending=True)
            num_low = int(len(returns) * 0.2)
            num_high = int(len(returns) * 0.8)
            selected_stocks = returns.index[:num_low]
                
            # selected_stocks = mom_sorted.index[num_high:]
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
    total_value.to_csv(os.path.join(output_dir, output_prefix + "_total_value.csv"))
    daily_returns.to_csv(os.path.join(output_dir, output_prefix + "_daily_returns.csv"))
    logger.info("Latest Value: %s", pf.final_value())
    logger.info("Total Return: %s", pf.total_return())
    # 可视化（可选）
    fig = pf.value(group_by=True).vbt.plot(title="Portfolio Value Curve")
    fig.write_html(os.path.join(output_dir, output_prefix + "_value_plot.html"))
    return pf
