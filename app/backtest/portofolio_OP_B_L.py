'''
---

### **1. 分组步骤**
1. **按市值分组**：
   - 使用全市场股票市值的中位数作为分界点，将所有股票分为**小市值（S）**和**大市值（B）**两组。
2. **按OP(ROE)分组**：
   - 在**小市值（S）**组内，按OP的30%和70%分位数将股票分为**低OP（L）**、**中OP（M）**和**高OP（H）**三组。
   - 在**大市值（B）**组内，同样按OP的30%和70%分位数将股票分为**低OP（L）**、**中OP（M）**和**高OP（H）**三组。

---

### **2. 构建组合**
- 经过上述分组后，会形成 **2（市值） × 3（OP） = 6个组合**：
  - 小市值 + 低OP（S/L）
  - 小市值 + 中OP（S/M）
  - 小市值 + 高OP（S/H）
  - 大市值 + 低OP（B/L）
  - 大市值 + 中OP（B/M）
  - 大市值 + 高OP（B/H）

---

### 本组合为大市值 + 低OP（B/L）
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
from .compute_profit import compute_and_cache_profitability, compute_and_cache_industry_avg_profit
from .compute_cashflowquality import compute_and_cache_cash_flow_quality, compute_and_cache_industry_avg_cash_flow_quality
from .compute_leverageratio import compute_and_cache_leverage, compute_and_cache_industry_avg_leverage
from app.utils.data_utils import format_date

logger = logging.getLogger(__name__)
logging.getLogger('numba').setLevel(logging.INFO)

fee_rate = 0.0003
slippage_rate = 0.0001
output_dir = r"./bt_result"
output_prefix = "portofolio_OP_B_L"

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
                    stock_info: pd.DataFrame, suspend_df: pd.DataFrame, rb_date: pd.Timestamp) -> list:
    """
    在给定再平衡日 rb_date 下，过滤股票：
      1. 剔除成交量在当日最低 1% 的股票；
      2. 剔除上市未满 1 年的股票；
      3. 剔除停牌或过去 6 个月内发生过停牌的股票；
    返回符合条件的股票代码列表。
    """
    # a. 成交量过滤：当日成交量
    vol = volumes.loc[rb_date]
    threshold = vol.quantile(0.01)
    valid_volume = set(vol[vol > threshold].index)
    
    # b. 上市时间过滤：上市时间至少在 rb_date 前 1 年
    valid_listing = set(stock_info.index[stock_info["listing_date"] <= (rb_date - pd.DateOffset(years=1))])
    
    # c. 停牌过滤：假设 suspend_df 包含所有停牌记录，排除过去 6 个月内（包括当日）停牌且未复牌的股票
    six_months_ago = rb_date - pd.DateOffset(months=6)
    recent_suspend = suspend_df[(suspend_df["suspend_date"] >= six_months_ago) & (suspend_df["suspend_date"] <= rb_date)]
    suspended_stocks = recent_suspend[
        (recent_suspend["resume_date"].isnull()) | (recent_suspend["resume_date"] > rb_date)
    ]["stock_code"].unique()
    valid_suspension = set(prices.columns) - set(suspended_stocks)
    
    valid = valid_volume & valid_listing & valid_suspension
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
        
            # quality筛选
            # 定义缓存路径
            profitability_cache = f'.cache/profitability_{rb_date}.csv'
            industry_profit_cache = f'.cache/industry_profits_{rb_date}.csv'
            cash_flow_quality_cache = f'.cache/cash_flow_quality_{rb_date}.csv'
            industry_cash_flow_quality_cache = f'.cache/industry_cash_flow_quality_{rb_date}.csv'
            leverage_cache = f'.cache/leverage_{rb_date}.csv'
            industry_leverage_cache = f'.cache/industry_leverage_{rb_date}.csv'

            # 加载或计算盈利能力
            if os.path.exists(profitability_cache):
                logging.info("Loading cached profitability data from %s", profitability_cache)
                profits_df = pd.read_csv(profitability_cache, dtype={'stock_code': str, 'industry': str, 'profitability': 'float64'})
                profits_dict = profits_df.set_index('stock_code')['profitability'].to_dict()
            else:
                profits_df, profits_dict = compute_and_cache_profitability(rb_date, fundamental_df)

            if os.path.exists(industry_profit_cache):
                logging.info("Loading cached industry avg profit data from %s", industry_profit_cache)
                industry_avg_df = pd.read_csv(industry_profit_cache, dtype={'industry': str, 'avg_profitability': 'float64', 'std_profitability': 'float64'})
                industry_avg_profit = industry_avg_df.set_index('industry')['avg_profitability'].to_dict()
                industry_std_profit = industry_avg_df.set_index('industry')['std_profitability'].to_dict()
            else:
                industry_avg_profit, industry_std_profit = compute_and_cache_industry_avg_profit(profits_df, rb_date)

            # 加载或计算现金流质量
            if os.path.exists(cash_flow_quality_cache):
                logging.info("Loading cached cash flow quality data from %s", cash_flow_quality_cache)
                cash_flow_df = pd.read_csv(cash_flow_quality_cache, dtype={'stock_code': str, 'industry': str, 'cash_flow_quality': 'float64'})
                cash_flow_dict = cash_flow_df.set_index('stock_code')['cash_flow_quality'].to_dict()
            else:
                cash_flow_df, cash_flow_dict = compute_and_cache_cash_flow_quality(rb_date, fundamental_df)

            if os.path.exists(industry_cash_flow_quality_cache):
                logging.info("Loading cached industry avg cash flow quality data from %s", industry_cash_flow_quality_cache)
                industry_cf_df = pd.read_csv(industry_cash_flow_quality_cache, dtype={'industry': str, 'avg_cash_flow_quality': 'float64', 'std_cash_flow_quality': 'float64'})
                industry_avg_cash_flow = industry_cf_df.set_index('industry')['avg_cash_flow_quality'].to_dict()
                industry_std_cash_flow = industry_cf_df.set_index('industry')['std_cash_flow_quality'].to_dict()
            else:
                industry_avg_cash_flow, industry_std_cash_flow = compute_and_cache_industry_avg_cash_flow_quality(cash_flow_df, rb_date)

            # 加载或计算资产负债率
            if os.path.exists(leverage_cache):
                logging.info("Loading cached leverage data from %s", leverage_cache)
                leverage_df = pd.read_csv(leverage_cache, dtype={'stock_code': str, 'industry': str, 'leverage_ratio': 'float64'})
                leverage_dict = leverage_df.set_index('stock_code')['leverage_ratio'].to_dict()
            else:
                leverage_df, leverage_dict = compute_and_cache_leverage(rb_date, fundamental_df)

            if os.path.exists(industry_leverage_cache):
                logging.info("Loading cached industry avg leverage data from %s", industry_leverage_cache)
                industry_leverage_df = pd.read_csv(industry_leverage_cache, dtype={'industry': str, 'avg_leverage_ratio': 'float64', 'std_leverage_ratio': 'float64'})
                industry_avg_leverage = industry_leverage_df.set_index('industry')['avg_leverage_ratio'].to_dict()
                industry_std_leverage = industry_leverage_df.set_index('industry')['std_leverage_ratio'].to_dict()
            else:
                industry_avg_leverage, industry_std_leverage = compute_and_cache_industry_avg_leverage(leverage_df, rb_date)

            # 计算综合质量因子（Z-score标准化处理）
            quality_ratios = {}
            for code in cap_selected_stocks:
                if code not in profits_dict or code not in cash_flow_dict or code not in leverage_dict:
                    continue
                industry = stock_info.loc[code, 'industry']
                if industry is None or pd.isna(industry) or industry == '':
                    continue
                if (industry not in industry_avg_profit or industry not in industry_avg_cash_flow or 
                    industry not in industry_avg_leverage):
                    logging.error("Missing industry data for %s, please check", industry)
                    raise Exception("Missing industry data for %s, please check" % industry)

                neutral_profit = (profits_dict[code] - industry_avg_profit[industry]) / industry_std_profit[industry]
                neutral_cash_flow = (cash_flow_dict[code] - industry_avg_cash_flow[industry]) / industry_std_cash_flow[industry]
                neutral_leverage = (leverage_dict[code] - industry_avg_leverage[industry]) / industry_std_leverage[industry]

                quality_value = neutral_profit + neutral_cash_flow - neutral_leverage

                quality_ratios[code] = quality_value

            if not quality_ratios:
                logger.info("No fundamental data for large cap group on %s.", rb_date.strftime("%Y-%m-%d"))

            quality_series = pd.Series(quality_ratios)

            quality_sorted = quality_series.sort_values(ascending=True)
            num_selected = int(len(quality_sorted) * 0.3)
            selected_stocks = quality_sorted.index[:num_selected]
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
    total_value = pf.value().loc[start_date:end_date]
    daily_returns = pf.returns().loc[start_date:end_date]
    # logger.info(pf.stats())
    total_value.to_csv(os.path.join(output_dir, format_date(end_date), output_prefix + "_total_value.csv"))
    daily_returns.to_csv(os.path.join(output_dir, format_date(end_date), output_prefix + "_daily_returns.csv"))
    logger.info("Latest Value: %s", pf.final_value())
    logger.info("Total Return: %s", pf.total_return())
    # 可视化（可选）
    fig = pf.value(group_by=True).vbt.plot(title="Portfolio Value Curve")
    fig.write_html(os.path.join(output_dir, format_date(end_date), output_prefix + "_value_plot.html"))
    return pf
