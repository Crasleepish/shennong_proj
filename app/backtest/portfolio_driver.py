# === portfolio_driver.py ===
# 主流程：构建并回测多个组合，重用数据，控制内存

import pandas as pd
import gc
import os
import logging
from app.data_fetcher.bt_data_fetcher import DataFetcher
from app.data_fetcher.calender_fetcher import CalendarFetcher
from app.backtest.rebalance_date_generator import RebalanceDateGenerator
from app.backtest.stock_selector import *
from app.backtest.weight_allocator import *
from app.backtest.backtest_engine import run_backtest
from app.backtest.backtest_engine import BacktestConfig

logger = logging.getLogger(__name__)

def build_all_portfolios(start_date: str, end_date: str):
    # === 历史数据归档目录 ===
    output_path = "./bt_result"
    os.makedirs(output_path, exist_ok=True)

    stock_info = DataFetcher.get_stock_info_df()
    blacklist = []  # 可扩展：从数据库/配置中读取

    # 为保证start_date有持仓，从start_date开始建仓
    prev_start_date = CalendarFetcher().get_prev_trade_date(start_date.replace("-", ""), format="%Y-%m-%d")

    # === 初始化共用组件 ===
    fetcher = DataFetcher()
    price = fetcher.fetch_adj_hist("close", prev_start_date, end_date)
    mkt_cap = fetcher.fetch_price("mkt_cap", prev_start_date, end_date)
    amount = fetcher.fetch_price("amount", prev_start_date, end_date)
    fundamentals = fetcher.fetch_fundamentals_on_all(
        prev_start_date,
        end_date,
        fields=[
            "total_equity",
            "operating_profit_ttm",
            "total_assets",
            "total_liabilities",
            "net_profit",
            "net_cash_from_operating"
        ]
    )
    price.to_csv(os.path.join(output_path, "price.csv"))
    mkt_cap.to_csv(os.path.join(output_path, "mkt_cap.csv"))
    amount.to_csv(os.path.join(output_path, "amount.csv"))
    fundamentals.to_csv(os.path.join(output_path, "fundamentals.csv"))

    # === 构造共用数据集 ===
    shared_data = {
        "price": price,
        "mkt_cap": mkt_cap,
        "fundamental": fundamentals
    }

    # === 策略配置 ===
    configs = [
        # factor_type: bm 每季度末；qmj 每年6/12月末
        ("SL", "bm", (0.0, 0.5), (0.0, 0.3), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"}),
        ("SM", "bm", (0.0, 0.5), (0.3, 0.7), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"}),
        ("SH", "bm", (0.0, 0.5), (0.7, 1.0), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"}),
        ("BL", "bm", (0.5, 1.0), (0.0, 0.3), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"}),
        ("BM", "bm", (0.5, 1.0), (0.3, 0.7), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"}),
        ("BH", "bm", (0.5, 1.0), (0.7, 1.0), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"}),
        ("SL", "qmj", (0.0, 0.5), (0.0, 0.3), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"}),
        ("SM", "qmj", (0.0, 0.5), (0.3, 0.7), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"}),
        ("SH", "qmj", (0.0, 0.5), (0.7, 1.0), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"}),
        ("BL", "qmj", (0.5, 1.0), (0.0, 0.3), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"}),
        ("BM", "qmj", (0.5, 1.0), (0.3, 0.7), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"}),
        ("BH", "qmj", (0.5, 1.0), (0.7, 1.0), {"freq": "custom_months", "custom_months": [3, 6, 9, 12], "anchor": "end"})
    ]

    for name, factor, size_percentile, score_percentile, rb_cfg in configs:
        logging.info(f"\n>>> Building portfolio: {name} ({factor})")

        # === 调仓日生成 ===
        rebalancer = RebalanceDateGenerator(
            freq=rb_cfg["freq"],
            anchor=rb_cfg.get("anchor", "start"),
            custom_months=rb_cfg.get("custom_months")
        )
        rebalance_dates = rebalancer.get_dates_from_range(start_date, end_date)

        allocator = MktCapWeightAllocator()

        # 确定建仓日权重，建仓日为start_date的前一天
        # 若历史数据中存在上一个调仓日的权重数据，则以上一个调仓日的权重进行建仓 ===
        prev_date = rebalancer.get_prev_balance_date(start_date)
        weight_path = os.path.join(output_path, f"{factor}_{name}_weights.csv")
        hist_not_exists_flag = False
        use_hist_weight_flag = False
        if os.path.exists(weight_path):
            hist_weight = pd.read_csv(weight_path, index_col=0, parse_dates=True)
            if prev_date in hist_weight.index:
                use_hist_weight_flag = True
                init_weight = hist_weight.loc[prev_date]
                logging.info(f"{weight_path} 中存在上一个调仓日 {prev_date} 的权重数据，使用历史数据建仓")
            else:
                hist_not_exists_flag = True
        else:
            hist_not_exists_flag = True
            
        # 若历史数据不存在，或历史数据中没有上一个调仓日的权重数据，则直接将prev_start_date加入再平衡日
        if hist_not_exists_flag:
            use_hist_weight_flag = False
            init_date = prev_start_date
            rebalance_dates = pd.DatetimeIndex([pd.to_datetime(init_date)] + rebalance_dates.tolist())
            logging.info(f"{weight_path} 中不存在上一个调仓日 {prev_date} 的权重数据，将{init_date}加入再平衡日")
            logging.warning(f"将{init_date}加入再平衡日，这会导致{init_date}日的收益数据为0")

        # === 构造调仓矩阵 ===
        records = []
        all_stocks = set(stock_info.index)
        for date in rebalance_dates:
            # === Selector 和 Allocator ===
            basic_selector = BasicSelector(stock_info, blacklist, shared_data.get("price"), date)
            basic_amount_selector = AmountSelector(amount, date, 0.01, parents=[basic_selector])
            size_selector = MktCapPercentileSelector(shared_data.get("mkt_cap"), date, size_percentile)
            if factor == "bm":
                selector = BMScoreSelector(fundamental_df=shared_data.get("fundamental"), mkt_cap_df=shared_data.get("mkt_cap"), asof_date=date, bm_percentile=score_percentile, parents=[basic_amount_selector, size_selector])
            elif factor == "qmj":
                selector = QualityScoreSelector(stock_info=stock_info, fundamental_df=shared_data.get("fundamental"), asof_date=date, score_percentile=score_percentile, parents=[basic_amount_selector, size_selector])
            else:
                raise ValueError("Unknown factor")
            
            selected = selector.select(all_stocks)
            weights = allocator.allocate(selected, shared_data, date)
            for stock, w in weights.items():
                records.append((date, stock, w))

        if records:
            weight_df = pd.DataFrame(records, columns=["date", "stock_code", "weight"])
            weight_df = weight_df.pivot(index="date", columns="stock_code", values="weight").sort_index()
            # === 保存权重数据 ===
            weight_path = os.path.join(output_path, f"{factor}_{name}_weights.csv")
            if os.path.exists(weight_path):
                old_weight = pd.read_csv(weight_path, index_col=0, parse_dates=True)
                weight_df_full = pd.concat([old_weight, weight_df])
                weight_df_full = weight_df_full[~weight_df_full.index.duplicated(keep='last')].sort_index()
                weight_df_full.to_csv(weight_path)
                del weight_df_full

            if use_hist_weight_flag:
                weight_df = pd.concat([pd.DataFrame([init_weight], index=[pd.to_datetime(prev_start_date)]), weight_df])
            
        del records

        # === 回测并保存结果 ===
        cfg = BacktestConfig(
            init_cash=100_000_000,
            buy_fee=0.0,
            sell_fee=0.0,
            slippage=0.0,
            cash_sharing=True
        )
        result = run_backtest(weight_df, shared_data.get("price"), cfg)

        # === 保存日常收益 ===
        daily_return = result["returns"][start_date:]
        daily_return.name = "value"
        return_path = os.path.join(output_path, f"{factor}_{name}_daily_returns.csv")
        daily_return.to_csv(return_path, index_label="date")

        # 可视化（可选）
        fig = result["nav"].vbt.plot(title="Portfolio Value Curve")
        fig.write_html(os.path.join(output_path, f"{factor}_{name}_{start_date}_{end_date}_value_plot.html"))

        # === 清理中间变量，控制内存 ===
        del weight_df, result
        gc.collect()
