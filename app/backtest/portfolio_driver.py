# === portfolio_driver.py ===
# 主流程：构建并回测多个组合，重用数据，控制内存

import pandas as pd
import gc
from app.data_fetcher.bt_data_fetcher import DataFetcher
from app.backtest.rebalance_date_generator import RebalanceDateGenerator
from app.backtest.stock_selector import *
from app.backtest.weight_allocator import *
from app.backtest.backtest_engine import run_backtest
from app.backtest.backtest_engine import BacktestConfig

def build_all_portfolios(start_date: str, end_date: str):
    stock_info = DataFetcher.get_stock_info_df()
    blacklist = []  # 可扩展：从数据库/配置中读取

    # === 初始化共用组件 ===
    fetcher = DataFetcher()
    price = fetcher.fetch_adj_hist("close", start_date, end_date)
    mkt_cap = fetcher.fetch_price("mkt_cap", start_date, end_date)
    fundamentals = fetcher.fetch_fundamentals_on_all(
        start_date,
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

    # === 构造共用数据集 ===
    shared_data = {
        "price": price,
        "mkt_cap": mkt_cap,
        "fundamental": fundamentals
    }

    # === 策略配置 ===
    configs = [
        # factor_type: bm 每季度末；qmj 每年6/12月末
        ("S/L", "bm", (0.0, 0.5), (0.0, 0.3), {"freq": "quarterly", "anchor": "end"}),
        ("S/M", "bm", (0.0, 0.5), (0.3, 0.7), {"freq": "quarterly", "anchor": "end"}),
        ("S/H", "bm", (0.0, 0.5), (0.7, 1.0), {"freq": "quarterly", "anchor": "end"}),
        ("B/L", "bm", (0.5, 1.0), (0.0, 0.3), {"freq": "quarterly", "anchor": "end"}),
        ("B/M", "bm", (0.5, 1.0), (0.3, 0.7), {"freq": "quarterly", "anchor": "end"}),
        ("B/H", "bm", (0.5, 1.0), (0.7, 1.0), {"freq": "quarterly", "anchor": "end"}),
        ("S/L", "qmj", (0.0, 0.5), (0.0, 0.3), {"freq": "custom_months", "custom_months": [6, 12], "anchor": "end"}),
        ("S/M", "qmj", (0.0, 0.5), (0.3, 0.7), {"freq": "custom_months", "custom_months": [6, 12], "anchor": "end"}),
        ("S/H", "qmj", (0.0, 0.5), (0.7, 1.0), {"freq": "custom_months", "custom_months": [6, 12], "anchor": "end"}),
        ("B/L", "qmj", (0.5, 1.0), (0.0, 0.3), {"freq": "custom_months", "custom_months": [6, 12], "anchor": "end"}),
        ("B/M", "qmj", (0.5, 1.0), (0.3, 0.7), {"freq": "custom_months", "custom_months": [6, 12], "anchor": "end"}),
        ("B/H", "qmj", (0.5, 1.0), (0.7, 1.0), {"freq": "custom_months", "custom_months": [6, 12], "anchor": "end"})
    ]

    selector_cls_map = {
        "bm": BMSelector,
        "qmj": QualitySelector
    }

    for name, factor, size_percentile, score_percentile, rb_cfg in configs:
        print(f"\n>>> Building portfolio: {name} ({factor})")

        # === 调仓日生成 ===
        rebalancer = RebalanceDateGenerator(
            freq=rb_cfg["freq"],
            anchor=rb_cfg.get("anchor", "start"),
            custom_months=rb_cfg.get("custom_months")
        )
        rebalance_dates = rebalancer.get_dates_from_range(start_date, end_date)

        # === Selector 和 Allocator ===
        selector_cls = selector_cls_map[factor]
        selector = selector_cls(stock_info, blacklist, size_percentile, score_percentile)
        allocator = EqualWeightAllocator()

        # === 构造调仓矩阵 ===
        weight_df = pd.DataFrame()
        for date in rebalance_dates:
            selected = selector.select(shared_data, date)
            weights = allocator.allocate(selected, shared_data, date)
            if weights:
                weight_df.loc[date] = weights

        # === 回测并保存结果 ===
        cfg = BacktestConfig(
            init_cash=100_000_000,
            buy_fee=0.0,
            sell_fee=0.0,
            slippage=0.0,
            cash_sharing=True
        )
        result = run_backtest(weight_df, price, cfg)

        # 可选：保存结果
        result["actual_weights"].to_csv(f"output/{name}_actual_weights.csv")
        result["nav"].to_csv(f"output/{name}_nav.csv")

        # === 清理中间变量，控制内存 ===
        del weight_df, result
        gc.collect()
