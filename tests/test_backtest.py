import pytest
from app import create_app
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch
import vectorbt as vbt
import os

from app.config import TestConfig
from app.backtest.legacy.value_strategy import backtest_strategy

@pytest.fixture
def app():
    app = create_app(config_class=TestConfig)
    yield app

# ===== Mock 数据函数 =====
def mock_prices():
    dates = pd.date_range(start="2020-01-01", end="2021-01-10", freq="D")
    stocks = list("ABCDEFGHIJKLMNOPQRSTU")
    return pd.DataFrame(np.random.uniform(10, 100, (len(dates), len(stocks))), index=dates, columns=stocks)

def mock_volumes():
    dates = pd.date_range(start="2020-01-01", end="2021-01-10", freq="D")
    stocks = list("ABCDEFGHIJKLMNOPQRSTU")
    return pd.DataFrame(np.random.randint(1000, 10000, (len(dates), len(stocks))), index=dates, columns=stocks)

def mock_mkt_cap():
    dates = pd.date_range(start="2020-01-01", end="2021-01-10", freq="D")
    stocks = list("ABCDEFGHIJKLMNOPQRSTU")
    return pd.DataFrame(np.random.uniform(1e6, 1e9, (len(dates), len(stocks))), index=dates, columns=stocks)

def mock_stock_info():
    stocks = list("ABCDEFGHIJKLMNOPQRSTU")
    listing_dates = [datetime(2018, 1, 1)] * len(stocks)
    industries = ["银行"] * 10 + ["半导体"] * 11
    return pd.DataFrame({
        "stock_code": stocks,
        "listing_date": listing_dates,
        "industry": industries
    }).set_index("stock_code")

def mock_fundamental_df():
    stocks = list("ABCDEFGHIJKLMNOPQRSTU")
    report_dates = [datetime(2019, 12, 31), datetime(2020, 6, 30)]
    return pd.DataFrame({
        "stock_code": np.repeat(stocks, len(report_dates)),
        "report_date": np.tile(report_dates, len(stocks)),
        "total_equity": np.random.uniform(1e6, 1e9, len(stocks) * len(report_dates))
    })

def mock_suspend_df():
    stocks = list("ABCDEFGHIJKLMNOPQRSTU")
    suspend_dates = [datetime(2020, 6, 15)]
    return pd.DataFrame({
        "stock_code": np.repeat(stocks[:5], len(suspend_dates)),
        "trade_date": np.tile(suspend_dates, 5),
        "suspend_type": ["S"] * 5
    })

def mock_list_status_df():
    stocks = list("ABCDEFGHIJKLMNOPQRSTU")
    return pd.DataFrame({
        "stock_code": stocks,
        "list_status": ["L"] * 20 + ["D"]  # 最后一个退市
    }).set_index("stock_code")

# ===== Patch 测试函数 =====
@patch("app.backtest.value_strategy.get_stock_status_map", side_effect=mock_list_status_df)
@patch("app.backtest.value_strategy.get_suspend_df", side_effect=mock_suspend_df)
@patch("app.backtest.value_strategy.get_fundamental_df", side_effect=mock_fundamental_df)
@patch("app.backtest.value_strategy.get_stock_info_df", side_effect=mock_stock_info)
@patch("app.backtest.value_strategy.get_mkt_cap_df", side_effect=mock_mkt_cap)
@patch("app.backtest.value_strategy.get_volume_df", side_effect=mock_volumes)
@patch("app.backtest.value_strategy.get_prices_df", side_effect=mock_prices)
@patch("app.backtest.compute_profit.get_stock_info_df", side_effect=mock_stock_info)
def test_backtest_strategy(
    mock_prices_df,
    mock_volume_df,
    mock_mkt_cap_df,
    mock_stock_info_df,
    mock_fundamental_df,
    mock_suspend_df,
    mock_list_status_df,
    app
):
    start_date = "2020-01-06"
    end_date = "2021-01-06"

    portfolio = backtest_strategy(start_date, end_date)

    assert isinstance(portfolio, vbt.Portfolio), "返回值应为 vbt.Portfolio 对象"
    assert portfolio.value().iloc[-1] > 0, "组合最终价值应大于 0"
    assert portfolio.returns().sum() != 0, "组合收益率不应为 0"

    # 验证 CSV 生成
    assert os.path.exists("result/portfolio_total_value.csv"), "总价值 CSV 文件未生成"
    assert os.path.exists("result/portfolio_daily_returns.csv"), "每日收益率 CSV 文件未生成"

    # 清理生成的文件
    try:
        os.remove("result/portfolio_total_value.csv")
        os.remove("result/portfolio_daily_returns.csv")
    except Exception as e:
        print(f"文件删除失败: {e}")
