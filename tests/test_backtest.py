import pytest
from app import create_app
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch
import vectorbt as vbt

# Use the TestConfig from our config module
from app.config import TestConfig

from app.backtest.value_strategy import backtest_strategy
# Create the Flask app using TestConfig
@pytest.fixture
def app():
    app = create_app(config_class=TestConfig)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

# Mock 数据生成函数
def mock_prices():
    dates = pd.date_range(start="2020-01-01", end="2021-01-10", freq="D")
    stocks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]
    prices = pd.DataFrame(
        np.random.uniform(10, 100, size=(len(dates), len(stocks))),
        index=dates,
        columns=stocks
    )
    return prices

def mock_volumes():
    dates = pd.date_range(start="2020-01-01", end="2021-01-10", freq="D")
    stocks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]
    volumes = pd.DataFrame(
        np.random.randint(1000, 10000, size=(len(dates), len(stocks))),
        index=dates,
        columns=stocks
    )
    return volumes

def mock_mkt_cap():
    dates = pd.date_range(start="2020-01-01", end="2021-01-10", freq="D")
    stocks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]
    mkt_cap = pd.DataFrame(
        np.random.uniform(1e6, 1e9, size=(len(dates), len(stocks))),
        index=dates,
        columns=stocks
    )
    return mkt_cap

def mock_stock_info():
    stocks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]
    listing_dates = [datetime(2019, 1, 1), datetime(2018, 1, 1), datetime(2017, 1, 1), datetime(2016, 1, 1), datetime(2019, 1, 1), datetime(2018, 1, 1), datetime(2017, 1, 1), datetime(2016, 1, 1), datetime(2016, 1, 1),
                     datetime(2019, 1, 1), datetime(2018, 1, 1), datetime(2017, 1, 1), datetime(2016, 1, 1), datetime(2019, 1, 1), datetime(2018, 1, 1), datetime(2017, 1, 1), datetime(2016, 1, 1), datetime(2016, 1, 1), 
                     datetime(2016, 1, 1), datetime(2018, 1, 1), datetime(2017, 1, 1) ]
    stock_info = pd.DataFrame({
        "stock_code": stocks,
        "listing_date": listing_dates
    }).set_index("stock_code")
    return stock_info

def mock_fundamental_df():
    stocks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]
    report_dates = [datetime(2019, 12, 31), datetime(2020, 6, 30)]
    fundamental_df = pd.DataFrame({
        "stock_code": np.repeat(stocks, len(report_dates)),
        "report_date": np.tile(report_dates, len(stocks)),
        "total_equity": np.random.uniform(1e6, 1e9, size=len(stocks) * len(report_dates))
    })
    return fundamental_df

def mock_suspend_df():
    stocks = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]
    suspend_dates = [datetime(2019, 3, 1)]
    resume_dates = [datetime(2019, 3, 15)]
    suspend_df = pd.DataFrame({
        "stock_code": np.repeat(stocks, len(suspend_dates)),
        "suspend_date": np.tile(suspend_dates, len(stocks)),
        "resume_date": np.tile(resume_dates, len(stocks))
    })
    return suspend_df

# Mock 数据加载函数
def mock_get_prices_df():
    return mock_prices()

def mock_get_volume_df():
    return mock_volumes()

def mock_get_mkt_cap_df():
    return mock_mkt_cap()

def mock_get_stock_info_df():
    return mock_stock_info()

def mock_get_fundamental_df():
    return mock_fundamental_df()

def mock_get_suspend_df():
    return mock_suspend_df()

# 单元测试
@patch("app.backtest.value_strategy.get_suspend_df", side_effect=mock_get_suspend_df)
@patch("app.backtest.value_strategy.get_fundamental_df", side_effect=mock_get_fundamental_df)
@patch("app.backtest.value_strategy.get_stock_info_df", side_effect=mock_get_stock_info_df)
@patch("app.backtest.value_strategy.get_mkt_cap_df", side_effect=mock_get_mkt_cap_df)
@patch("app.backtest.value_strategy.get_volume_df", side_effect=mock_get_volume_df)
@patch("app.backtest.value_strategy.get_prices_df", side_effect=mock_get_prices_df)
def test_backtest_strategy(
    mock_prices_df,
    mock_volume_df, 
    mock_mkt_cap_df,
    mock_stock_info_df,
    mock_fundamental_df,
    mock_suspend_df,
    app
):
    # 设置回测日期范围
    start_date = "2020-01-06"
    end_date = "2021-01-06"
    
    # 调用 backtest_strategy 函数
    portfolio = backtest_strategy(start_date, end_date)
    
    # 验证输出
    assert isinstance(portfolio, vbt.Portfolio), "返回值应为 vbt.Portfolio 对象"
    assert portfolio.value().iloc[-1] > 0, "组合最终价值应大于 0"
    assert portfolio.returns().sum() != 0, "组合收益率不应为 0"
    
    # 验证 CSV 文件是否生成
    import os
    assert os.path.exists("portfolio_total_value.csv"), "总价值 CSV 文件未生成"
    assert os.path.exists("portfolio_daily_returns.csv"), "每日收益率 CSV 文件未生成"
    
    # 清理生成的 CSV 文件
    os.remove("portfolio_total_value.csv")
    os.remove("portfolio_daily_returns.csv")