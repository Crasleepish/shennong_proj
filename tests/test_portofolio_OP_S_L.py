import pytest
from app import create_app
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch
import vectorbt as vbt

# Use the TestConfig from our config module
from app.config import TestConfig, Config

from app.backtest.portfolio_OP_S_H import backtest_strategy
# Create the Flask app using TestConfig
@pytest.fixture
def app():
    app = create_app(config_class=Config)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app


# 单元测试
def test_backtest_strategy(
    app
):
    # 设置回测日期范围
    start_date = "2008-01-01"
    end_date = "2010-12-31"
    
    # 调用 backtest_strategy 函数
    portfolio = backtest_strategy(start_date, end_date)
    
    # 验证输出
    assert isinstance(portfolio, vbt.Portfolio), "返回值应为 vbt.Portfolio 对象"
    assert portfolio.value().iloc[-1] > 0, "组合最终价值应大于 0"
    assert portfolio.returns().sum() != 0, "组合收益率不应为 0"
    