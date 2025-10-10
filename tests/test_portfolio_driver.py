# tests/test_fund_beta_dao.py

import pytest
import pandas as pd
from app import create_app
from app.config import TestConfig, Config
from app.backtest.portfolio_driver import build_all_portfolios



@pytest.fixture
def app():
    app = create_app(config_class=Config)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app


def test_build_all_portfolios(app):
    build_all_portfolios("2025-09-24", "2025-09-29", "realtime")