import pytest
import datetime
import pandas as pd
from app import create_app
from app.database import Base, engine
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.data.fetcher import StockInfoSynchronizer, StockHistSynchronizer, AdjFactorSynchronizer, CompanyActionSynchronizer, FundamentalDataSynchronizer, SuspendDataSynchronizer
from app.data.fetcher import stock_adj_hist_synchronizer
from app.data.cninfo_fetcher import cninfo_stock_share_change_fetcher
from types import SimpleNamespace
from app.database import get_db

# Use the TestConfig from our config module
from app.config import TestConfig, Config
from app.dao.stock_info_dao import StockInfoDao, StockHistUnadjDao, StockHistAdjDao, AdjFactorDao, FundamentalDataDao, SuspendDataDao, StockShareChangeCNInfoDao, CompanyActionDao, FutureTaskDao
from app.data.helper import get_prices_df, get_fund_prices_by_code_list, get_fund_fees_by_code_list
from app.data_fetcher import StockDataReader, IndexDataReader
from app.service.portfolio_opt import optimize_portfolio_realtime

# Create the Flask app using TestConfig
@pytest.fixture
def app():
    app = create_app(config_class=TestConfig)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

def test_optimize_portfolio_realtime(app):
    optimize_portfolio_realtime()