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
from app.data_fetcher.csi_index_data_fetcher import CSIIndexDataFetcher
from app.data_fetcher.factor_data_reader import FactorDataReader
from app.data_fetcher.trade_calender_reader import TradeCalendarReader

# Create the Flask app using TestConfig
@pytest.fixture
def app():
    app = create_app(config_class=Config)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

def test_optimize_portfolio_realtime(app):
    optimize_portfolio_realtime()

def test_additional_factor_df(app):
    today = datetime.datetime.strftime(TradeCalendarReader.get_trade_dates(end=datetime.datetime.strftime(datetime.datetime.today(), "%Y%m%d"))[-1], "%Y-%m-%d")

    injected_factor_df = pd.DataFrame([{
        "MKT": 0.012,
        "SMB": -0.004,
        "HML": 0.006,
        "QMJ": -0.001
    }], index=[today])
    injected_factor_df.index.name = "date"
    injected_factor_df.index = pd.to_datetime(injected_factor_df.index, format='%Y-%m-%d').date

    injected_index_df = pd.DataFrame([{
        "index_code": "H11001.CSI",
        "date": today,
        "close": 105.55
    }])
    injected_index_df["date"] = pd.to_datetime(injected_index_df["date"], format='%Y-%m-%d').dt.date

    factor_reader = FactorDataReader(additional_df=injected_factor_df)
    index_fetcher = CSIIndexDataFetcher(additional_map={"H11001.CSI": injected_index_df})

    df_factors = factor_reader.read_daily_factors()
    df_index = index_fetcher.get_data_by_code_and_date(code="H11001.CSI")
    print(df_factors)
    print(df_index)