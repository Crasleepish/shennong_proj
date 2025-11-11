import pytest
import datetime
import pandas as pd
from app import create_app
from app.database import Base, engine
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.data.fetcher import StockInfoSynchronizer, StockHistSynchronizer, AdjFactorSynchronizer, CompanyActionSynchronizer, FundamentalDataSynchronizer, SuspendDataSynchronizer
from app.data.cninfo_fetcher import cninfo_stock_share_change_fetcher
from types import SimpleNamespace
from app.database import get_db

# Use the TestConfig from our config module
from app.config import TestConfig, Config
from app.dao.stock_info_dao import StockInfoDao, StockHistUnadjDao, StockHistAdjDao, AdjFactorDao, FundamentalDataDao, SuspendDataDao, StockShareChangeCNInfoDao, CompanyActionDao, FutureTaskDao
from app.data.helper import get_prices_df, get_fund_prices_by_code_list, get_fund_fees_by_code_list
from app.data_fetcher import StockDataReader, IndexDataReader, EtfDataReader

# Create the Flask app using TestConfig
@pytest.fixture
def app():
    app = create_app(config_class=TestConfig)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

# Example test for the index route
def test_StockInfoSynchronizer(app):
    try:
        stock_info_synchronizer = StockInfoSynchronizer()
        stock_info_synchronizer.sync()
    finally:
        StockInfoDao.delete_all()

@pytest.fixture
def dummy_stock_list():
    """
    返回仅包含三个股票的列表：
      - 000004
      - 600601
      - 000001
    使用 SimpleNamespace 模拟股票对象，要求至少有 stock_code 属性。
    """
    return [
        SimpleNamespace(stock_code="000004.SZ", market="主板"),
        SimpleNamespace(stock_code="600655.SH", market="主板"),
        SimpleNamespace(stock_code="000007.SZ", market="主板"),
    ]

@pytest.fixture
def init_update_flag_data(app):
    """
    测试开始前自动执行，将 update_flag 表中插入 mock 数据
    """
    # 获取数据库会话
    with get_db() as db:
        try:
            # 执行插入 SQL
            sql = """
            INSERT INTO update_flag (stock_code, action_update_flag, fundamental_update_flag)
            VALUES ('000004', '0', '1'), ('600655', '0', '1'), ('301600', '0', '1');
            """
            db.execute(text(sql))
            db.commit()
        except Exception as e:
            print(f"Error executing SQL: {e}")


def test_StockHistSynchronizer(app, monkeypatch, dummy_stock_list):
    # 1. 替换 StockInfoDao.load_stock_info，使其返回 dummy_stock_list
    def fake_load_stock_info(self):
        return dummy_stock_list

    monkeypatch.setattr(StockInfoDao, "load_stock_info", fake_load_stock_info)

    # 2. 创建同步器实例并调用 sync 方法
    synchronizer = StockHistSynchronizer()
    synchronizer.sync()

    df0 = StockHistUnadjDao.select_dataframe_by_code("600655")
    print(df0)

    StockHistUnadjDao.delete_all()

def test_CompanyActionSynchronizer(app, init_update_flag_data, monkeypatch, dummy_stock_list):
    def fake_load_stock_info(self):
        return dummy_stock_list
    
    try:
        monkeypatch.setattr(StockInfoDao, "load_stock_info", fake_load_stock_info)

        stock_hist_synchronizer = StockHistSynchronizer()
        stock_hist_synchronizer.sync()
        synchronizer = CompanyActionSynchronizer()
        synchronizer.sync()
    finally:
        company_action_dao = CompanyActionDao._instance
        company_action_dao.delete_all()
        StockHistUnadjDao.delete_all()
        
def test_AdjFactorSynchronizer(app, monkeypatch, dummy_stock_list):
    def fake_load_stock_info(self):
        return dummy_stock_list
    
    try:
        monkeypatch.setattr(StockInfoDao, "load_stock_info", fake_load_stock_info)

        adj_factor_synchronizer = AdjFactorSynchronizer()
        adj_factor_synchronizer.sync()
    finally:
        adj_factor_dao = AdjFactorDao._instance
        adj_factor_dao.delete_all()

def test_AdjSynchronizer(app, init_update_flag_data, monkeypatch, dummy_stock_list):
    def fake_load_stock_info(self):
        return dummy_stock_list
    
    monkeypatch.setattr(StockInfoDao, "load_stock_info", fake_load_stock_info)
    try:
        stock_hist_synchronizer = StockHistSynchronizer()
        stock_hist_synchronizer.sync()
        synchronizer = CompanyActionSynchronizer()
        synchronizer.sync()

        df0 = StockHistUnadjDao.select_dataframe_by_code("600655")
        print(df0)

        stock_hist_adj_dao = StockHistAdjDao._instance
        df = stock_hist_adj_dao.select_dataframe_by_code("600655")
        print(df)

        stock_hist_synchronizer.sync()
        synchronizer.sync()

        df2 = stock_hist_adj_dao.select_dataframe_by_code("600655")
        print(df2)

        assert df2.equals(df)
    finally:
        stock_hist_adj_dao = StockHistAdjDao._instance
        company_action_dao = CompanyActionDao._instance
        future_task_dao = FutureTaskDao._instance
        StockHistUnadjDao.delete_all()
        stock_hist_adj_dao.delete_all()
        company_action_dao.delete_all()
        future_task_dao.delete_all()


def test_FundamentalDataSynchronizer(app, init_update_flag_data, monkeypatch, dummy_stock_list):
    def fake_load_stock_info(self):
        return dummy_stock_list
    
    monkeypatch.setattr(StockInfoDao, "load_stock_info", fake_load_stock_info)
    try:
        fundatmental_data_synchronizer = FundamentalDataSynchronizer()
        fundatmental_data_synchronizer.sync()
        fundatmental_data_synchronizer.sync()

        df = FundamentalDataDao.select_dataframe_all()
        print(df)
    finally:
        FundamentalDataDao.delete_all()

def test_SuspendDataSynchronizer(app):
    suspend_data_synchronizer = SuspendDataSynchronizer()
    suspend_data_synchronizer.sync_all("20120222")
    suspend_data_synchronizer.sync_today()
    suspend_data_dao = SuspendDataDao._instance
    df1 = suspend_data_dao.select_dataframe_all()
    print(df1)
    df2 = suspend_data_dao.get_suspended_stocks_by_date(datetime.date(2024, 10, 22))
    print(df2)

def test_CninfoStockShareChangeFetcher(app, monkeypatch, dummy_stock_list):
    def fake_load_stock_info(self):
        return dummy_stock_list
    
    monkeypatch.setattr(StockInfoDao, "load_stock_info", fake_load_stock_info)

    cninfo_stock_share_change_fetcher.fetch_cninfo_data()
    stock_share_change_cninfo_dao = StockShareChangeCNInfoDao._instance
    df = stock_share_change_cninfo_dao.select_dataframe_all()
    print(df)
    stock_share_change_cninfo_dao.delete_all()
    
def test_update_industry(app):
	StockInfoDao.update_all_industry()
    
def test_get_fund_prices_by_code_list(app):
    code_list = ['008115', '019919', '018733']
    df = get_fund_prices_by_code_list(code_list, '2024-03-01', '2025-01-05')
    print(df)
    
def test_get_fund_fees_by_code_list(app):
    code_list = ['008115', '019919', '018733']
    fees_dict = get_fund_fees_by_code_list(code_list)
    print(fees_dict)

def test_stock_data_fetcher_fetch_latest_close_prices(app):
    fetcher = StockDataReader()

    # 第一次调用，应触发实际查询
    df = fetcher.fetch_latest_close_prices_from_cache(exchange_filter=['SSE', 'SZSE'], list_status_filter=['L'])
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'stock_code' in df.columns
    assert 'close' in df.columns

    # 第三次调用，使用不同 exchange_filter，应重新查询
    df3 = fetcher.fetch_latest_close_prices_from_cache(exchange_filter=['SSE'], list_status_filter=['L'])
    assert isinstance(df3, pd.DataFrame)
    assert not df3.empty
    assert df3 is not df  # 确保是新数据，不是缓存命中

    # 验证 filter key 缓存是否按条件记录
    assert fetcher._last_cache_filter_key == (('SSE',), ('L',))

def test_fetch_realtime_prices(app):
    fetcher = StockDataReader()
    df_rt = fetcher.fetch_realtime_prices()
    print(df_rt)


from app.data_fetcher.xueqiu_quote import stock_individual_spot_xq_safe, stock_zh_a_xq_list
def test_stock_individual_spot_xq_safe(app):
    df = stock_individual_spot_xq_safe("CSI000985")
    print(df)

def test_stock_zh_a_xq_list(app):
    df = stock_zh_a_xq_list()
    print(df)


def test_index_data_fetcher_fetch_latest_close_prices(app):
    fetcher = IndexDataReader()
    df1 = fetcher.fetch_latest_close_prices_from_cache('000985.CSI')
    assert isinstance(df1, pd.DataFrame)
    assert not df1.empty
    assert 'stock_code' in df1.columns
    assert 'close' in df1.columns

    df2 = fetcher.fetch_latest_close_prices_from_cache('000985.CSI')
    assert df2 is fetcher._last_df_cache  # 确保是同一个对象（未重新查询）

    # 第三次调用，使用不同参数，应重新查询
    df3 = fetcher.fetch_latest_close_prices_from_cache('000300.SH')
    assert isinstance(df3, pd.DataFrame)
    assert not df3.empty
    assert df3 is not df2  # 确保是新数据，不是缓存命中

def test_index_fetch_realtime_prices(app):
    fetcher = IndexDataReader()
    df_rt = fetcher.fetch_realtime_prices("000985.CSI")
    print(df_rt)

def test_etf_fetch_realtime_prices(app):
    fetcher = EtfDataReader()
    df_rt = fetcher.fetch_realtime_prices("511010.SH")
    print(df_rt)