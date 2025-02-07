import pytest
from app import create_app
from app.database import Base, engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.data.fetcher import StockInfoSynchronizer, StockHistSynchronizer
from types import SimpleNamespace

# Use the TestConfig from our config module
from app.config import TestConfig
from app.dao.stock_info_dao import StockInfoDao, StockHistUnadjDao

# Create the Flask app using TestConfig
@pytest.fixture
def app():
    app = create_app(config_class=TestConfig)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

# Example test for the index route
def test_StockInfoSynchronizer(app):
    stock_info_synchronizer = StockInfoSynchronizer()
    stock_info_synchronizer.sync()


@pytest.fixture
def dummy_stock_list():
    """
    返回仅包含三个股票的列表：
      - 000004
      - 600601
      - 600519
    使用 SimpleNamespace 模拟股票对象，要求至少有 stock_code 属性。
    """
    return [
        SimpleNamespace(stock_code="000004"),
        SimpleNamespace(stock_code="600601"),
        SimpleNamespace(stock_code="600519"),
    ]


def test_StockHistSynchronizer(app, monkeypatch, dummy_stock_list):
    # 1. 替换 StockInfoDao.load_stock_info，使其返回 dummy_stock_list
    def fake_load_stock_info(self):
        return dummy_stock_list

    monkeypatch.setattr(StockInfoDao, "load_stock_info", fake_load_stock_info)

    # 2. 创建同步器实例并调用 sync 方法
    synchronizer = StockHistSynchronizer()
    synchronizer.sync()
