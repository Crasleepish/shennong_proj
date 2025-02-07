import pytest
from app import create_app
from app.database import Base, engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.data.fetcher import StockInfoSynchronizer

# Use the TestConfig from our config module
from app.config import TestConfig

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

