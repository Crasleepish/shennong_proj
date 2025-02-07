import pytest
from app import create_app
from app.database import Base, engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.data.fetcher import StockInfoSynchronizer

# Use the TestConfig from our config module
from app.config import TestConfig

@pytest.fixture(scope="session")
def test_engine():
    test_engine = create_engine(TestConfig.SQLALCHEMY_DATABASE_URI)
    Base.metadata.create_all(bind=test_engine)
    yield test_engine
    Base.metadata.drop_all(bind=test_engine)

# Create the Flask app using TestConfig
@pytest.fixture
def app():
    app = create_app(config_class=TestConfig)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

# Create a test client for the app
@pytest.fixture
def client(app):
    return app.test_client()

# Example test for the index route
def test_StockInfoSynchronizer(client):
    stock_info_synchronizer = StockInfoSynchronizer()
    stock_info_synchronizer.sync()

