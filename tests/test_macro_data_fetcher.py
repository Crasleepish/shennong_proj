import pytest
import pandas as pd
from unittest.mock import patch
from app.data_fetcher.macro_data_fetcher import MacroDataFetcher
from app.database import get_db
from app.models.macro_models import SocialFinancing
from app import create_app
from app.config import TestConfig

@pytest.fixture
def app():
    app = create_app(config_class=TestConfig)
    yield app

@pytest.fixture(scope="function")
def clean_social_financing(app):
    # 清空测试前的数据
    with get_db() as db:
        db.query(SocialFinancing).delete()
    yield
    # 清空测试后的数据
    with get_db() as db:
        db.query(SocialFinancing).delete()


def test_fetch_social_financing(app, clean_social_financing):
    sample_data = pd.DataFrame({
        "月份": ["202201", "202202"],
        "社会融资规模增量": [50000.0, 60000.0],
    })

    with patch("akshare.macro_china_shrzgm", return_value=sample_data):
        df = MacroDataFetcher.fetch_social_financing()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

        with get_db() as db:
            results = db.query(SocialFinancing).all()
            assert len(results) == 2
            assert results[0].total == 50000.0
            assert results[1].total == 60000.0


def test_fetch_all_runs_without_error(app):
    # 用 patch 跳过 akshare 的实际访问，确保函数调用顺利
    try:
        MacroDataFetcher.fetch_all()
    except Exception as e:
        pytest.fail(f"fetch_all raised an exception: {e}")
