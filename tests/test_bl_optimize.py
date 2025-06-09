import pytest
import pandas as pd
from datetime import datetime
from app import create_app
from app.config import Config

# 假设 get_ttm_value 函数已经定义
from scripts.optimize_portfolio import build_bl_views

@pytest.fixture
def app():
    app = create_app(config_class=Config)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

def test_build_bl_views(app):
    # 创建测试数据
    codes = ['008114.OF', '020602.OF', '019918.OF', '002236.OF', '006341.OF', '019311.OF']
    trade_date = '2025-06-06'
    bl_views = build_bl_views(codes, trade_date)
    print(bl_views)
    return