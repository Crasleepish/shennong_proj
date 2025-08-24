import pytest
import pandas as pd
from datetime import datetime
from app import create_app
from app.config import Config

# 假设 get_ttm_value 函数已经定义
from scripts.optimize_portfolio import build_bl_views, optimize
from app.service.portfolio_assets_service import get_portfolio_assets

@pytest.fixture
def app():
    app = create_app(config_class=Config)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

def test_build_bl_views(app):
    # 创建测试数据
    codes = ['008114.OF', '020602.OF', '019918.OF', '002236.OF', '019311.OF', '006712.OF', '011041.OF', '110003.OF', '019702.OF']
    trade_date = '2025-06-06'
    bl_views = build_bl_views(codes, trade_date)
    print(bl_views)
    return

def test_optimize(app):
    # 创建测试数据
    portfolio_id = 1
    asset_info = get_portfolio_assets(portfolio_id)
    asset_source_map = asset_info["asset_source_map"]
    code_factors_map = asset_info["code_factors_map"]
    view_codes = asset_info["view_codes"]
    trade_date = '2025-06-13'
    window = 20
    portfolio_plan = optimize(asset_source_map, code_factors_map, trade_date, window, view_codes)
    print(portfolio_plan)
    