import pytest
import pandas as pd
from datetime import datetime
from app import create_app
from app.config import Config

# 假设 get_ttm_value 函数已经定义
from scripts.optimize_portfolio import build_bl_views, optimize

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
    asset_source_map = {
        'H11004.CSI': 'index',
        'Au99.99.SGE': 'index',
        '008114.OF': 'factor',
        '020602.OF': 'factor',
        '019918.OF': 'factor', 
        '002236.OF': 'factor',
        '019311.OF': 'factor',
        '006712.OF': 'factor',
        '011041.OF': 'factor',
        '110003.OF': 'factor',
        '019702.OF': 'factor',
    }
    code_factors_map = {
        "008114.OF": ["MKT", "SMB", "HML", "QMJ"],
        "020602.OF": ["MKT", "SMB", "HML", "QMJ"],
        "019918.OF": ["MKT", "SMB", "HML", "QMJ"],
        "002236.OF": ["MKT", "SMB", "HML", "QMJ"],
        "019311.OF": ["MKT", "SMB", "HML", "QMJ"],
        "006712.OF": ["MKT", "SMB", "HML", "QMJ"],
        "011041.OF": ["MKT", "SMB", "HML", "QMJ"],
        "110003.OF": ["MKT", "SMB", "HML", "QMJ"],
        "019702.OF": ["MKT", "SMB", "HML", "QMJ"],
        "H11004.CSI": ["10YBOND"], 
        "Au99.99.SGE": ["GOLD"]
    }
    trade_date = '2025-06-13'
    window = 20
    view_codes = ['H11004.CSI', 'Au99.99.SGE', '008114.OF', '020602.OF', '019918.OF', '002236.OF', '019311.OF', '006712.OF', '011041.OF', '110003.OF', '019702.OF']
    portfolio_plan = optimize(asset_source_map, code_factors_map, trade_date, window, view_codes)
    print(portfolio_plan)
    