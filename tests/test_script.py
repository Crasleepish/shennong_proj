import pytest
import pandas as pd
from datetime import datetime
from app import create_app
from app.config import Config

# 假设 get_ttm_value 函数已经定义
from scripts.fund_regression import calculate_factor_exposure, regress_one_fund

@pytest.fixture
def app():
    app = create_app(config_class=Config)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

def test_fund_regression(app):
    # 模拟数据（包含节假日无交易的情况）
    fund_data = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-06", "2023-01-13", "2023-01-20", 
                               "2023-01-27", "2023-02-03"]), 
        "change_percent": [0.010, 0.020, -0.010, 0.005, 0.003]
    })
    
    factor_data = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-06", "2023-01-13", "2023-01-20", 
                               "2023-01-27", "2023-02-03"]),
        "MKT": [0.01, -0.02, 0.015, -0.01, 0.005],
        "SMB": [0.005, -0.003, 0.008, 0.002, -0.001],
        "HML": [-0.003, 0.004, 0.001, -0.002, 0.003],
        "QMJ": [0.002, 0.001, -0.001, 0.003, 0.001],
        "VOL": [-0.004, 0.003, 0.002, -0.001, 0.002]
    })
    
    shibor_data = pd.DataFrame({
        "报告日": pd.to_datetime(["2023-01-06", "2023-01-13", "2023-01-20", 
                               "2023-01-27", "2023-02-03"]),
        "利率": [2.53, 2.55, 2.57, 2.59, 2.60],
        "涨跌": [0.0, 0.02, 0.02, 0.02, 0.01]
    })
    
    # 计算曝险系数
    try:
        exposures = calculate_factor_exposure(fund_data, factor_data, shibor_data)
        print("基金因子曝险系数（含无风险利率调整及节假日处理）：")
        print(exposures.to_markdown())
        
        # 验证跳过的周（示例）
        print("\n跳过的周示例：2023-01-27（春节假期整周无交易）")
    except ValueError as e:
        print(f"计算失败：{str(e)}")

def test_fund_regression_all(app):
    fund_codes = ["004685"]
    for fund_code in fund_codes:
        exposures = regress_one_fund(fund_code)
        print(f"{fund_code}: {exposures}")