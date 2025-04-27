import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest
from app import create_app
from app.config import TestConfig, Config
from app.database import get_db
from app.data.helper import to_adjusted_hist, get_qfq_price_by_code, get_prices_df
from sqlalchemy import text
from app.dao.stock_info_dao import StockHistUnadjDao, AdjFactorDao

@pytest.fixture
def app():
    app = create_app(config_class=TestConfig)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

@pytest.fixture
def setup_test_data(app):
    """准备测试数据并在测试后清理"""
    # 插入股票历史数据
    with get_db() as db:
        # 插入股票历史数据
        db.execute(text("""
            INSERT INTO stock_hist_unadj (stock_code, date, open, high, low, close, pre_close, volume, amount)
            VALUES 
            ('600001', '2023-01-01', 10.0, 11.0, 9.5, 10.5, 10.2, 1000000, 10000000),
            ('600001', '2023-01-02', 10.5, 11.2, 10.1, 11.0, 10.5, 1200000, 12000000),
            ('600001', '2023-01-03', 11.0, 11.5, 10.8, 11.2, 11.0, 1300000, 13000000),
            ('600002', '2023-01-01', 20.0, 21.0, 19.5, 20.5, 20.0, 500000, 10000000),
            ('600002', '2023-01-02', 20.5, 21.2, 20.1, 21.0, 20.5, 520000, 10500000),
            ('600002', '2023-01-03', 21.0, 21.5, 20.8, 21.2, 21.0, 530000, 11000000)
        """))
    
        # 插入复权因子数据
        db.execute(text("""
            INSERT INTO adj_factor (stock_code, date, adj_factor)
            VALUES 
            ('600001', '2023-01-01', 1.0),
            ('600001', '2023-01-02', 1.0),
            ('600001', '2023-01-03', 2.0),  -- 假设有一次分红/配股
            ('600002', '2023-01-01', 1.0),
            ('600002', '2023-01-02', 1.5),  -- 假设有一次分红/配股
            ('600002', '2023-01-03', 1.5)
        """))
        db.commit()
    
        yield
    
        # 清理测试数据
        db.execute(text("DELETE FROM stock_hist_unadj"))
        db.execute(text("DELETE FROM adj_factor"))
        db.commit()

class TestStockHistUnadjDao:
    @staticmethod
    def select_dataframe_by_date_range(stock_code, start_date, end_date):
        with get_db() as db:
            query = f"""
                SELECT * FROM stock_hist_unadj 
                WHERE stock_code = '{stock_code}' 
                AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date
            """
            return pd.read_sql(query, db.bind)

class TestAdjFactorDao:
    @staticmethod
    def get_adj_factors(stock_code, start_date, end_date):
        with get_db() as db:
            query = f"""
                SELECT * FROM adj_factor 
                WHERE stock_code = '{stock_code}' 
                AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date
            """
            return pd.read_sql(query, db.bind)

class StockHistHolder:
    def get_stock_hist(self, start_date, end_date):
        with get_db() as db:
            query = f"""
                SELECT * FROM stock_hist_unadj 
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date
            """
            return pd.read_sql(query, db.bind)

class AdjFactorHolder:
    def get_adj_factor(self, start_date, end_date):
        with get_db() as db:
            query = f"""
                SELECT * FROM adj_factor 
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date
            """
            return pd.read_sql(query, db.bind)

# 设置模拟的DAO单例实例
StockHistUnadjDao = MagicMock()
StockHistUnadjDao._instance = TestStockHistUnadjDao()
AdjFactorDao = MagicMock()
AdjFactorDao._instance = TestAdjFactorDao()

# 设置数据持有者实例
stock_hist_holder = StockHistHolder()
adj_factor_holder = AdjFactorHolder()

def test_to_adjusted_hist(setup_test_data):
    """测试复权转换函数"""
    # 准备测试数据
    unadj_df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'open': [10.0, 10.5, 11.0],
        'high': [11.0, 11.2, 11.5],
        'low': [9.5, 10.1, 10.8],
        'close': [10.5, 11.0, 11.2],
        'volume': [1000000, 1200000, 1300000]
    })
    
    adj_factor_df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'adj_factor': [1.0, 1.0, 2.0]
    })
    
    hist_columns = ['open', 'high', 'low', 'close']
    
    # 调用被测试函数
    result_df = to_adjusted_hist(unadj_df, adj_factor_df, hist_columns, 'adj_factor', 'date')
    
    # 验证结果
    # 最新日期的复权因子是2.0，所以调整比率应该是：1.0/2.0=0.5, 1.0/2.0=0.5, 2.0/2.0=1.0
    expected_open = [10.0 * 0.5, 10.5 * 0.5, 11.0 * 1.0]
    expected_close = [10.5 * 0.5, 11.0 * 0.5, 11.2 * 1.0]
    
    np.testing.assert_almost_equal(result_df['open'].values, expected_open)
    np.testing.assert_almost_equal(result_df['close'].values, expected_close)
    
    # 确保结果DataFrame包含所有原始列
    assert set(unadj_df.columns).issubset(set(result_df.columns))
    # 确保中间计算列已被删除
    assert 'adj_factor' not in result_df.columns
    assert 'adj_ratio' not in result_df.columns

@patch('app.dao.stock_info_dao.StockHistUnadjDao._instance.select_dataframe_by_date_range')
@patch('app.dao.stock_info_dao.AdjFactorDao._instance.get_adj_factors')
def test_get_qfq_price_by_code(mock_get_adj_factors, mock_select_dataframe, setup_test_data):
    """测试获取单个股票前复权价格的函数"""
    # 准备模拟返回数据
    hist_data = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'open': [10.0, 10.5, 11.0],
        'high': [11.0, 11.2, 11.5],
        'low': [9.5, 10.1, 10.8],
        'close': [10.5, 11.0, 11.2],
        'pre_close': [10.2, 10.5, 11.0]
    })
    
    adj_factor_data = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'adj_factor': [1.0, 1.0, 2.0]
    })
    
    mock_select_dataframe.return_value = hist_data
    mock_get_adj_factors.return_value = adj_factor_data
    
    # 调用被测试函数
    result = get_qfq_price_by_code('600001', '2023-01-01', '2023-01-03')
    
    # 验证结果
    # 由于最新日期的复权因子是2.0，所以前两天的价格应该被除以2
    expected_close = [10.5 * 0.5, 11.0 * 0.5, 11.2]
    expected_change = [
        (10.5 * 0.5) - (10.2 * 0.5),
        (11.0 * 0.5) - (10.5 * 0.5),
        11.2 - (11.0 * 0.5)
    ]
    expected_pct_chg = [
        (10.5 * 0.5 - 10.2 * 0.5) / (10.2 * 0.5) * 100,
        (11.0 * 0.5 - 10.5 * 0.5) / (10.5 * 0.5) * 100,
        (11.2 - 11.0 * 0.5) / (11.0 * 0.5) * 100
    ]
    
    np.testing.assert_almost_equal(result['close'].values, expected_close)
    np.testing.assert_almost_equal(result['change'].values, expected_change)
    np.testing.assert_almost_equal(result['pct_chg'].values, expected_pct_chg)
    
    # 确保调用了正确的DAO方法
    mock_select_dataframe.assert_called_once_with('600001', '2023-01-01', '2023-01-03')
    mock_get_adj_factors.assert_called_once_with('600001', '2023-01-01', '2023-01-03')

def test_get_prices_df(setup_test_data):
    """测试获取所有股票前复权价格数据框的函数"""
    # 调用被测试函数
    result_df = get_prices_df('2023-01-01', '2023-01-03')
    
    # 验证结果
    # 确保返回的是一个DataFrame
    assert isinstance(result_df, pd.DataFrame)
    
    # 确保包含了两个股票代码
    assert set(result_df.columns) == {'600001', '600002'}
    
    # 确保日期是索引且类型正确
    assert result_df.index.name == 'date'
    assert isinstance(result_df.index, pd.DatetimeIndex)
    
    # 验证前复权计算是否正确
    # 600001: 最新复权因子2.0，所以前两天的价格需要除以2（乘以0.5）
    # 600002: 最新复权因子1.5，所以第一天的价格需要除以1.5（乘以2/3）
    expected_values = {
        '600001': [10.5 * 0.5, 11.0 * 0.5, 11.2],
        '600002': [20.5 * (1.0/1.5), 21.0, 21.2]
    }
    
    for stock_code in ['600001', '600002']:
        np.testing.assert_almost_equal(
            result_df[stock_code].values, 
            expected_values[stock_code]
        )