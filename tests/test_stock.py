import pytest
import datetime
from app import create_app
from app.database import Base, engine
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.data.fetcher import StockInfoSynchronizer, StockHistSynchronizer, CompanyActionSynchronizer, FundamentalDataSynchronizer, SuspendDataSynchronizer
from app.data.fetcher import stock_adj_hist_synchronizer
from app.data.cninfo_fetcher import cninfo_stock_share_change_fetcher
from types import SimpleNamespace
from app.database import get_db

# Use the TestConfig from our config module
from app.config import TestConfig
from app.dao.stock_info_dao import StockInfoDao, StockHistUnadjDao, StockHistAdjDao, FundamentalDataDao, SuspendDataDao, StockShareChangeCNInfoDao, CompanyActionDao, FutureTaskDao
from app.data.helper import get_prices_df

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
      - 000001
    使用 SimpleNamespace 模拟股票对象，要求至少有 stock_code 属性。
    """
    return [
        SimpleNamespace(stock_code="000004"),
        SimpleNamespace(stock_code="600655"),
        SimpleNamespace(stock_code="000001"),
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
            VALUES ('000004', '0', '1'), ('600655', '0', '1'), ('000001', '0', '1');
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

    stock_hist_unadj_dao = StockHistUnadjDao._instance
    df0 = stock_hist_unadj_dao.select_dataframe_by_code("600655")
    print(df0)

    stock_hist_unadj_dao.delete_all()

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
        stock_hist_unadj_dao = StockHistUnadjDao._instance
        stock_hist_unadj_dao.delete_all()

def test_AdjSynchronizer(app, init_update_flag_data, monkeypatch, dummy_stock_list):
    def fake_load_stock_info(self):
        return dummy_stock_list
    
    monkeypatch.setattr(StockInfoDao, "load_stock_info", fake_load_stock_info)
    try:
        stock_hist_synchronizer = StockHistSynchronizer()
        stock_hist_synchronizer.sync()
        synchronizer = CompanyActionSynchronizer()
        synchronizer.sync()
        stock_adj_hist_synchronizer.sync()

        stock_hist_unadj_dao = StockHistUnadjDao._instance
        df0 = stock_hist_unadj_dao.select_dataframe_by_code("600655")
        print(df0)

        stock_hist_adj_dao = StockHistAdjDao._instance
        df = stock_hist_adj_dao.select_dataframe_by_code("600655")
        print(df)

        stock_hist_synchronizer.sync()
        synchronizer.sync()
        stock_adj_hist_synchronizer.sync()

        df2 = stock_hist_adj_dao.select_dataframe_by_code("600655")
        print(df2)

        assert df2.equals(df)
    finally:
        stock_hist_unadj_dao = StockHistUnadjDao._instance
        stock_hist_adj_dao = StockHistAdjDao._instance
        company_action_dao = CompanyActionDao._instance
        future_task_dao = FutureTaskDao._instance
        stock_hist_unadj_dao.delete_all()
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

        fundamental_data_dao = FundamentalDataDao._instance
        df = fundamental_data_dao.select_dataframe_by_code("600655")
        print(df)
    finally:
        fundamental_data_dao.delete_all()

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
    
def test_helper(app):
    with get_db() as db:
        try:
            # 执行插入 SQL
            sql1 = """
            INSERT INTO stock_hist_adj (stock_code,"date","open","close",high,low,volume,amount,amplitude,change_percent,"change",turnover_rate,mkt_cap,total_shares) VALUES
	 ('000001','2025-02-28',11.58,11.53,11.68,11.5,94908800,1094298464,1.5490533562822695,-0.7745266781411347,-0.08999999999999986,0.49,223750236823,19405918198),
	 ('000001','2025-02-27',11.53,11.62,11.63,11.46,97731000,1135634220,1.475694444444444,0.8680555555555525,0.09999999999999964,0.5,225496769461,19405918198),
	 ('000001','2025-02-26',11.47,11.52,11.6,11.47,84164600,969576192,1.1333914559720926,0.43591979075849113,0.049999999999998934,0.43,223556177641,19405918198),
	 ('000001','2025-02-25',11.56,11.47,11.58,11.46,91715600,1051977932,1.035375323554782,-1.035375323554782,-0.11999999999999922,0.47,222585881731,19405918198),
	 ('000001','2025-02-24',11.63,11.59,11.69,11.56,94995600,1100999004,1.1168384879725,-0.42955326460481713,-0.05000000000000071,0.49,224914591915,19405918198),
	 ('000001','2025-02-21',11.69,11.64,11.71,11.55,97396900,1133699916,1.3722126929674112,-0.17152658662092257,-0.019999999999999574,0.5,225884887825,19405918198),
	 ('000001','2025-02-20',11.71,11.66,11.76,11.65,78439600,914605736,0.9393680614859046,-0.4269854824936013,-0.05000000000000071,0.4,226273006189,19405918198),
	 ('000001','2025-02-19',11.8,11.71,11.81,11.68,117774900,1379144079,1.1007620660457307,-0.8467400508044001,-0.09999999999999964,0.61,227243302099,19405918198);
            """
            db.execute(text(sql1))
            sql2 = """
            INSERT INTO public.stock_hist_adj (stock_code,"date","open","close",high,low,volume,amount,amplitude,change_percent,"change",turnover_rate,mkt_cap,total_shares) VALUES
	 ('600605','2025-02-28',35.63,35.15,35.81,35.05,2223000,78138450,2.133033960145959,-1.3471793432500812,-0.480000000000004,1.08,7250827379,206282429),
	 ('600605','2025-02-27',35.4,35.63,36.47,35.21,2841500,101242645,3.5522977163800338,0.45108542430223764,0.1600000000000037,1.38,7349842945,206282429),
	 ('600605','2025-02-26',35.89,35.47,36.24,34.89,3698900,131199983,3.7583518930957722,-1.252783964365264,-0.45000000000000284,1.79,7316837757,206282429),
	 ('600605','2025-02-25',36.5,35.92,36.5,35.66,2839200,101984064,2.3178807947019964,-0.8830022075055195,-0.3200000000000003,1.38,7409664850,206282429),
	 ('600605','2025-02-24',36.77,36.24,36.95,35.96,3520600,127586544,2.6756756756756808,-2.0540540540540486,-0.759999999999998,1.71,7475675227,206282429),
	 ('600605','2025-02-21',37.99,37.0,37.99,36.71,3053500,112979500,3.3737480231945205,-2.477596204533468,-0.9399999999999977,1.48,7632449873,206282429);
            """
            db.execute(text(sql2))
            sql3 = """
            INSERT INTO public.stock_hist_adj (stock_code,"date","open","close",high,low,volume,amount,amplitude,change_percent,"change",turnover_rate,mkt_cap,total_shares) VALUES
	 ('600519','2025-02-27',1460.02,1485.56,1489.9,1454.0,4976200,7392443672,2.458887267895432,1.7499880137807244,25.549999999999955,0.4,1866157203768,1256197800),
	 ('600519','2025-02-26',1455.45,1460.01,1464.96,1445.0,2636600,3849462366,1.3727647867950505,0.41334250343878887,6.009999999999991,0.21,1834061349978,1256197800),
	 ('600519','2025-02-25',1470.01,1454.0,1473.39,1452.0,2838700,4127469800,1.4461790178963878,-1.6949840102226357,-25.069999999999936,0.23,1826511601200,1256197800),
	 ('600519','2025-02-24',1488.0,1479.07,1499.52,1474.0,3474400,5138880808,1.714811753717552,-0.6141606359317636,-9.1400000000001,0.28,1858004480046,1256197800),
	 ('600519','2025-02-21',1480.0,1488.21,1496.73,1473.01,3641800,5419763178,1.609226594301223,0.9640434192673023,14.210000000000036,0.29,1869486127938,1256197800),
	 ('600519','2025-02-20',1483.0,1474.0,1491.97,1473.39,2375000,3500750000,1.2461435278336639,-1.1401743796109993,-17.0,0.19,1851635557200,1256197800),
	 ('600519','2025-02-19',1475.05,1491.0,1494.44,1464.9,3239300,4829796300,2.002711864406777,1.0847457627118644,16.0,0.26,1872990919800,1256197800),
	 ('600519','2025-02-18',1470.0,1475.0,1492.99,1462.08,2780000,4100500000,2.100406354901407,0.22967885731371615,3.380000000000109,0.22,1852891755000,1256197800),
	 ('600519','2025-02-17',1481.0,1471.62,1494.98,1467.1,3247100,4778497302,1.890169491525431,-0.22915254237288873,-3.380000000000109,0.26,1848645806436,1256197800),
	 ('600519','2025-02-14',1465.06,1475.0,1477.0,1458.22,2710000,3997250000,1.2818587634636105,0.6784705063273897,9.940000000000055,0.22,1852891755000,1256197800);
            """
            db.execute(text(sql3))
            db.commit()
            df = get_prices_df()
            print(df)
        except Exception as e:
            print(f"Error executing SQL: {e}")
        finally:
            db.rollback()
            stock_hist_adj_dao = StockHistAdjDao._instance
            stock_hist_adj_dao.delete_all()
