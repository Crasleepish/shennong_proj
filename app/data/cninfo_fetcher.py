import requests
import time
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.models.stock_models import StockShareChangeCNInfo
from app.dao.stock_info_dao import StockInfoDao, StockShareChangeCNInfoDao
from app.database import get_db
import logging

logger = logging.getLogger(__name__)

CNINFO_API_KEY = "4GOgirPSOzlyG7485WGbBnBROn0QKj32"
CNINFO_API_SECRET = "iePAXL1M7kvilEwHZTGji6H81fzlyqb8"

def get_token(client_id, client_secret):
    url = 'http://webapi.cninfo.com.cn/api-cloud-platform/oauth2/token'
    post_data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    req = requests.post(url, data=post_data)
    tokendic = json.loads(req.text)
    return tokendic['access_token']


class CninfoStockShareChangeFetcher:

    def __init__(self):
        global CNINFO_API_KEY, CNINFO_API_SECRET
        self.stock_info_dao = StockInfoDao._instance
        self.stock_share_change_data_dao = StockShareChangeCNInfoDao._instance
        self.cninfo_api_key = CNINFO_API_KEY
        self.cninfo_api_secret = CNINFO_API_SECRET

    # 获取股票数据并插入数据库
    def get_stock_share_change_data(self, token, stock_code):
        url = f'http://webapi.cninfo.com.cn/api/stock/p_stock2215?subtype=002&access_token={token}&scode={stock_code}&%40limit=100'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers)
        data = response.json()
        if data['resultcode'] == 200:
            for record in data['records']:
                try:
                    line = StockShareChangeCNInfo(**record)
                    self.stock_share_change_data_dao.insert(line)
                except Exception as e:
                    logger.error("插入数据失败: %s, %s", stock_code, e)

    def fetch_cninfo_data(self, progress_callback=None):
        token = get_token(self.cninfo_api_key, self.cninfo_api_secret)
        
        stock_info_lst = self.stock_info_dao.load_stock_info()
        
        # 控制调用频率：每分钟最多20次
        call_interval = 60 / 20  # 每次调用的间隔，单位秒

        for idx, stock in enumerate(stock_info_lst, start=1):
            try:
                stock_code = stock.stock_code
                df = self.stock_share_change_data_dao.select_dataframe_by_stock_code(stock_code)
                if not df.empty:
                    logger.info(f"数据已存在，跳过: {stock_code}")
                    continue
                self.get_stock_share_change_data(token, stock_code)
                logger.info(f"数据已获取并存入数据库: {stock_code}")
                time.sleep(call_interval)  # 控制调用频率
            except Exception as e:
                logger.error("获取数据失败: %s, %s", stock_code, e)
            finally:
                if progress_callback:
                    progress_callback(idx, len(stock_info_lst))
        if progress_callback:
            progress_callback(len(stock_info_lst), len(stock_info_lst))

cninfo_stock_share_change_fetcher = CninfoStockShareChangeFetcher()