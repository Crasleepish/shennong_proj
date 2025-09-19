import requests
import time
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.models.stock_models import StockShareChangeCNInfo
from app.dao.stock_info_dao import StockInfoDao, StockShareChangeCNInfoDao
from app.database import get_db
import logging
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes
import base64
import time

logger = logging.getLogger(__name__)

CNINFO_API_KEY = "4GOgirPSOzlyG7485WGbBnBROn0QKj32"
CNINFO_API_SECRET = "iePAXL1M7kvilEwHZTGji6H81fzlyqb8"

def get_res_code1():
    key = b'1234567887654321'  # 16 bytes key
    iv = b'1234567887654321'   # 16 bytes IV
    timestamp = str(int(time.time()))  # current timestamp in seconds

    cipher = AES.new(key, AES.MODE_CBC, iv)
    encrypted = cipher.encrypt(pad(timestamp.encode('utf-8'), AES.block_size))

    return base64.b64encode(encrypted).decode('utf-8')

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
        self.stock_share_change_data_dao = StockShareChangeCNInfoDao._instance
        self.cninfo_api_key = CNINFO_API_KEY
        self.cninfo_api_secret = CNINFO_API_SECRET
        self.token = None
        self.token_time = None

    def get_token_if_needed(self):
        """如果token过期，重新获取token"""
        if not self.token or (datetime.now() - self.token_time) > timedelta(minutes=5):
            self.token = get_token(self.cninfo_api_key, self.cninfo_api_secret)
            self.token_time = datetime.now()
            logger.info("Token refreshed successfully.")
        return self.token
    
    def get_cninfo_guest(self, token, stock_codes):
        url = f'http://webapi.cninfo.com.cn/api/stock/p_stock2215'
        headers = {
            "Accept": "*/*",
            "Accept-Enckey": get_res_code1(),
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Content-Length": "0",
            "Host": "webapi.cninfo.com.cn",
            "Origin": "https://webapi.cninfo.com.cn",
            "Pragma": "no-cache",
            "Proxy-Connection": "keep-alive",
            "Referer": "https://webapi.cninfo.com.cn/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/93.0.4577.63 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            }
        params = {
            "scode": ",".join(stock_codes)
        }
        response = requests.post(url, params=params, headers=headers)
        data = response.json()
        if data['resultcode'] == 200:
            for record in data['records']:
                try:
                    line = StockShareChangeCNInfo(**record)
                    if line.SECCODE is None:
                        raise Exception("error fetching data")
                    self.stock_share_change_data_dao.insert(line)
                except Exception as e:
                    logger.error("插入数据失败: %s, %s", stock_codes, e)
        else:
            raise Exception("error fetching data")

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
                    if line.SECCODE is None:
                        raise Exception("error fetching data")
                    return self.stock_share_change_data_dao.insert(line)
                except Exception as e:
                    logger.error("插入数据失败: %s, %s", stock_code, e)
        else:
            return None

    def fetch_cninfo_data(self, progress_callback=None):
        # 获取token（如果需要的话）
        token = self.get_token_if_needed()
        
        stock_info_lst = StockInfoDao.load_stock_info()
        
        # 控制调用频率：每分钟最多5次
        call_interval = 60  # 每次调用的间隔，单位秒
        batch_size = 50  # 每批次最多50个stock_code
        stock_codes_batch = []

        for idx, stock in enumerate(stock_info_lst, start=1):
            stock_code = stock.stock_code
            try:
                # Check if data already exists for the current stock_code
                df = self.stock_share_change_data_dao.select_dataframe_by_stock_code(stock_code)
                if not df.empty:
                    logger.info(f"数据已存在，跳过: {stock_code}")
                else:
                    # No data for this stock_code, add to batch
                    stock_codes_batch.append(stock_code)
                # Once batch size is reached or it's the last stock_code
                if len(stock_codes_batch) == batch_size or idx == len(stock_info_lst):
                    if stock_codes_batch:
                        # token = self.get_token_if_needed()
                        # self.get_stock_share_change_data(token, stock_code)
                        self.get_cninfo_guest(token, stock_codes_batch)
                        logger.info(f"数据已获取并存入数据库: {stock_codes_batch}")
                        time.sleep(call_interval)  # 控制调用频率
                    stock_codes_batch = []
            except Exception as e:
                logger.error("获取数据失败: %s, %s", stock_codes_batch, e)
                continue
            finally:
                if progress_callback:
                    progress_callback(idx, len(stock_info_lst))

cninfo_stock_share_change_fetcher = CninfoStockShareChangeFetcher()