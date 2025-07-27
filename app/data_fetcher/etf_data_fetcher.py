# app/data_fetcher/etf_data_fetcher.py

import tushare as ts
import pandas as pd
from datetime import datetime
from app.models.etf_model import EtfInfo, EtfHist
from app.database import get_db
import logging
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
from sqlalchemy import or_

logger = logging.getLogger(__name__)

class EtfDataFetcher:
    def __init__(self):
        self.pro = ts.pro_api()

    @staticmethod
    def _standardize_code(ts_code: str) -> str:
        return ts_code  # EtfHist表使用ts_code作为主键，保留原格式如"510300.SH"

    def fetch_etf_info_all(self):
        df = self.pro.fund_basic(market='E', status='L')
        df = df.rename(columns={
            'ts_code': 'etf_code',
            'name': 'etf_name',
            'fund_type': 'fund_type',
            'invest_type': 'invest_type',
            'found_date': 'found_date'
        })
        df = df[['etf_code', 'etf_name', 'fund_type', 'invest_type', 'found_date']]
        df['found_date'] = pd.to_datetime(df['found_date'], errors='coerce')

        with get_db() as db:
            for _, row in df.iterrows():
                if pd.isna(row['etf_code']) or pd.isna(row['etf_name']):
                    continue
                db.merge(EtfInfo(**row.to_dict()))
        logger.info("ETF 基本信息入库完成，共 %d 条", len(df))

    def fetch_etf_hist_by_code_and_date(self, code: str, start_date: str, end_date: str):
        ts_code = self._standardize_code(code)
        all_data = []
        offset = 0
        limit = 2000
        while True:
            df = self.pro.fund_daily(
                ts_code=ts_code,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                offset=offset,
                limit=limit,
                fields='ts_code,trade_date,open,high,low,close,vol,amount,change,pct_chg'
            )
            if df.empty:
                break
            all_data.append(df)
            if len(df) < limit:
                break
            offset += limit

        if not all_data:
            return

        df = pd.concat(all_data, ignore_index=True)
        df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df['volume'] = df['vol'] * 100  # 手 → 份
        df['amount'] = df['amount'] * 1000  # 千元 → 元
        df = df.rename(columns={
            'ts_code': 'etf_code',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'change': 'change',
            'pct_chg': 'change_percent'
        })
        df = df[['etf_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'change_percent', 'change']]

        with get_db() as db:
            for _, row in df.iterrows():
                if pd.isna(row['etf_code']) or pd.isna(row['date']):
                    continue
                db.merge(EtfHist(**row.to_dict()))
        logger.info("ETF 日行情数据入库完成: %s (%s - %s), 共 %d 条", code, start_date, end_date, len(df))

    def fetch_etf_hist_all(self):
        from app.models.etf_model import EtfInfo
        with get_db() as db:
            etfs = db.query(EtfInfo).filter(
                (EtfInfo.invest_type.in_(['被动指数型', '增强指数型'])) |
                (EtfInfo.fund_type == '商品型')
            ).all()
            for etf in etfs:
                self.fetch_etf_hist_by_code_and_date(
                    code=etf.etf_code,
                    start_date='2004-10-13',
                    end_date=datetime.today().strftime('%Y-%m-%d')
                )

    def fetch_etf_hist_all_by_date(self, start_date: str, end_date: str):
        """
        增量同步指定日期范围（含首尾）的数据并入库。
        遍历交易日，对每个交易日拉取所有ETF的行情数据。
        """
        trade_dates = TradeCalendarReader.get_trade_dates(start=start_date, end=end_date)
        for date in trade_dates:
            date_str = date.strftime('%Y%m%d')
            logger.info("正在拉取交易日 %s 的ETF行情数据", date_str)

            offset = 0
            limit = 2000
            all_data = []

            while True:
                df = self.pro.fund_daily(
                    trade_date=date_str,
                    offset=offset,
                    limit=limit,
                    fields='ts_code,trade_date,open,high,low,close,vol,amount,change,pct_chg'
                )
                if df.empty:
                    break
                all_data.append(df)
                if len(df) < limit:
                    break
                offset += limit

            if not all_data:
                continue

            df = pd.concat(all_data, ignore_index=True)
            df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df['volume'] = df['vol'] * 100  # 手 → 份
            df['amount'] = df['amount'] * 1000  # 千元 → 元
            df = df.rename(columns={
                'ts_code': 'etf_code',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'change': 'change',
                'pct_chg': 'change_percent'
            })
            df = df[['etf_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'change_percent', 'change']]

            with get_db() as db:
                for _, row in df.iterrows():
                    if pd.isna(row['etf_code']) or pd.isna(row['date']):
                        continue
                    db.merge(EtfHist(**row.to_dict()))
            logger.info("入库完成：%s，共 %d 条", date_str, len(df))
