import tushare as ts
import pandas as pd
from app.database import get_db
from app.models.calendar_model import TradeCalendar
from typing import List
import logging
from datetime import datetime
from sqlalchemy import and_, desc

logger = logging.getLogger(__name__)

class CalendarFetcher:
    """
    用于从 Tushare 获取交易日历数据并入库。
    """

    def __init__(self):
        self.pro = ts.pro_api()

    def fetch_trade_calendar(self, start: str = "20000101", end: str = None):
        """
        拉取并写入交易日数据。
        :param start: 开始日期，格式如 "20000101"
        :param end: 结束日期，默认今天
        """
        if end is None:
            end = pd.Timestamp.today().strftime("%Y%m%d")

        df = self.pro.trade_cal(exchange='', start_date=start, end_date=end)
        df = df[df["is_open"] == 1].copy()  # 只保留交易日
        df["cal_date"] = pd.to_datetime(df["cal_date"])

        with get_db() as db:
            count = 0
            for _, row in df.iterrows():
                cal = TradeCalendar(date=row["cal_date"])
                db.merge(cal)
                count += 1
            logger.info("交易日数据入库完成，共 %d 天", count)

    def get_trade_date(self, start: str, end: str = None, format: str = "%Y%m%d", limit: int = None, ascending: bool = True) -> List[str]:
        """
        获取指定日期范围内的交易日列表
        
        :param start: 开始日期，格式如 "20000101"
        :param end: 结束日期，默认今天
        :param format: 返回日期的格式，默认 "%Y%m%d"
        :return: 交易日列表，如 ["20240101", "20240102", ...]
        """
        if end is None:
            end = datetime.now().strftime("%Y%m%d")
        
        # 将字符串日期转换为datetime对象用于查询
        start_date = datetime.strptime(start, "%Y%m%d")
        end_date = datetime.strptime(end, "%Y%m%d")
        
        trade_dates = []
        with get_db() as db:
            # 查询指定日期范围内的交易日
            calendars = db.query(TradeCalendar).filter(
                and_(
                    TradeCalendar.date >= start_date,
                    TradeCalendar.date <= end_date
                )
            )

            if ascending:
                calendars = calendars.order_by(TradeCalendar.date)
            else:
                calendars = calendars.order_by(desc(TradeCalendar.date))

            if limit is not None:
                calendars = calendars.limit(limit)
            calendars = calendars.all()
            
            # 将日期格式化为指定格式
            trade_dates = [cal.date.strftime(format) for cal in calendars]
        
        return trade_dates
    
    def get_prev_trade_date(self, current_date: str, format: str = "%Y%m%d") -> str:
        """
        获取给定交易日的前一个交易日。
        如果current_date不是交易日，则返回最后一个交易日。
        """
        all_dates = self.get_trade_date("19900101", current_date, format=format, limit=2, ascending=False)
        formated_current_date = datetime.strptime(current_date, '%Y%m%d').strftime(format)
        if formated_current_date not in all_dates:
            if len(all_dates) == 0:
                return None
            return all_dates[0]
        if len(all_dates) < 2:
            return None  # 没有更早的交易日
        return all_dates[1]
    
    def get_next_trade_date(self, current_date: str, format: str = "%Y%m%d") -> str:
        """
         获取下一交易日。
         如果当前日期不是交易日，则返回下一最近的交易日。
        """
        all_dates = self.get_trade_date(start=current_date, format=format, limit=1, ascending=True)
        formated_current_date = datetime.strptime(current_date, '%Y%m%d').strftime(format)
        if formated_current_date not in all_dates: # 当前日期不在交易日列表中
            if len(all_dates) == 0: # 没有交易日
                return None
            return all_dates[0]
        if len(all_dates) < 2:
            return None # 没有更晚的交易日
        return all_dates[1]