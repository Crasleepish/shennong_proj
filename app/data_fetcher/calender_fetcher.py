import tushare as ts
import pandas as pd
from app.database import get_db
from app.models.calendar_model import TradeCalendar
import logging

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
