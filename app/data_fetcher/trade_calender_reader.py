import pandas as pd
from app.database import get_db
from app.models.calendar_model import TradeCalendar
import logging

logger = logging.getLogger(__name__)


class TradeCalendarReader:
    """
    从数据库中读取交易日历数据。
    """

    @staticmethod
    def get_trade_dates(start: str = None, end: str = None) -> pd.DatetimeIndex:
        with get_db() as db:
            query = db.query(TradeCalendar)
            if start:
                query = query.filter(TradeCalendar.date >= pd.to_datetime(start))
            if end:
                query = query.filter(TradeCalendar.date <= pd.to_datetime(end))

            df = pd.read_sql(query.statement, db.bind)
            df = df.sort_values("date")
            return pd.DatetimeIndex(df["date"])
