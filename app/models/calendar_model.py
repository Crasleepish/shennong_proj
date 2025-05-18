from app.database import Base
from sqlalchemy import Column, Date
import logging

logger = logging.getLogger(__name__)


class TradeCalendar(Base):
    __tablename__ = "trade_calendar"
    date = Column(Date, primary_key=True)