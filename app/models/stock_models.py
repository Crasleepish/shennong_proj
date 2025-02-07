from sqlalchemy import Column, String, Float, Date, BigInteger
from app.database import Base


class StockInfo(Base):
    __tablename__ = 'stock_info'
    # 使用证券代码作为主键
    stock_code = Column('stock_code', String(10), primary_key=True)
    stock_name = Column('stock_name', String(50), nullable=True)
    listing_date = Column('listing_date', Date, nullable=True)
    market = Column('market', String(2), nullable=True)

    def __repr__(self):
        return f"<StockInfo(stock_code='{self.stock_code}', stock_name='{self.stock_name}')>"
    

class StockHistUnadj(Base):
    __tablename__ = 'stock_hist_unadj'
    # 联合主键 (stock_code, date)
    stock_code = Column("stock_code", String(10), primary_key=True)
    date = Column("date", Date, primary_key=True)
    open = Column("open", Float)
    close = Column("close", Float)
    high = Column("high", Float)
    low = Column("low", Float)
    volume = Column("volume", BigInteger)      # 注意：原数据成交量单位为“手”，入库单位为“股”
    turnover = Column("amount", Float)          # 单位：元
    amplitude = Column("amplitude", Float)           # 单位：%
    change_percent = Column("change_percent", Float)      # 单位：%
    change = Column("change", Float)             # 单位：元
    turnover_rate = Column("turnover_rate", Float)      # 单位：%
    
    def __repr__(self):
        return f"<StockHist(stock_code='{self.stock_code}', date='{self.date}')>"