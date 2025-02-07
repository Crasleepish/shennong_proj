from sqlalchemy import Column, Integer, String, Date
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