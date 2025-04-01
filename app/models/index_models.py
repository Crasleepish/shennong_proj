from sqlalchemy import Column, String, Float, Date, BigInteger, Index, Integer, Numeric, Sequence
from app.database import Base


class IndexInfo(Base):
    __tablename__ = 'index_info'
    # 使用证券代码作为主键
    index_code = Column('index_code', String(10), primary_key=True)
    index_name = Column('index_name', String(50), nullable=False)
    market = Column('market', String(10), nullable=False)

    def __repr__(self):
        return f"<IndexInfo(index_code='{self.index_code}', index_name='{self.index_name}')>"
    

class IndexHist(Base):
    __tablename__ = 'index_hist'
    # 联合主键 (index_code, date)
    index_code = Column("index_code", String(10), primary_key=True)
    date = Column("date", Date, primary_key=True)
    open = Column("open", Float)
    close = Column("close", Float)
    high = Column("high", Float)
    low = Column("low", Float)
    volume = Column("volume", BigInteger)      # 注意：原数据成交量单位为“手”，入库单位为“股”
    amount = Column("amount", Float)          # 单位：元
    amplitude = Column("amplitude", Float)           # 单位：%
    change_percent = Column("change_percent", Float)      # 单位：%
    change = Column("change", Float)             # 单位：元
    turnover_rate = Column("turnover_rate", Float)      # 单位：%
    
    def __repr__(self):
        return f"<IndexHist(index_code='{self.index_code}', date='{self.date}')>"
