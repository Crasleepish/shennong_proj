from sqlalchemy import Column, String, Float, Date, BigInteger, Index, Integer, Numeric, Sequence
from app.database import Base


class EtfInfo(Base):
    __tablename__ = 'etf_info'
    # 使用证券代码作为主键
    etf_code = Column('etf_code', String(20), primary_key=True)
    etf_name = Column('etf_name', String(50), nullable=False)
    fund_type = Column('fund_type', String(20), nullable=True)
    invest_type = Column('invest_type', String(20), nullable=True)
    found_date = Column('found_date', Date, nullable=True)

    def __repr__(self):
        return f"<EtfInfo(etf_code='{self.etf_code}', etf_name='{self.etf_name}')>"
    

class EtfHist(Base):
    __tablename__ = 'etf_hist'
    # 联合主键 (etf_code, date)
    etf_code = Column("etf_code", String(20), primary_key=True)
    date = Column("date", Date, primary_key=True)
    open = Column("open", Float)
    close = Column("close", Float)
    high = Column("high", Float)
    low = Column("low", Float)
    volume = Column("volume", BigInteger)      # 注意：原数据成交量单位为“手”，入库单位为“份”，1手=100份
    amount = Column("amount", Float)          # 注意：原数据成交量单位为“千元”，入库单位为“元”
    change_percent = Column("change_percent", Float)      # 单位：%
    change = Column("change", Float)             # 单位：元
    
    def __repr__(self):
        return f"<EtfHist(etf_code='{self.etf_code}', date='{self.date}')>"
