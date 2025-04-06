from sqlalchemy import Column, String, Float, Date, BigInteger, Index, Integer, Numeric, Sequence
from app.database import Base


class FundInfo(Base):
    __tablename__ = 'fund_info'
    # 使用证券代码作为主键
    fund_code = Column('fund_code', String(10), primary_key=True)
    fund_name = Column('fund_name', String(50), nullable=False)
    fee_rete = Column('fee_rete', Float)

    def __repr__(self):
        return f"<FundInfo(fund_code='{self.fund_code}', fund_name='{self.fund_name}')>"
    

class FundHist(Base):
    __tablename__ = 'fund_hist'
    # 联合主键 (fund_code, date)
    fund_code = Column("fund_code", String(10), primary_key=True)
    date = Column("date", Date, primary_key=True)
    value = Column("value", Float)
    #复权因子
    adjust_factor = Column("adjust_factor", Float)
    #复权净值
    net_value = Column("net_value", Float)
    change_percent = Column("change_percent", Float)      # 单位：%
    
    def __repr__(self):
        return f"<FundHist(fund_code='{self.fund_code}', date='{self.date}')>"
