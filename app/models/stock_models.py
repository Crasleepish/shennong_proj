from sqlalchemy import Column, String, Float, Date, BigInteger, Index, Integer
from app.database import Base


class StockInfo(Base):
    __tablename__ = 'stock_info'
    # 使用证券代码作为主键
    stock_code = Column('stock_code', String(10), primary_key=True)
    stock_name = Column('stock_name', String(50), nullable=False)
    listing_date = Column('listing_date', Date, nullable=False)
    market = Column('market', String(10), nullable=False)

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
        return f"<StockHistUnadj(stock_code='{self.stock_code}', date='{self.date}')>"

class UpdateFlag(Base):
    __tablename__ = 'update_flag'
    stock_code = Column("stock_code", String(10), primary_key=True)
    action_update_flag = Column('action_update_flag', String(1), nullable=True) #是否需要更新该股票的公司行动数据, 1:需要更新，0：不需要更新

    def __repr__(self):
        return f"<UpdateFlag(stock_code='{self.stock_code}')>"

class FutureTask(Base):
    __tablename__ = 'future_task'
    task_id = Column("task_id", Integer, primary_key=True, autoincrement=True)
    task_type = Column("task_type", String(50)) #任务类型，如：update_adj 更新前复权数据
    stock_code = Column("stock_code", String(10))
    task_date = Column("task_date", Date)
    task_status = Column("task_status", String(10))

    __table_args__ = (
        Index('idx_future_task_stock_code_task_date', 'stock_code', 'task_date'),
    )

    def __repr__(self):
        return f"<FutureTask(task_id='{self.task_id}')>"

class CompanyAction(Base):
    __tablename__ = 'company_action'
    # 联合主键(stock_code, ex_dividend_date)
    stock_code = Column("stock_code", String(10), primary_key=True)
    ex_dividend_date = Column("ex_dividend_date", Date, primary_key=True)
    bonus_ratio = Column("bonus_ratio", Float)
    conversion_ratio = Column("conversion_ratio", Float)
    dividend_per_share = Column("dividend_per_share", Float)
    rights_issue_ratio = Column("rights_issue_ratio", Float)
    rights_issue_price = Column("rights_issue_price", Float)
    announcement_date = Column("announcement_date", Date)
    record_date = Column("record_date", Date)

    def __repr__(self):
        return f"<CompanyAction(stock_code='{self.stock_code}', ex_dividend_date='{self.ex_dividend_date}')>"
    

class StockHistAdj(Base):
    __tablename__ = 'stock_hist_adj'
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
        return f"<StockHistAdj(stock_code='{self.stock_code}', date='{self.date}')>"
