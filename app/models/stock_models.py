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
    total_shares = Column("total_shares", BigInteger)  # 单位：原数据成交量单位为“万股”, 入库单位为“股”
    mkt_cap = Column("mkt_cap", BigInteger)  # 单位：元
    
    def __repr__(self):
        return f"<StockHistUnadj(stock_code='{self.stock_code}', date='{self.date}')>"

class UpdateFlag(Base):
    __tablename__ = 'update_flag'
    stock_code = Column("stock_code", String(10), primary_key=True)
    action_update_flag = Column('action_update_flag', String(1), nullable=False, default=1) #是否需要更新该股票的公司行动数据, 1:需要更新，0：不需要更新
    fundamental_update_flag = Column('fundamental_update_flag', String(1), nullable=False, default=1) #是否需要更新该股票的基本面数据, 1:需要更新，0：不需要更新

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
    total_shares = Column("total_shares", BigInteger)  # 单位：原数据成交量单位为“万股”, 入库单位为“股”
    mkt_cap = Column("mkt_cap", BigInteger)  # 单位：元
    
    def __repr__(self):
        return f"<StockHistAdj(stock_code='{self.stock_code}', date='{self.date}')>"

class FundamentalData(Base):
    __tablename__ = "fundamental_data"
    
    # 主键：股票代码和报告日期
    stock_code = Column(String(10), primary_key=True)
    report_date = Column(Date, primary_key=True)
    
    # 基本面数据字段
    total_equity = Column(Float, nullable=False)                # 归属于母公司所有者权益合计
    total_assets = Column(Float, nullable=False)                # 资产合计
    current_liabilities = Column(Float, nullable=True)          # 流动负债合计
    noncurrent_liabilities = Column(Float, nullable=True)       # 非流动负债合计
    net_profit = Column(Float, nullable=False)                  # 归属于母公司所有者的净利润
    operating_profit = Column(Float, nullable=True)             # 营业利润
    total_revenue = Column(Float, nullable=False)               # 营业总收入
    total_cost = Column(Float, nullable=False)                  # 营业总成本
    net_cash_from_operating = Column(Float, nullable=False)     # 经营活动产生的现金流量净额
    cash_for_fixed_assets = Column(Float, nullable=True)        # 购建固定资产、无形资产和其他长期资产支付的现金

    def __repr__(self):
        return f"<FundamentalData(stock_code='{self.stock_code}', report_date='{self.report_date}')>"
    
class SuspendData(Base):
    __tablename__ = "suspend_data"
    
    # 主键：序号
    id = Column(Integer, primary_key=True, autoincrement=True)
    #停复牌数据
    stock_code = Column(String(10), nullable=False)         # 股票代码
    suspend_date = Column(Date, nullable=False)             # 停牌时间
    resume_date = Column(Date, nullable=True)               # 停牌截止时间
    suspend_period = Column(String(10), nullable=True)      # 停牌期限
    suspend_reason = Column(String(100), nullable=True)     # 停牌原因
    market = Column(String(20), nullable=True)              # 所属市场

    __table_args__ = (
        Index('idx_suspend_data_stock_code_suspend_date', 'stock_code', 'suspend_date'),
    )

    def __repr__(self):
        return f"<SuspendData(id='{self.id}'>"
