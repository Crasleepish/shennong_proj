from sqlalchemy import Column, String, Float, Date, BigInteger, Index, Integer, Numeric, Sequence
from app.database import Base


class StockInfo(Base):
    __tablename__ = 'stock_info'
    # 使用证券代码作为主键
    stock_code = Column("stock_code", String(20), primary_key=True)
    stock_name = Column('stock_name', String(50), nullable=False)
    market = Column('market', String(10), nullable=True)
    exchange = Column('exchange', String(10), nullable=True)
    industry = Column('industry', String(50), nullable=True)
    listing_date = Column('listing_date', Date, nullable=False)

    __table_args__ = (
        Index('idx_stock_code_task_date_stock_info', 'stock_code'),
    )

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
    amount = Column("amount", Float)          # 单位：元
    pre_close = Column("pre_close", Float)
    change_percent = Column("change_percent", Float)      # 单位：%
    change = Column("change", Float)             # 单位：元
    turnover_rate = Column("turnover_rate", Float)      # 单位：%
    turnover_rate_f = Column("turnover_rate_f", Float)  # 单位：%
    volume_ratio = Column("volume_ratio", Float)
    pe = Column("pe", Float)
    pe_ttm = Column("pe_ttm", Float)
    pb = Column("pb", Float)
    ps = Column("ps", Float)
    ps_ttm = Column("ps_ttm", Float)
    dv_ratio = Column("dv_ratio", Float)
    dv_ttm = Column("dv_ttm", Float)
    total_shares = Column("total_share", BigInteger)
    float_shares = Column("float_share", BigInteger)
    free_shares = Column("free_share", BigInteger)
    mkt_cap = Column("mkt_cap", BigInteger)  # 单位：元
    circ_mv = Column("circ_mv", BigInteger)

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
    

class AdjFactor(Base):
    __tablename__ = 'adj_factor'
    # 联合主键(stock_code, date)
    stock_code = Column("stock_code", String(10), primary_key=True)
    date = Column("date", Date, primary_key=True)
    adj_factor = Column("adj_factor", Float)

    def __repr__(self):
        return f"<AdjFactor(stock_code='{self.stock_code}', date='{self.date}')>"


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
    amount = Column("amount", Float)          # 单位：元
    amplitude = Column("amplitude", Float)           # 单位：%
    change_percent = Column("change_percent", Float)      # 单位：%
    change = Column("change", Float)             # 单位：元
    turnover_rate = Column("turnover_rate", Float)      # 单位：%
    mkt_cap = Column("mkt_cap", BigInteger)  # 单位：元
    total_shares = Column("total_shares", BigInteger)  # 单位：股
    
    def __repr__(self):
        return f"<StockHistAdj(stock_code='{self.stock_code}', date='{self.date}')>"

class FundamentalData(Base):
    __tablename__ = "fundamental_data"
    
    # 主键：股票代码和报告日期
    stock_code = Column(String(10), primary_key=True)
    report_date = Column(Date, primary_key=True)
    
    # 基本面数据字段
    total_equity = Column(Float, nullable=True)                # 归属于母公司所有者权益合计
    total_assets = Column(Float, nullable=True)                # 资产合计
    current_liabilities = Column(Float, nullable=True)         # 流动负债合计
    noncurrent_liabilities = Column(Float, nullable=True)       # 非流动负债合计
    net_profit = Column(Float, nullable=True)                  # 归属于母公司所有者的净利润
    operating_profit = Column(Float, nullable=True)             # 营业利润
    total_revenue = Column(Float, nullable=True)               # 营业总收入
    total_cost = Column(Float, nullable=True)                  # 营业总成本
    net_cash_from_operating = Column(Float, nullable=True)     # 经营活动产生的现金流量净额
    cash_for_fixed_assets = Column(Float, nullable=True)        # 购建固定资产、无形资产和其他长期资产支付的现金

    def __repr__(self):
        return f"<FundamentalData(stock_code='{self.stock_code}', report_date='{self.report_date}')>"
    
class SuspendData(Base):
    __tablename__ = "suspend_data"
    
    # 主键：序号
    id = Column(BigInteger, primary_key=True, autoincrement=True)
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

class StockShareChangeCNInfo(Base):
    __tablename__ = 'stock_share_change_cn_info'
    id = Column(BigInteger, primary_key=True, autoincrement=True)

    SECCODE = Column(String, comment='证券代码')
    SECNAME = Column(String, comment='证券简称')
    ORGNAME = Column(String, comment='机构名称')
    DECLAREDATE = Column(Date, comment='公告日期')
    VARYDATE = Column(Date, comment='变动日期')
    F001V = Column(String, comment='变动原因编码')
    F002V = Column(String, comment='变动原因')
    F003N = Column(Numeric(12, 4), comment='总股本 (单位：万股)')
    F004N = Column(Numeric(12, 4), comment='未流通股份 (单位：万股)')
    F005N = Column(Numeric(12, 4), comment='发起人股份 (单位：万股)')
    F006N = Column(Numeric(12, 4), comment='国家持股 (单位：万股)')
    F007N = Column(Numeric(12, 4), comment='国有法人持股 (单位：万股)')
    F008N = Column(Numeric(12, 4), comment='境内法人持股 (单位：万股)')
    F009N = Column(Numeric(12, 4), comment='境外法人持股 (单位：万股)')
    F010N = Column(Numeric(12, 4), comment='自然人持股 (单位：万股)')
    F011N = Column(Numeric(12, 4), comment='募集法人股 (单位：万股)')
    F012N = Column(Numeric(12, 4), comment='内部职工股 (单位：万股)')
    F013N = Column(Numeric(12, 4), comment='转配股 (单位：万股)')
    F014N = Column(Numeric(12, 4), comment='其他流通受限股份 (单位：万股)')
    F015N = Column(Numeric(12, 4), comment='优先股 (单位：万股)')
    F016N = Column(Numeric(12, 4), comment='其他未流通股 (单位：万股)')
    
    F021N = Column(Numeric(12, 4), comment='已流通股份 (单位：万股)')
    F022N = Column(Numeric(12, 4), comment='人民币普通股 (单位：万股)')
    F023N = Column(Numeric(12, 4), comment='境内上市外资股（B股） (单位：万股)')
    F024N = Column(Numeric(12, 4), comment='境外上市外资股（H股） (单位：万股)')
    F025N = Column(Numeric(12, 4), comment='高管股 (单位：万股)')
    F026N = Column(Numeric(12, 4), comment='其他流通股 (单位：万股)')
    F028N = Column(Numeric(12, 4), comment='流通受限股份 (单位：万股)')
    
    F017N = Column(Numeric(12, 4), comment='配售法人股 (单位：万股)')
    F018N = Column(Numeric(12, 4), comment='战略投资者持股 (单位：万股)')
    F019N = Column(Numeric(12, 4), comment='证券投资基金持股 (单位：万股)')
    F020N = Column(Numeric(12, 4), comment='一般法人持股 (单位：万股)')
    F029N = Column(Numeric(12, 4), comment='国家持股（受限） (单位：万股)')
    F030N = Column(Numeric(12, 4), comment='国有法人持股（受限） (单位：万股)')
    F031N = Column(Numeric(12, 4), comment='其他内资持股（受限） (单位：万股)')
    F032N = Column(Numeric(12, 4), comment='其中：境内法人持股 (单位：万股)')
    F033N = Column(Numeric(12, 4), comment='其中：境内自然人持股 (单位：万股)')
    F034N = Column(Numeric(12, 4), comment='外资持股（受限） (单位：万股)')
    F035N = Column(Numeric(12, 4), comment='其中：境外法人持股 (单位：万股)')
    F036N = Column(Numeric(12, 4), comment='其中：境外自然人持股 (单位：万股)')
    
    F037N = Column(Numeric(12, 4), comment='其中：限售高管股 (单位：万股)')
    F038N = Column(Numeric(12, 4), comment='其中：限售B股 (单位：万股)')
    F040N = Column(Numeric(12, 4), comment='其中：限售H股 (单位：万股)')
    F027C = Column(String(1), comment='最新记录标识 (0-否，1-是)')
    F049N = Column(Numeric(12, 4), comment='其他 (单位：万股，仅适用于北交所上市公司)')
    F050N = Column(Numeric(12, 4), comment='控股股东、实际控制人 (单位：万股，仅适用于北交所上市公司)')
    
class MarketFactors(Base):
    __tablename__ = 'market_factors'
    # 主键：日期
    date = Column(Date, primary_key=True)

    MKT = Column(Numeric(10, 6), comment='市场因子')
    SMB = Column(Numeric(10, 6), comment='市值因子')
    HML = Column(Numeric(10, 6), comment='价值因子')
    QMJ = Column(Numeric(10, 6), comment='质量因子')
    VOL = Column(Numeric(10, 6), comment='波动率因子')
    LIQ = Column(Numeric(10, 6), comment='流动性因子')

    def __repr__(self):
        return f"<MarketFactors(date='{self.date}'>"