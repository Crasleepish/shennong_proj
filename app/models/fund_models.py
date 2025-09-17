from sqlalchemy import Column, String, Float, Date, BigInteger, Index, Integer, Numeric, Sequence, Text, LargeBinary, desc
from app.database import Base


class FundInfo(Base):
    __tablename__ = 'fund_info'
    # 使用证券代码作为主键
    fund_code = Column('fund_code', String(10), primary_key=True)
    fund_name = Column('fund_name', String(50), nullable=False)
    fund_type = Column('fund_type', String(20), nullable=True)                 # 基金类型
    invest_type = Column('invest_type', String(20), nullable=True)             # 投资类型
    found_date = Column('found_date', Date, nullable=False)                      # 上市时间
    fee_rate = Column('fee_rate', Float, nullable=True)                           # 管理费
    commission_rate = Column('commission_rate', Float, nullable=True)                    # 托管费
    market = Column('market', String(8), nullable=True)                        # E场内O场外

    def __repr__(self):
        return f"<FundInfo(fund_code='{self.fund_code}', fund_name='{self.fund_name}')>"
    

class FundHist(Base):
    __tablename__ = 'fund_hist'
    # 联合主键 (fund_code, date)
    fund_code = Column("fund_code", String(10), primary_key=True)
    date = Column("date", Date, primary_key=True)
    value = Column("value", Float)  # 单位净值
    net_value = Column("net_value", Float)  # 复权单位净值
    change_percent = Column("change_percent", Float)  # 相对上一交易日的净值变化率，单位：%

    def __repr__(self):
        return f"<FundHist(fund_code='{self.fund_code}', date='{self.date}')>"

class FundBeta(Base):
    __tablename__ = "fund_beta"

    # 联合主键：code + date
    code = Column("code", String(20), primary_key=True)  # 基金/ETF 代码
    date = Column("date", Date, primary_key=True)        # 回归结果对应的交易日

    # 五因子暴露
    MKT = Column("MKT", Float, nullable=True)   # 市场因子暴露
    SMB = Column("SMB", Float, nullable=True)   # 市值因子暴露
    HML = Column("HML", Float, nullable=True)   # 价值因子暴露
    QMJ = Column("QMJ", Float, nullable=True)   # 质量因子暴露

    # 常数项（alpha）
    const = Column("const", Float, nullable=True)

    P_json = Column("P_json", Text, nullable=True)
    P_bin = Column("P_bin", LargeBinary, nullable=True)
    log_nav_fit = Column("log_nav_fit", Float, nullable=True) #历史累计对数收益
    log_nav_true = Column("log_nav_true", Float, nullable=True)  #拟合的历史累计收益，用于ECM（误差修正项，在观测里加净值偏差）
    gamma = Column("gamma", Float, nullable=True)

    __table_args__ = (
        Index("idx_fund_beta_date", "date"),  # 新增索引
    )

    def __repr__(self):
        return f"<FundBeta(code='{self.code}', date='{self.date}')>"