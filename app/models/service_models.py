from sqlalchemy import Column, BigInteger, DateTime, Text, PrimaryKeyConstraint, LargeBinary
from app.database import Base

class PortfolioWeights(Base):
    __tablename__ = 'portfolio_weights'
    
    portfolio_id = Column(BigInteger, nullable=False)
    date = Column(DateTime, nullable=False)
    codes = Column(Text, nullable=True)
    weights = Column(Text, nullable=True)
    weights_ewma = Column(Text, nullable=True)
    cov_meta = Column(Text, nullable=True)              # 存储元信息，如 shape、dtype 等
    cov_matrix = Column(LargeBinary, nullable=True)     # 新增 BLOB 字段

    __table_args__ = (
        PrimaryKeyConstraint('portfolio_id', 'date', name='portfolio_weights_pk'),
    )

    def __repr__(self):
        return f"<PortfolioWeights(portfolio_id={self.portfolio_id}, date={self.date})>"
    
from sqlalchemy import Column, String, Float, Date, JSON
from app.database import Base

# 当前持仓信息表（逻辑资产到实际资产映射 + 当前份额）
from sqlalchemy import Column, String, Float, Integer, PrimaryKeyConstraint
from app.database import Base

class CurrentHolding(Base):
    __tablename__ = 'current_holdings'

    asset = Column(String, nullable=False)
    code = Column(String, nullable=False)
    name = Column(String, nullable=True)
    amount = Column(Float, nullable=True)
    portfolio_id = Column(Integer, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('portfolio_id', 'code', name='current_holdings_pk'),
    )

    def __repr__(self):
        return f"<CurrentHolding(portfolio_id={self.portfolio_id}, asset={self.asset}, code={self.code})>"


# 调仓记录日志表（记录每次调仓的明细变更）
class RebalanceLog(Base):
    __tablename__ = 'rebalance_logs'

    date = Column(Date, primary_key=True)       # 调仓执行日期
    operations = Column(JSON, nullable=False)   # JSON 格式，示例: [{"from": "xxx", "to": "yyy", "amount": 123.4}, ...]

    def __repr__(self):
        return f"<RebalanceLog(date={self.date})>"
