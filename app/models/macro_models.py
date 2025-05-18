# app/models/macro_models.py

from sqlalchemy import Column, Date, Float
from app.database import Base

class SocialFinancing(Base):
    __tablename__ = "macro_social_financing"
    date = Column(Date, primary_key=True)
    total = Column(Float)

class OfficialPmi(Base):
    __tablename__ = "macro_pmi"
    date = Column(Date, primary_key=True)
    value = Column(Float)

class LprRate(Base):
    __tablename__ = "macro_lpr"
    date = Column(Date, primary_key=True)
    lpr_1y = Column(Float)

class CpiYearly(Base):
    __tablename__ = "macro_cpi"
    date = Column(Date, primary_key=True)
    cpi = Column(Float)

class MoneySupply(Base):
    __tablename__ = "macro_money"
    date = Column(Date, primary_key=True)
    m1 = Column(Float)
    m1_yoy = Column(Float)
    m1_mom = Column(Float)
    m2 = Column(Float)
    m2_yoy = Column(Float)
    m2_mom = Column(Float)

class FxReserves(Base):
    __tablename__ = "macro_fx_reserve"
    date = Column(Date, primary_key=True)
    reserve = Column(Float)

class GoldReserve(Base):
    __tablename__ = "macro_gold_reserve"

    date = Column(Date, primary_key=True, comment="月份")
    gold_reserve = Column(Float, nullable=True, comment="黄金储备-数值（万盎司）")
    gold_yoy = Column(Float, nullable=True, comment="黄金储备-同比（万盎司）")
    gold_mom = Column(Float, nullable=True, comment="黄金储备-环比（万盎司）")

class BondYield(Base):
    __tablename__ = "bond_yield"

    date = Column(Date, primary_key=True, comment="日期")
    cn_2y = Column(Float, nullable=True, comment="中国国债收益率2年")
    cn_5y = Column(Float, nullable=True, comment="中国国债收益率5年")
    cn_10y = Column(Float, nullable=True, comment="中国国债收益率10年")
    cn_30y = Column(Float, nullable=True, comment="中国国债收益率30年")
    cn_10y_2y = Column(Float, nullable=True, comment="中国国债收益率10年-2年")
