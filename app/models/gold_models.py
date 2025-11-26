# gold_models.py
from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Date,
    DateTime,
    Numeric,
    UniqueConstraint,
)

from app.database import Base


class GoldCFTCReport(Base):
    """
    CFTC Disaggregated Futures-and-Options Combined 报告中
    GOLD - COMMODITY EXCHANGE INC. 的周度数据。
    对应 com_disagg_txt_{year}.zip 里的 GOLD 行。
    """
    __tablename__ = "gold_cftc_report"
    __table_args__ = (
        UniqueConstraint("report_date", "contract_market_code", "market_code",
                         name="uq_gold_cftc_report_date_codes"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 关键标识字段
    market_name = Column(String(128), nullable=False)  # "GOLD - COMMODITY EXCHANGE INC."
    as_of_date = Column(Date, nullable=False)          # As_of_Date_In_Form_YYMMDD -> date
    report_date = Column(Date, nullable=False)         # Report_Date_as_YYYY_MM_DD

    contract_market_code = Column(String(16), nullable=True)  # CFTC_Contract_Market_Code
    market_code = Column(String(8), nullable=True)            # CFTC_Market_Code
    region_code = Column(String(8), nullable=True)            # CFTC_Region_Code
    commodity_code = Column(String(8), nullable=True)         # CFTC_Commodity_Code
    futonly_or_combined = Column(String(16), nullable=True)   # FutOnly_or_Combined

    # 关键仓位字段（All = Futures + Options Combined）
    open_interest_all = Column(BigInteger, nullable=True)

    prod_merc_long_all = Column(BigInteger, nullable=True)
    prod_merc_short_all = Column(BigInteger, nullable=True)

    swap_long_all = Column(BigInteger, nullable=True)
    swap_short_all = Column(BigInteger, nullable=True)
    swap_spread_all = Column(BigInteger, nullable=True)

    m_money_long_all = Column(BigInteger, nullable=True)
    m_money_short_all = Column(BigInteger, nullable=True)
    m_money_spread_all = Column(BigInteger, nullable=True)

    other_rept_long_all = Column(BigInteger, nullable=True)
    other_rept_short_all = Column(BigInteger, nullable=True)
    other_rept_spread_all = Column(BigInteger, nullable=True)

    tot_rept_long_all = Column(BigInteger, nullable=True)
    tot_rept_short_all = Column(BigInteger, nullable=True)

    nonrept_long_all = Column(BigInteger, nullable=True)
    nonrept_short_all = Column(BigInteger, nullable=True)

    # 非商业净多头 = 管理基金 + 其他可报告 的 Long - Short
    noncomm_net_all = Column(BigInteger, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class GoldFutureCurve(Base):
    """
    黄金期货价格曲线（每日一个截面，一天内同一 symbol 只保留一条）。
    来自 barchart GC 根合约列表的快照。
    """
    __tablename__ = "gold_future_curve"
    __table_args__ = (
        UniqueConstraint("trade_date", "symbol", name="uq_gold_future_curve_date_symbol"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    trade_date = Column(Date, nullable=False)     # 根据 tradeTime 解析出来的日期
    symbol = Column(String(16), nullable=False)   # 如 GCG25
    contract_symbol = Column(String(64), nullable=True)  # 如 "GCG25 (Feb '25)"

    last_price = Column(Numeric(18, 4), nullable=True)
    price_change = Column(Numeric(18, 4), nullable=True)
    open_price = Column(Numeric(18, 4), nullable=True)
    high_price = Column(Numeric(18, 4), nullable=True)
    low_price = Column(Numeric(18, 4), nullable=True)
    previous_price = Column(Numeric(18, 4), nullable=True)

    volume = Column(BigInteger, nullable=True)
    open_interest = Column(BigInteger, nullable=True)

    # Barchart 的原始字段（字符串）也可以留一个备用
    trade_time_str = Column(String(32), nullable=True)  # 如 '11/21/25'
    product_code = Column(String(16), nullable=True)    # 一般是 'FUT'
    symbol_code = Column(String(8), nullable=True)      # 'FUT'
    symbol_type = Column(String(8), nullable=True)      # '2'
    has_options = Column(String(8), nullable=True)      # 'Yes' / 'No'

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
