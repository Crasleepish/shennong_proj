from app.models.service_models import PortfolioWeights
from app.database import get_db
import json
import pandas as pd

def query_weights_by_date(date: str, portfolio_id: int = 1) -> dict:
    """
    查询某一日期的平滑后组合权重（weights_ewma 字段）

    :param date: 形如 '2024-07-20'
    :param portfolio_id: 默认组合ID为 1
    :return: { "weights": { asset: percent, ... } }
    """
    query_date = pd.to_datetime(date).date()

    with get_db() as db:
        row = db.query(PortfolioWeights).filter_by(
            portfolio_id=portfolio_id,
            date=query_date
        ).first()

        if row is None or not row.weights_ewma:
            return { "weights": {} }

        try:
            weights_dict = json.loads(row.weights_ewma)
        except Exception:
            weights_dict = {}

        return { "weights": weights_dict }
    
    
from typing import Optional
from sqlalchemy import desc

def query_latest_portfolio_by_id(portfolio_id: int) -> pd.Series:
    """
    查询指定 portfolio_id 最新一条组合权重记录
    返回: pd.Series，包含 fields: date, weights, weights_ewma
    - weights / weights_ewma: 若为 JSON 字符串则解析为 dict，若为空返回 None
    - 若查无记录，返回空 Series
    """
    with get_db() as db:
        row: Optional[PortfolioWeights] = (
            db.query(PortfolioWeights)
              .filter(PortfolioWeights.portfolio_id == portfolio_id)
              .order_by(desc(PortfolioWeights.date))
              .first()
        )

        if not row:
            return pd.Series(dtype=object)

        def _loads_or_none(s: Optional[str]):
            if not s:
                return None
            try:
                return json.loads(s)
            except Exception:
                # 若不是合法 JSON，原样返回字符串，避免报错
                return s

        return pd.Series({
            "date": row.date,
            "weights": _loads_or_none(row.weights),
            "weights_ewma": _loads_or_none(row.weights_ewma),
        })

from app.data.helper import get_fund_current_prices_by_code_list
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
import pandas as pd

def query_price_by_codes(codes: list[str]) -> dict[str, float]:
    today = pd.Timestamp.today()
    trade_dates = TradeCalendarReader.get_trade_dates(end=today.strftime("%Y-%m-%d"))
    if len(trade_dates) < 2:
        raise ValueError("交易日不足")

    t1_date = trade_dates[-2]
    t1_str = t1_date.strftime("%Y-%m-%d")
    t_date = trade_dates[-1]
    t_str = t_date.strftime("%Y-%m-%d")

    price_df = get_fund_current_prices_by_code_list(codes, start_date=t1_str, end_date=t_str)
    if price_df.empty:
        return {}

    prices = price_df.iloc[-1].to_dict()
    return {code: round(float(prices.get(code, 0)), 6) for code in codes}

from app.models.service_models import CurrentHolding
from app.database import get_db

def query_current_position() -> list[dict]:
    """
    查询当前持仓，包括 asset, code, name, amount, price（由行情接口补充）
    """
    with get_db() as db:
        rows = db.query(CurrentHolding).all()

    code_list = [row.code for row in rows]
    price_map = query_price_by_codes(code_list)

    result = []
    for row in rows:
        result.append({
            "asset": row.asset,
            "code": row.code,
            "name": row.name,
            "amount": row.amount,
            "price": price_map.get(row.code, 0)
        })
    return result

def generate_reallocation_ops(codes: list[str], from_value: list[float], to_value: list[float], price: list[float]) -> list[dict]:
    """
    基于当前 value、目标 value 和 price，生成调仓操作转换指令。
    贪心策略：将需卖出的资产拆分为若干卖→买操作。

    参数:
        codes: 资产唯一标识（如基金代码）
        from_value: 当前市值
        to_value: 目标市值
        price: 当前价格

    返回:
        List[Dict]，每项形如 { from: code_from, to: code_to, amount_from: 份额 }
    """
    df = pd.DataFrame({
        "code": codes,
        "delta": [t - f for f, t in zip(from_value, to_value)],
        "price": price
    })

    sellers = df[df["delta"] < -1e-6].copy()
    buyers = df[df["delta"] > 1e-6].copy()

    sellers["amount"] = -sellers["delta"]
    buyers["amount"] = buyers["delta"]

    transfers = []

    for s_idx, s_row in sellers.iterrows():
        s_code = s_row["code"]
        s_price = s_row["price"]
        s_amount = s_row["amount"]

        for b_idx, b_row in buyers.iterrows():
            b_code = b_row["code"]
            b_price = b_row["price"]
            b_amount = b_row["amount"]

            if b_amount <= 1e-4:
                continue

            transfer_value = min(s_amount, b_amount)
            amount_from = transfer_value / s_price

            transfers.append({
                "from": s_code,
                "to": b_code,
                "amount": round(amount_from, 4)
            })

            s_amount -= transfer_value
            buyers.at[b_idx, "amount"] -= transfer_value

            if s_amount <= 1e-4:
                break

    return transfers

from app.models.service_models import RebalanceLog
from app.database import get_db
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
import pandas as pd

def save_transfer_logs(transfers: list[dict]) -> None:
    """
    保存调仓转换操作记录到数据库（按最近一个交易日）
    """
    trade_dates = TradeCalendarReader.get_trade_dates(end=pd.Timestamp.today().strftime("%Y-%m-%d"))
    if trade_dates.empty:
        raise ValueError("无法获取交易日")

    today = trade_dates[-1]

    with get_db() as db:
        row = RebalanceLog(date=today, operations=transfers)
        db.merge(row)
        db.commit()

from datetime import datetime
from sqlalchemy.orm import Session
from app.models.service_models import CurrentHolding
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
from app.database import get_db


def save_current_holdings(new_holdings: list[dict]) -> None:
    """
    保存目标持仓至 current_holdings 表。入库为最新交易日。
    每行包括 asset, code, name, amount。
    """
    trade_dates = TradeCalendarReader.get_trade_dates(end=datetime.today().strftime("%Y-%m-%d"))
    if trade_dates.empty:
        raise ValueError("无法获取交易日")
    today = trade_dates[-1]

    with get_db() as db:
        # 先删除该交易日已有记录
        db.query(CurrentHolding).delete()

        for row in new_holdings:
            record = CurrentHolding(
                asset=row["asset"],
                code=row["code"],
                name=row["name"],
                amount=row["amount"],
            )
            db.add(record)
        db.commit()

def query_latest_transfer() -> list[dict]:
    """
    查询最近一次调仓的操作记录
    """
    from app.models.service_models import RebalanceLog
    from sqlalchemy import desc

    with get_db() as db:
        row = db.query(RebalanceLog).order_by(desc(RebalanceLog.date)).first()
        if row and row.operations:
            return row.operations
        return []

from app.dao.fund_info_dao import FundInfoDao
from typing import List, Dict

def query_fund_names_by_codes(codes: List[str]) -> Dict[str, str]:
    """
    根据基金代码列表，查询对应名称
    :param codes: list[str]
    :return: dict[code] = name
    """
    df = FundInfoDao._instance.select_dataframe_by_code(codes)
    df = df.dropna(subset=["fund_code", "fund_name"])
    return dict(zip(df["fund_code"], df["fund_name"]))