from app.models.service_models import PortfolioWeights
from app.database import get_db
import json
import pandas as pd
import numpy as np
from app.utils.cov_packer import pack_covariance, unpack_covariance
import logging

logger = logging.getLogger(__name__)

def query_weights_by_date(date: str, portfolio_id: int) -> dict:
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

        if row is None or not row.codes or not row.weights_ewma:
            return { "weights": {} }

        try:
            codes = json.loads(row.codes)
            weights = json.loads(row.weights_ewma)
            weights_dict = dict(zip(codes, weights))
        except Exception:
            weights_dict = {}

        return { "weights": weights_dict }
    

from typing import Optional
from sqlalchemy import desc

def query_latest_portfolio_by_id(portfolio_id: int, as_of_date: str = None) -> pd.Series:
    """
    查询指定 portfolio_id 最新一条组合权重记录
    返回: pd.Series，包含 fields: date, weights, weights_ewma
    - weights / weights_ewma: 若为 JSON 字符串则解析为 dict，若为空返回 None
    - 若查无记录，返回空 Series
    """
    with get_db() as db:
        condition = [PortfolioWeights.portfolio_id == portfolio_id]
        if as_of_date:
            condition.append(PortfolioWeights.date < as_of_date)
        row: Optional[PortfolioWeights] = (
            db.query(PortfolioWeights)
              .filter(*condition)
              .order_by(desc(PortfolioWeights.date))
              .first()
        )

        if not row:
            return pd.Series(dtype=object)

        def _loads_or_none(c: Optional[str], w: Optional[str]):
            if not c or not w:
                return None
            try:
                codes = json.loads(c)
                weights = json.loads(w)
                d = dict(zip(codes, weights))
                return d
            except Exception:
                # 若不是合法 JSON，原样返回字符串，避免报错
                return "invalid json"

        return pd.Series({
            "date": row.date,
            "weights": _loads_or_none(row.codes, row.weights),
            "weights_ewma": _loads_or_none(row.codes, row.weights_ewma),
        })
    
def query_cov_matrix_by_date(date: str, portfolio_id: int = 1) -> np.ndarray:
    """
    :param date: 形如 '2024-07-20'
    :param portfolio_id: 默认组合ID为 1
    :return: codes, numpy.ndarray
    """
    query_date = pd.to_datetime(date).date()

    with get_db() as db:
        row = db.query(PortfolioWeights).filter_by(
            portfolio_id=portfolio_id,
            date=query_date
        ).first()

        if row is None or not row.codes or not row.weights_ewma:
            return None

        try:
            codes = json.loads(row.codes)
            cov_matrix = unpack_covariance(row.cov_matrix, row.cov_meta)
        except Exception:
            logger.error(f"fail to fetch cov matrix on {date}")
            cov_matrix = None

        return codes, cov_matrix

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

def query_current_position(portfolio_id: int) -> list[dict]:
    """
    查询当前持仓，包括 asset, code, name, amount, price（由行情接口补充）
    """
    with get_db() as db:
        rows = db.query(CurrentHolding).filter(CurrentHolding.portfolio_id == portfolio_id).all()

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


def save_current_holdings(portfolio_id: int, new_holdings: list[dict]) -> None:
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
        db.query(CurrentHolding).filter(CurrentHolding.portfolio_id == portfolio_id).delete()

        for row in new_holdings:
            record = CurrentHolding(
                asset=row["asset"],
                code=row["code"],
                name=row["name"],
                amount=row["amount"],
                portfolio_id=portfolio_id,
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

def store_portfolio(
    portfolio_id: int,
    trade_date: str,
    weights_raw: Dict[str, float],
    weights_ewma: Dict[str, float],
    cov_matrix: np.ndarray,
    codes: List[str],
    eps: float = 1e-4,
) -> None:
    """
    将组合信息入库，并在入库前做“同步清理”：
    若第 i 个资产满足：
        |weights_today[i]| < eps 且 |weights_smooth[i]| < eps 且 |cov_matrix[i,i]| < eps
    则删除该资产（即同时从 codes / weights / 协方差矩阵中移除该位置），
    且必须保证 **相对顺序** 不变。

    最终入库的 weights / weights_ewma / cov_matrix 的顺序与清理后的 codes 完全一致。

    参数
    ----
    portfolio_id : int
    trade_date   : 'YYYY-MM-DD'
    weights_raw  : dict[code] -> weight
    weights_ewma : dict[code] -> weight
    cov_matrix   : np.ndarray  (方阵，形状与 codes 对应)
    codes        : List[str]   (与 cov_matrix 的行列顺序一致)
    eps          : float       (近零阈值)
    """

    # 1) 按 codes 顺序构造两条权重向量
    w_today  = np.array([weights_raw.get(c, 0.0)  for c in codes], dtype=float)
    w_smooth = np.array([weights_ewma.get(c, 0.0) for c in codes], dtype=float)

    # 基础校验：cov 矩阵维度应与 codes 长度一致
    n = len(codes)
    if cov_matrix.shape != (n, n):
        raise ValueError(f"cov_matrix shape {cov_matrix.shape} does not match codes length {n}")

    # 2) 计算要保留的索引（满足“非同时近零”）
    diag = np.diag(cov_matrix)
    keep_mask = ~(
        (np.abs(w_today)  < eps) &
        (np.abs(w_smooth) < eps) &
        (np.abs(diag)     < eps)
    )
    # 保留索引，保持原始顺序
    keep_idx = np.nonzero(keep_mask)[0].tolist()

    # 3) 若全部被过滤，容错：此处可根据业务选择报错或保留最小集
    if len(keep_idx) == 0:
        logger.warning(
            f"[store_portfolio] All assets filtered out by eps={eps}. "
            f"portfolio_id={portfolio_id}, date={trade_date}. "
            f"Will store empty codes & empty cov."
        )
        codes_kept = []
        weights_today_kept = []
        weights_smooth_kept = []
        cov_kept = np.zeros((0, 0), dtype=np.float32)
    else:
        # 4) 按同一 keep_idx 子集化，严格保持相对顺序一致
        codes_kept = [codes[i] for i in keep_idx]
        weights_today_kept  = w_today[keep_idx].tolist()
        weights_smooth_kept = w_smooth[keep_idx].tolist()
        cov_kept = cov_matrix[np.ix_(keep_idx, keep_idx)]

    # 5) 协方差矩阵仅保存下三角（float32 + BLOB），并存入 meta
    cov_blob, meta_json = pack_covariance(cov_kept.astype(np.float32, copy=False))

    # 6) 入库（merge upsert）
    with get_db() as db:
        new_row = PortfolioWeights(
            portfolio_id=portfolio_id,
            date=pd.Timestamp(trade_date),
            codes=json.dumps(codes_kept, ensure_ascii=False),
            weights=json.dumps(weights_today_kept, ensure_ascii=False),
            weights_ewma=json.dumps(weights_smooth_kept, ensure_ascii=False),
            cov_matrix=cov_blob,
            cov_meta=meta_json,
        )
        db.merge(new_row)
        db.commit()

    logger.info(
        f"✅ store_portfolio done | pid={portfolio_id} date={trade_date} "
        f"kept={len(codes_kept)}/{len(codes)} eps={eps}"
    )

from typing import Optional, Dict, List
import pandas as pd
import json
from sqlalchemy import asc

def query_all_weights_by_date(
    portfolio_id: int,
    start_date: str,
    end_date: str,
    *,
    fill_missing_zero: bool = True
) -> pd.DataFrame:
    """
    批量查询某个组合在给定时间段内的“平滑后权重”（weights_ewma）。

    参数
    ----
    portfolio_id : int
        组合 ID
    start_date : str
        起始日期（含），格式 'YYYY-MM-DD'
    end_date : str
        截止日期（含），格式 'YYYY-MM-DD'
    fill_missing_zero : bool, default True
        是否用 0.0 填充缺失资产权重（列并齐后产生的 NaN）

    返回
    ----
    pd.DataFrame
        行索引为 date（升序），列为资产代码，单元为对应日期的平滑后权重。
        若时间段内无数据，返回空 DataFrame。
    """
    if not start_date or not end_date:
        raise ValueError("start_date 与 end_date 不能为空")

    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()

    with get_db() as db:
        rows: List[PortfolioWeights] = (
            db.query(PortfolioWeights)
              .filter(
                  PortfolioWeights.portfolio_id == portfolio_id,
                  PortfolioWeights.date >= start_dt,
                  PortfolioWeights.date <= end_dt,
              )
              .order_by(asc(PortfolioWeights.date))
              .all()
        )

    if not rows:
        return pd.DataFrame()

    # 将每一行记录转换为 date -> {code: weight} 的映射
    by_date: Dict[pd.Timestamp, Dict[str, float]] = {}
    for row in rows:
        if not row.codes or not row.weights_ewma:
            continue
        try:
            codes = json.loads(row.codes)
            weights = json.loads(row.weights_ewma)
            # 长度不一致时做安全处理：截断到最短长度
            n = min(len(codes), len(weights))
            d = dict(zip(codes[:n], (float(w) for w in weights[:n])))
            by_date[pd.to_datetime(row.date)] = d
        except Exception:
            # 若 JSON 非法或解析异常，跳过该日
            continue

    if not by_date:
        return pd.DataFrame()

    # 组装为 DataFrame：index=日期，columns=资产代码
    df = pd.DataFrame.from_dict(by_date, orient="index").sort_index()
    df.index.name = "date"

    if fill_missing_zero:
        df = df.fillna(0.0)

    return df