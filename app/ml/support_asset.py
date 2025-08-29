# app/ml/support_asset.py

import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from app.dao.betas_dao import FundBetaDao
from app.ml.greedy_convex_hull_volume import select_representatives
from app.utils.cov_packer import unpack_covariance

logger = logging.getLogger(__name__)

FACTOR_COLS = ["MKT", "SMB", "HML", "QMJ"]  # 仅用前四个因子作为 X
P_VAR_SUM_THRESH = 0.2                      # diag(P[:4,:4]) 之和阈值
CONST_THRESH = -0.005


def _load_latest_betas_asof(trade_date: str) -> pd.DataFrame:
    """
    截至 trade_date，获取每只基金最近一条因子暴露记录（包含 P_json）。
    允许不同环境下 DAO 方法名不同：尝试多种入口，缺失时自动回退。
    期望输出列至少包含：fund_code, date, MKT, SMB, HML, QMJ, P_json
    """
    dao = FundBetaDao

    # 1) 优先尝试“按日期就近取最近一条”的专用方法（若存在）
    for meth in ("get_latest_fund_betas"):
        if hasattr(dao, meth):
            df = getattr(dao, meth)(trade_date)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df

    fund_type_list = ["股票型"]
    invest_type_list = ["被动指数型", "增强指数型"]
    one_year_ago = (pd.to_datetime(trade_date) - pd.DateOffset(years=1)).date().strftime('%Y-%m-%d')
    df = dao.get_latest_fund_betas(fund_type_list, invest_type_list, one_year_ago, trade_date)

    if df is None or df.empty:
        logger.error(f"获取因子暴露记录失败，未获取到{trade_date}之前的因子暴露。")
        return pd.DataFrame()

    # 规范列类型
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


def _stable_mask_and_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    根据 P_json 的稳定性条件筛选样本：
      - 解析 P_json -> P (5x5 或以上)，取 P[:4,:4] 的对角线和 < 0.2
      - 同时保证前四个因子暴露为有限数
    返回：
      mask: bool 数组（保留为 True）
      X:    shape (n_keep, 4) 的因子暴露矩阵（MKT,SMB,HML,QMJ）
      codes: 保留样本的 fund_code 列表（与 X 行对齐）
    """
    if df is None or df.empty:
        return np.array([], dtype=bool), np.empty((0, 4)), []
    df = df.dropna(how="any")

    # 提前准备暴露矩阵和代码
    missing_cols = [c for c in FACTOR_COLS + ["code", "P_bin"] if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"FundBeta 数据缺少必要列: {missing_cols}")

    X_full = df[FACTOR_COLS].to_numpy(dtype=float)
    codes = df["code"].astype(str).tolist()

    keep_mask = []
    for i, row in df.iterrows():
        try:
            P = unpack_covariance(row["P_bin"], "{\"dtype\": \"float32\", \"n\": 5}")
            const = row["const"]
            if isinstance(P, str):
                P = json.loads(P)
            P = np.array(P, dtype=float)

            if P.ndim != 2 or P.shape[0] < 4 or P.shape[1] < 4:
                keep_mask.append(False)
                continue

            # 稳定性过滤：diag(P[:4,:4]) 之和 < 阈值
            diag_sum = float(np.trace(P[:4, :4]))
            if not np.isfinite(diag_sum):
                keep_mask.append(False)
                continue

            # 暴露有限性检查
            x = X_full[len(keep_mask)]
            if not np.all(np.isfinite(x)):
                keep_mask.append(False)
                continue

            keep_mask.append(diag_sum < P_VAR_SUM_THRESH and const > CONST_THRESH)

        except Exception:
            keep_mask.append(False)

    keep_mask = np.array(keep_mask, dtype=bool)
    X = X_full[keep_mask]
    kept_codes = [c for c, m in zip(codes, keep_mask) if m]

    return keep_mask, X, kept_codes


def find_support_assets(
    trade_date: str,
    *,
    epsilon: float = 0.03,
    M: int = 4096,
    topk_per_iter: int = 32,
    debug: bool = False,
) -> List[str]:
    """
    主入口：从全市场基金中筛选“支撑资产”，返回 fund_code 列表（按加入顺序）。
    步骤：
      1) 截至 trade_date 获取每只基金最近一条因子暴露
      2) 用 P_json 的前4×4 对角线和 < 0.2 做稳定性过滤
      3) 用筛后的 4维暴露 (MKT, SMB, HML, QMJ) 调用 select_representatives

    参数
    ----
    trade_date : 截止日期（YYYY-MM-DD）
    epsilon    : select_representatives 的逼近精度
    M          : 方向采样数（建议 8192）
    topk_per_iter : 每轮最多加入的元素数（建议 64）
    debug, logger : 传给底层以打印迭代日志

    返回
    ----
    选中资产的 fund_code 列表（按加入顺序）。若无可选，返回空列表。
    """
    logger.info("开始筛选支撑资产: trade_date=%s, epsilon=%s, M=%s, topk_per_iter=%s", trade_date, epsilon, M, topk_per_iter)
    df = _load_latest_betas_asof(trade_date)
    if df is None or df.empty:
        logger and logger.warning("截至 %s 未获取到任何基金的因子暴露记录。", trade_date)
        return []

    mask, X, codes = _stable_mask_and_matrix(df)
    if X.shape[0] == 0:
        logger and logger.warning("满足稳定性条件的基金为空（P[:4,:4] 对角线和 < %.4f）。", P_VAR_SUM_THRESH)
        return []

    # 直接调用贪心凸包体积代表点选择
    try:
        idx = select_representatives(
            X=X,
            epsilon=epsilon,
            M=M,
            topk_per_iter=topk_per_iter,
            debug=debug,
        )
    except Exception as e:
        logger and logger.exception("select_representatives 调用失败：%s", e)
        return []

    if idx is None or len(idx) == 0:
        return []

    # idx 是基于“传入 X”的行号；映射回保留后的 codes
    selected_codes = [codes[i] for i in idx if 0 <= i < len(codes)]
    return selected_codes

def generate_source_map_and_factors_map(factor_codes: List[str]):
    asset_source_map = {c: 'factor' for c in factor_codes}
    code_factors_map = {c: ["MKT", "SMB", "HML", "QMJ"] for c in factor_codes}
    return asset_source_map, code_factors_map