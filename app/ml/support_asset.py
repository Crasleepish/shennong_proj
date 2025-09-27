# app/ml/support_asset.py

import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from app.dao.betas_dao import FundBetaDao
from app.ml.greedy_convex_hull_volume import select_representatives
from app.utils.cov_packer import unpack_covariance
from app.data_fetcher import CalendarFetcher

logger = logging.getLogger(__name__)

FACTOR_COLS = ["MKT", "SMB", "HML", "QMJ"]  # 仅用前四个因子作为 X
P_VAR_SUM_THRESH = 0.08                      # diag(P[:4,:4]) 之和阈值
CONST_VAR_THRESH = 3e-5                     # P[4,4]阈值
CONST_THRESH = -0.0005


def _load_latest_betas_asof(trade_date: str, period: int = 60) -> pd.DataFrame:
    """
    截至 trade_date，获取每只基金最近 period 个“交易日”的平均因子暴露，
    并将窗口内 P_bin 解包后逐元素平均为 P（np.ndarray）。

    返回列：
        fund_code, MKT, SMB, HML, QMJ, const, P   （P 为 np.ndarray，可能为 None）

    说明：
    - 窗口为交易日口径：对每只基金按日期排序后直接 tail(period)。
    - P 的解包使用固定 meta（示例：{"dtype":"float32","n":6}）。
    - 若某基金窗口内 P_bin 全为空或解包失败，P 返回 None。
    """
    dao = FundBetaDao
    fund_type_list = ["股票型"]
    invest_type_list = ["被动指数型", "增强指数型"]
    one_year_ago = (pd.to_datetime(trade_date) - pd.DateOffset(years=1)).date().strftime('%Y-%m-%d')
    start_date = CalendarFetcher().get_trade_date(start="19900101", end=trade_date.replace("-", ""), format="%Y-%m-%d", limit=period, ascending=False)[-1]
    df = dao.get_all_betas_by_type_cond(fund_type_list, invest_type_list, one_year_ago, trade_date, start_date)

    if df is None or df.empty:
        logger.error(f"获取因子暴露记录失败，未获取到{trade_date}之前的因子暴露。")
        return pd.DataFrame()

    # 先 dropna(any)，再剔除“最新日期 < trade_date”的清盘资产 ===
    df = df.dropna(how="any").copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    trade_dt = pd.to_datetime(trade_date)
    # 每只资产的最近日期
    last_date = df.groupby("fund_code")["date"].max()
    alive_codes = last_date[last_date >= trade_dt].index  # 仅保留“未清盘”的资产
    df = df[df["fund_code"].isin(alive_codes)]

    df = df.sort_values(["fund_code", "date"]).reset_index(drop=True)

    df_last = df.groupby("fund_code", group_keys=False).tail(period)
    fac_cols = ["MKT", "SMB", "HML", "QMJ", "const"]
    expo_mean = (
        df_last
        .groupby("fund_code")[fac_cols]
        .mean()
        .astype(float)
    )

    # P_bin：解包→逐元素平均→再打包（二进制）
    def _avg_P(g: pd.Series):
        mats = []
        for b in g.dropna():
            try:
                # 按你的存储元信息解包；样例 meta：{"dtype":"float32","n":6}
                P = unpack_covariance(b, "{\"dtype\": \"float32\", \"n\": 6}")
                mats.append(np.asarray(P, dtype=np.float64))
            except Exception:
                # 单条坏数据忽略
                continue
        if not mats:
            return None
        return np.mean(mats, axis=0)  # np.ndarray
    
    P_avg = (
        df_last.groupby("fund_code")["P_bin"]
        .apply(_avg_P)
        .to_frame(name="P")
    )

    res = (
        expo_mean.join(P_avg, how="left")
        .reset_index()  # fund_code 回到列
    )
    res = res[["fund_code", "MKT", "SMB", "HML", "QMJ", "const", "P"]] \
            .sort_values("fund_code") \
            .reset_index(drop=True)
    if res.empty:
        logger.warning(f"未能生成任何基金的平均因子暴露，trade_date={trade_date}, period={period}")

    return res


def _stable_mask_and_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    根据已解包的 P (6x6) 的稳定性条件筛选样本：
      - 取 P[:4,:4] 的对角线和 < P_VAR_SUM_THRESH
      - P[4,4] < CONST_VAR_THRESH
      - const > CONST_THRESH
      - 同时保证前四个因子暴露为有限数
    返回：
      mask:  bool 数组（保留为 True）
      X:     shape (n_keep, 4) 的因子暴露矩阵（MKT, SMB, HML, QMJ）
      codes: 保留样本的 fund_code 列表（与 X 行对齐）
    """
    if df is None or df.empty:
        return np.array([], dtype=bool), np.empty((0, 4)), []

    # 提前准备暴露矩阵和代码
    id_col = "code" if "code" in df.columns else ("fund_code" if "fund_code" in df.columns else None)
    if id_col is None:
        raise RuntimeError("FundBeta 数据缺少标识列：需要 'code' 或 'fund_code'")
    missing_cols = [c for c in (FACTOR_COLS + [id_col, "P", "const"]) if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"FundBeta 数据缺少必要列: {missing_cols}")
    
    df = df.dropna(subset=FACTOR_COLS + ["P"]).copy()

    X_full = df[FACTOR_COLS].to_numpy(dtype=float)
    codes = df[id_col].astype(str).tolist()

    keep_mask = []
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            P = row["P"]
            if isinstance(P, str):
                P = json.loads(P)
            P = np.array(P, dtype=float)

            if P.ndim != 2 or P.shape[0] < 5 or P.shape[1] < 5:
                keep_mask.append(False)
                continue

            # 稳定性过滤：diag(P[:4,:4]) 之和 < 阈值
            diag_sum = float(np.trace(P[:4, :4]))
            if not np.isfinite(diag_sum):
                keep_mask.append(False)
                continue
            const_var = float(P[4, 4])

            # 暴露有限性检查
            x = X_full[i]
            if not np.all(np.isfinite(x)):
                keep_mask.append(False)
                continue

            # const 阈值（需要列存在）
            const_val = float(row["const"]) if pd.notna(row["const"]) else -np.inf

            keep_mask.append(
                (diag_sum < P_VAR_SUM_THRESH) and
                (const_var < CONST_VAR_THRESH) and
                (const_val > CONST_THRESH)
            )

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
      1) 截至 trade_date 获取每只基金最近 period 个交易日的“平均因子暴露”（_load_latest_betas_asof）
      2) 用解包后的 P 做稳定性过滤（如：P[:4,:4] 对角线和、P[4,4]、const 阈值）
      3) 用筛后的 4 维暴露矩阵 X (MKT, SMB, HML, QMJ) 调用 select_representatives

    参数
    ----
    trade_date      : 截止日期（YYYY-MM-DD）
    epsilon         : select_representatives 的逼近精度
    M               : 方向采样数（建议 8192）
    topk_per_iter   : 每轮最多加入的元素数（建议 64）
    debug           : 打印迭代日志

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
        logger.warning("满足稳定性条件的基金为空（P[:4,:4] 对角线和 < %.4f）。", P_VAR_SUM_THRESH)
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
