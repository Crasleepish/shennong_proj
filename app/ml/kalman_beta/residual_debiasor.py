from collections import deque
import numpy as np

class ResidualDebiasor:
    """
    有状态的一侧去偏器（支持 EWMA / 窗口均值），
    但无需持久化：可通过历史回放重建到 ref_date 前一刻的内部状态。
    """
    def __init__(self, method: str = "ewma", window: int = 60, lam: float = 0.98, trim_p: float = 0.02):
        assert method in ("ewma", "window")
        self.method = method
        self.window = int(window)
        self.lam = float(lam)
        self.trim_p = float(trim_p)

        self.buf = deque(maxlen=self.window)
        self._ewma = 0.0
        self._inited = False

    @staticmethod
    def _winsorize(x: np.ndarray, p: float):
        if len(x) == 0 or p <= 0:
            return x
        lo, hi = np.quantile(x, [p/2, 1 - p/2])
        return np.clip(x, lo, hi)

    def mean_bias_prev(self) -> float:
        """
        返回“上一期”的偏差估计 \hat{b}_{t-1}。
        - 对 EWMA：返回当前 ewma（如果还未初始化则 0）
        - 对窗口：返回当前窗口均值（winsorize 后）
        """
        if self.method == "ewma":
            return float(self._ewma) if self._inited else 0.0
        else:
            if len(self.buf) == 0:
                return 0.0
            x = np.asarray(self.buf, dtype=float)
            x = self._winsorize(x, self.trim_p)
            return float(np.mean(x))

    def update_with_residual(self, resid_t: float):
        """
        用“本期预测残差” ε_t 更新内部状态，得到新的 \hat{b}_t。
        注意：调用顺序应是：
          先取 mean_bias_prev() 给当期用，再用当期 ε_t 调用本函数更新。
        """
        r = float(resid_t)
        if self.method == "ewma":
            if not self._inited:
                self._ewma = r
                self._inited = True
            else:
                self._ewma = self.lam * self._ewma + (1.0 - self.lam) * r
        else:
            self.buf.append(r)

import pandas as pd
import numpy as np

# 这些 import 与你的项目一致（请按你的真实包路径替换）
from app.dao.betas_dao import FundBetaDao
from app.dao.stock_info_dao import MarketFactorsDao
from app.data.helper import get_fund_daily_return_for_beta_regression as get_fund_daily_return
from app.data_fetcher.trade_calender_reader import TradeCalendarReader as CalendarFetcher

FACTOR_NAMES = ["MKT", "SMB", "HML", "QMJ"]

def _bootstrap_debiasor_from_history(
    fund_code: str,
    ref_date: str,
    *,
    method: str = "ewma",
    window_size: int = 60,
    lam: float = 0.98,
    trim_p: float = 0.02,
) -> ResidualDebiasor:
    """
    使用 ref_date 之前的交易日历史（严格 < ref_date），
    以 2*window_size 的缓冲回放“预测残差 ε_t = y_t - x_t^T z_{t-1}”，
    重建去偏器的内部状态（\hat{b}_{t-1}）。

    返回：一个已就绪的 ResidualDebiasor 实例（可直接在 ref_date 当天使用 mean_bias_prev()）
    """

    # 1) 拿交易日列表（< ref_date），取最多 2*window_size+1 个
    trade_dates = CalendarFetcher().get_trade_date(
        start="19900101",
        end=ref_date.replace("-", ""),
        format="%Y-%m-%d",
        limit=(2 * window_size + 1),
        ascending=False
    )
    trade_dates = [td for td in trade_dates if pd.to_datetime(td) < pd.to_datetime(ref_date)]
    if not trade_dates:
        return ResidualDebiasor(method=method, window=window_size, lam=lam, trim_p=trim_p)

    trade_dates.sort()
    start_hist = trade_dates[0]
    end_hist = trade_dates[-1]

    # 2) 读取因子与基金日收益，并对齐
    factor_df = MarketFactorsDao.select_dataframe_by_date(start_hist, end_hist).set_index("date")
    fund_df = get_fund_daily_return(fund_code, start_hist, end_hist)  # 返回包含 "date","daily_return"
    fund_df = fund_df.set_index("date")

    df = factor_df.join(fund_df, how="inner").sort_index()
    if df.empty:
        return ResidualDebiasor(method=method, window=window_size, lam=lam, trim_p=trim_p)

    # 3) 读取历史 β（posterior），要求至少覆盖 [start_hist, end_hist] 的前一日
    betas_df = FundBetaDao.select_all_by_code_date(fund_code, start_hist, end_hist)  # 你项目里的真实方法名可能不同
    if betas_df is None or len(betas_df) == 0:
        return ResidualDebiasor(method=method, window=window_size, lam=lam, trim_p=trim_p)

    betas_df = betas_df.sort_values("date")
    z_by_date = {}
    for _, r in betas_df.iterrows():
        # 与你入库时的命名保持一致：MKT, SMB, HML, QMJ, const
        z_by_date[pd.to_datetime(r["date"])] = np.array([r["MKT"], r["SMB"], r["HML"], r["QMJ"], r["const"]], dtype=float).reshape(-1, 1)

    # 4) 组装 (x_t, y_t) 并以 z_{t-1} 计算预测残差 ε_t
    #    注意：要保证 t-1 有 β（posterior），否则该 t 跳过
    df["intercept"] = 1.0
    dates = df.index.to_list()

    # 我们只用最后 2*window_size 个样本进行“回放”
    if len(dates) > 2 * window_size:
        dates = dates[-(2 * window_size):]

    debiasor = ResidualDebiasor(method=method, window=window_size, lam=lam, trim_p=trim_p)

    prev_date = None
    for d in dates:
        # 需要 z_{t-1}：如果没有，上一条样本无法计算“预测残差”，跳过
        if prev_date is None or prev_date not in z_by_date:
            prev_date = d
            continue

        row = df.loc[d]
        x = np.array([row[f] for f in FACTOR_NAMES] + [1.0], dtype=float).reshape(-1, 1)
        y = float(row["daily_return"])
        z_prev = z_by_date[prev_date]  # posterior at t-1

        y_pred_t_given_prev = float((x.T @ z_prev).ravel()[0])
        eps_t = y - y_pred_t_given_prev

        # 一侧协议：当前期使用的是 hat{b}_{t-1}，然后再用 ε_t 更新到 hat{b}_t
        debiasor.update_with_residual(eps_t)

        prev_date = d

    return debiasor
