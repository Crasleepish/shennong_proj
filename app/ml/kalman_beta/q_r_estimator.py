import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from collections import deque
from app.data_fetcher.calender_fetcher import CalendarFetcher
from app.dao.stock_info_dao import MarketFactorsDao
from app.dao.betas_dao import FundBetaDao
from app.data.helper import get_fund_daily_return_for_beta_regression, get_etf_daily_return_for_beta_regression

FACTOR_NAMES = ["MKT", "SMB", "HML", "QMJ"]


class QREstimator:
    def __init__(self, window_size=60, base_dim=5, winsor_p=0.035, q_floor=1e-6, q_init=1e-4, r_init=1e-4, q_shrink=0.5):
        self.window_size = int(window_size)
        self.base_dim = int(base_dim)      # 应为 5：4 因子 + 截距
        self.winsor_p = float(winsor_p)    # 残差剪尾比例（稳健 R）
        self.q_floor = float(q_floor)      # Q 的最小对角下限（收缩防数值问题）
        self.q_init = float(q_init)        # 历史不足时的退化 Q
        self.r_init = float(r_init)        # 历史不足时的退化 R
        self.q_shrink = float(q_shrink)

        self.X_window = deque(maxlen=self.window_size)
        self.y_window = deque(maxlen=self.window_size)
        self.beta_history = deque(maxlen=self.window_size)
        self.prev_beta = None

    @staticmethod
    def _robust_var(x, p=0.02):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return 0.0
        if p <= 0:
            return float(np.var(x, ddof=1)) if x.size > 1 else 0.0
        lo, hi = np.quantile(x, [p/2, 1-p/2])
        xr = np.clip(x, lo, hi)
        return float(np.var(xr, ddof=1)) if xr.size > 1 else 0.0

    def update_data(self, X_row, y):
        """
        X_row: shape (1, base_dim) 的行向量（只含基础特征，不含 TE）
        y: 标量（日收益）
        """
        X_row = np.asarray(X_row, dtype=float)
        assert X_row.shape == (1, self.base_dim), f"X_row shape {X_row.shape} != (1,{self.base_dim})"
        self.X_window.append(X_row)
        self.y_window.append(float(y))

    def estimate(self):
        """
        返回：
          Q: (base_dim, base_dim)
          R: float
        """
        # 历史不足：退化
        if len(self.X_window) < self.window_size:
            return np.eye(self.base_dim) * self.q_init, self.r_init

        X = np.vstack(self.X_window)                      # (W, base_dim)
        y = np.asarray(self.y_window, dtype=float)        # (W,)

        # OLS（不含截距，因为 X 的最后一列已经是 1）
        model = LinearRegression(fit_intercept=False).fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred

        # 稳健观测方差
        R = self._robust_var(residuals, p=self.winsor_p)
        if not np.isfinite(R) or R <= 0:
            R = self.r_init

        # 用相邻窗口的 beta 变化估计 Q（经验协方差）
        current_beta = model.coef_.reshape(-1)            # (base_dim,)
        if self.prev_beta is not None:
            delta_beta = current_beta - self.prev_beta
            self.beta_history.append(delta_beta)
        self.prev_beta = current_beta

        if len(self.beta_history) < max(6, self.base_dim + 1):
            Q = np.eye(self.base_dim) * self.q_init
        else:
            D = np.vstack(self.beta_history)              # (K, base_dim)
            # 协方差需要转置：np.cov 期望变量为行或列，根据 rowvar 决定
            Q_emp = np.cov(D, rowvar=False)               # (base_dim, base_dim)

            # 数值稳定处理：对称化 + 对角下限 + 轻度收缩
            Q_emp = 0.5 * (Q_emp + Q_emp.T)
            # 对角最小下限
            diag = np.clip(np.diag(Q_emp), self.q_floor, None)
            Q = Q_emp.copy()
            np.fill_diagonal(Q, diag)
            # 轻度收缩到对角阵，避免近奇异
            shrink = 0.1
            Q = (1 - shrink) * Q + shrink * np.eye(self.base_dim) * np.mean(diag)

        return Q * self.q_shrink, R

def _bootstrap_qr_from_history(code: str, ref_date: str, window_size: int = 60, asset_type: str = "fund_info") -> QREstimator:
    """
    使用 ref_date 之前的最近 window_size 个【交易日】数据，重建 QREstimator 的内部状态：
    - 填满 X_window / y_window
    - 用窗口 OLS 显式设定 prev_beta（长度=5，含截距）
    若历史不足 window_size，则返回一个空窗的 QREstimator（按其内部冷启动退化逻辑工作）。
    """
    # 1) 找到 ref_date 之前的交易日列表（严格小于 ref_date），取2 * window_size 个交易日
    trade_dates = CalendarFetcher().get_trade_date(start="19900101", end=ref_date.replace("-", ""), format="%Y-%m-%d", limit=(2 * window_size + 1), ascending=False)
    trade_dates = [td for td in trade_dates if pd.to_datetime(td) < pd.to_datetime(ref_date)]
    if len(trade_dates) == 0:
        return QREstimator(window_size=window_size)

    # 取最后 2 * window_size 个交易日作为窗口
    trade_dates.sort()
    start_hist = trade_dates[0]
    end_hist = trade_dates[-1]

    # 2) 拉取因子 & 基金收益，并对齐
    factor_df = MarketFactorsDao._instance.select_dataframe_by_date(start_hist, end_hist).set_index("date")
    if asset_type == "fund_info":
        ret_df = get_fund_daily_return_for_beta_regression(code, start_hist, end_hist)
    elif asset_type == "etf_info":
        ret_df = get_etf_daily_return_for_beta_regression(code, start_hist, end_hist)
    else:
        raise ValueError(f"Unknown asset_type: {asset_type}")

    df_hist = factor_df.join(ret_df).sort_index()
    df_hist = df_hist.dropna(how="any")
    if df_hist.empty:
        # （没有历史数据让 QREstimator 自己走冷启动）
        return QREstimator(window_size=window_size)

    # 3) 组装 X(含常数1.0)、y，并截取最近 2 * window_size 条
    df_hist["intercept"] = 1.0
    X_all = df_hist[FACTOR_NAMES + ["intercept"]].values.astype(float)
    y_all = df_hist["daily_return"].values.astype(float)
    X_win = X_all[-2*window_size:, :]
    y_win = y_all[-2*window_size:]

    # 4) 建立 QREstimator，并滚动填充窗口
    qr = QREstimator(window_size=window_size, base_dim=5)
    for i in range(2*window_size):
        # QREstimator 里约定 X_window 的元素是形状 (1, n_feat) 的行向量
        qr.update_data(X_win[i:i+1, :], float(y_win[i]))
        qr.estimate()

    return qr