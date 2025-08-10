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
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.X_window = deque(maxlen=window_size)
        self.y_window = deque(maxlen=window_size)
        self.beta_history = deque(maxlen=window_size)
        self.prev_beta = None

    def update_data(self, X, y):
        self.X_window.append(X)
        self.y_window.append(y)

    def estimate(self):
        if len(self.X_window) < self.window_size:
            return np.eye(5) * 1e-4, 1e-4

        X = np.vstack(self.X_window)
        y = np.array(self.y_window)

        model = LinearRegression(fit_intercept=False).fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        R = np.var(residuals)

        current_beta = model.coef_
        if self.prev_beta is not None:
            delta_beta = current_beta - self.prev_beta
            self.beta_history.append(delta_beta)
        self.prev_beta = current_beta

        if len(self.beta_history) < 6:
            Q = np.eye(5) * 1e-4
        else:
            Q = np.cov(np.array(self.beta_history).T)

        return Q, R

def _bootstrap_qr_from_history(fund_code: str, ref_date: str, window_size: int = 60) -> QREstimator:
    """
    使用 ref_date 之前的最近 window_size 个【交易日】数据，重建 QREstimator 的内部状态：
    - 填满 X_window / y_window
    - 用窗口 OLS 显式设定 prev_beta（长度=5，含截距）
    若历史不足 window_size，则返回一个空窗的 QREstimator（按其内部冷启动退化逻辑工作）。
    """
    # 1) 找到 ref_date 之前的交易日列表（严格小于 ref_date）
    trade_dates = CalendarFetcher().get_trade_date(start="19900101", end=ref_date.replace("-", ""), format="%Y-%m-%d", limit=window_size, ascending=False)
    trade_dates = [td for td in trade_dates if pd.to_datetime(td) <= pd.to_datetime(ref_date)]
    if len(trade_dates) == 0:
        return QREstimator(window_size=window_size)

    # 取最后 window_size 个交易日作为窗口
    trade_dates.sort()
    start_hist = trade_dates[0]
    end_hist = trade_dates[-1]

    # 2) 拉取因子 & 基金收益，并对齐
    factor_df = MarketFactorsDao._instance.select_dataframe_by_date(start_hist, end_hist).set_index("date")
    fund_df = get_fund_daily_return_for_beta_regression(fund_code, start_hist, end_hist)

    # 注意：有可能 date 既是列又是索引，所以统一 reset_index 再 merge
    df_hist = factor_df.join(fund_df).sort_index()
    df_hist = df_hist.dropna(how="any")
    if df_hist.empty or len(df_hist) < window_size:
        # 历史不足，返回默认空窗（让 QREstimator 自己走冷启动）
        return QREstimator(window_size=window_size)

    # 3) 组装 X(含常数1.0)、y，并截取最近 window_size 条
    df_hist["intercept"] = 1.0
    X_all = df_hist[FACTOR_NAMES + ["intercept"]].values.astype(float)
    y_all = df_hist["daily_return"].values.astype(float)
    X_win = X_all[-window_size:, :]
    y_win = y_all[-window_size:]

    # 4) 建立 QREstimator，并填充窗口
    qr = QREstimator(window_size=window_size)
    for i in range(window_size):
        # QREstimator 里约定 X_window 的元素是形状 (1, n_feat) 的行向量
        qr.update_data(X_win[i:i+1, :], float(y_win[i]))

    # 5) 用窗口 OLS 显式设置 prev_beta（维度=5），确保后续第一天就能产生 delta_beta
    #    这里使用 fit_intercept=False，因为我们已在 X 中加入了 "intercept" 列
    ols = LinearRegression(fit_intercept=False).fit(X_win, y_win)
    qr.prev_beta = ols.coef_.astype(float)  # shape = (5,)

    return qr