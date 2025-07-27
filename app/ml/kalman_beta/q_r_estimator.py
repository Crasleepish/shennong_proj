import numpy as np
from sklearn.linear_model import LinearRegression
from collections import deque

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
