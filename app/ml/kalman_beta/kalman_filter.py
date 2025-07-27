import numpy as np

class KalmanFilter:
    def __init__(self, state_dim, z0=None, P0=None):
        self.state_dim = state_dim
        self.z = z0 if z0 is not None else np.zeros((state_dim, 1))
        self.P = P0 if P0 is not None else np.eye(state_dim)

    def step(self, H, y, Q, R):
        """
        单步更新
        H: shape (state_dim, 1), y: float, Q: (state_dim, state_dim), R: float
        """
        # 预测
        z_pred = self.z
        P_pred = self.P + Q

        # 计算Kalman增益
        S = H.T @ P_pred @ H + R
        K = P_pred @ H / S

        # 更新
        residual = y - (H.T @ z_pred)[0, 0]
        self.z = z_pred + K * residual
        self.P = (np.eye(self.state_dim) - K @ H.T) @ P_pred

        return self.z.copy()
