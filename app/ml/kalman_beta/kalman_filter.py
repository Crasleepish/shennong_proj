import numpy as np

class KalmanFilter:
    def __init__(self,
                 state_dim: int,
                 z0: np.ndarray = None,
                 P0: np.ndarray = None,
                 *,
                 alpha_index: int = -1,
                 alpha_rho: float = 0.97,
                 use_joseph: bool = True,
                 # === ECM 相关 ===
                 use_ecm: bool = True,
                 gamma_rho: float = 0.95,   # gamma 的 AR(1) 系数（轻度均值回复）
                 q_gamma: float = 1e-4      # gamma 的过程噪声（快响应）
                 ):
        """
        Parameters
        ----------
        state_dim : int
            原始状态维度（不含 ECM）。例如 5: 4个 beta + 1个 alpha。
        z0 : (state_dim[, +1], 1)
            初始状态。若 use_ecm=True 而传入的是原始维度，将在末尾补上 gamma=0。
        P0 : (state_dim[, +1], state_dim[, +1])
            初始协方差。若 use_ecm=True 而传入的是原始维度，将在末尾给 gamma 赋默认方差。
        alpha_index : int
            alpha 所在的状态下标（默认最后一维 -1；指的是“原始维度内”的索引）。
        alpha_rho : float
            alpha 的 AR(1) 系数 rho，建议 0.95~0.99；=1.0 等价随机游走。
        use_joseph : bool
            使用 Joseph 形式更新协方差，增强数值稳定性。
        use_ecm : bool
            是否启用 ECM：在观测中加入 TE_{t-1}，并将其系数 gamma 纳入状态。
        gamma_rho : float
            gamma 的 AR(1) 系数（可取 0.9~0.99；=1 为 RW）。
        q_gamma : float
            gamma 维度的过程噪声，通常比 beta/alpha 稍大一些以便快响应。
        """
        # 处理 alpha 索引（按原始维度）
        base_dim = state_dim
        self.use_ecm = bool(use_ecm)

        # 状态维度：是否增广
        if self.use_ecm:
            self.gamma_index = base_dim       # 新增的 gamma 维在末尾
            self.state_dim = base_dim + 1
        else:
            self.gamma_index = None
            self.state_dim = base_dim

        # 解析 alpha 索引（映射到新维度中）
        self.alpha_index = alpha_index if alpha_index >= 0 else (base_dim - 1)
        # 注意：alpha_index 是在“原始维度”里定义的；扩维后位置不变

        self.alpha_rho = float(alpha_rho)
        self.gamma_rho = float(gamma_rho)
        self.use_joseph = bool(use_joseph)

        # 初始化状态
        if z0 is None:
            self.z = np.zeros((self.state_dim, 1))
        else:
            z0 = np.asarray(z0, dtype=float)
            if z0.shape[0] == self.state_dim:
                self.z = z0.reshape(self.state_dim, 1)
            elif z0.shape[0] == base_dim and self.use_ecm:
                # 扩一维，gamma 初始为 0
                self.z = np.vstack([z0.reshape(base_dim, 1), [[0.0]]])
            else:
                raise ValueError("z0 shape mismatch w.r.t. use_ecm/state_dim")

        # 初始化协方差
        if P0 is None:
            self.P = np.eye(self.state_dim)
        else:
            P0 = np.asarray(P0, dtype=float)
            if P0.shape == (self.state_dim, self.state_dim):
                self.P = P0.copy()
            elif P0.shape == (base_dim, base_dim) and self.use_ecm:
                self.P = np.eye(self.state_dim)
                self.P[:base_dim, :base_dim] = P0
                self.P[self.gamma_index, self.gamma_index] = max(q_gamma, 1e-6)
            else:
                raise ValueError("P0 shape mismatch w.r.t. use_ecm/state_dim")

        # 状态转移矩阵 F
        self.F = np.eye(self.state_dim)
        # alpha: AR(1)
        self.F[self.alpha_index, self.alpha_index] = self.alpha_rho
        # gamma: AR(1)（仅在 use_ecm 时存在）
        if self.use_ecm:
            self.F[self.gamma_index, self.gamma_index] = self.gamma_rho

        # 数值稳定的小常数
        self._eps = 1e-12

        # 记录 gamma 的过程噪声（用于 Q 自动扩维时赋值）
        self._q_gamma_default = float(q_gamma)

    def _maybe_augment_H_Q(self, H: np.ndarray, Q: np.ndarray, te_prev: float):
        """
        若 use_ecm=True，则将 H 扩一维，把最后一个元素设为 te_prev；
        同时将 Q 在行列末尾扩一维（若来的是 base_dim 维度），并给 gamma 的过程噪声。
        """
        H = np.asarray(H, dtype=float).reshape(-1, 1)
        Q = np.asarray(Q, dtype=float)

        base_dim = H.shape[0]
        if self.use_ecm:
            # 扩展 H
            if base_dim + 1 == self.state_dim:
                # H 是 base_dim，内部状态是 base_dim+1
                H_ext = np.zeros((self.state_dim, 1))
                H_ext[:base_dim, 0] = H[:, 0]
                H_ext[self.gamma_index, 0] = float(te_prev)
                H = H_ext
                base_dim = self.state_dim  # 仅用于后续断言
            else:
                # 外部已经传了包含 gamma 的 H（最后一维应为 te_prev）
                pass

            # 扩展 Q（若来的是 base_dim 的 Q）
            if Q.shape == (self.state_dim - 1, self.state_dim - 1):
                Q_ext = np.eye(self.state_dim) * 0.0
                Q_ext[:self.state_dim - 1, :self.state_dim - 1] = Q
                Q_ext[self.gamma_index, self.gamma_index] = self._q_gamma_default
                Q = Q_ext

        # 断言形状匹配
        if H.shape != (self.state_dim, 1):
            raise ValueError(f"H shape {H.shape} mismatch with internal state_dim={self.state_dim}")
        if Q.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Q shape {Q.shape} mismatch with internal state_dim={self.state_dim}")

        return H, Q

    def step(self,
             H: np.ndarray,
             y: float,
             Q: np.ndarray,
             R: float,
             te_prev: float = 0.0) -> np.ndarray:
        """
        单步预测+更新（标量观测）
        - 若 use_ecm=True，则会把 te_prev 放到 H 的最后一维（gamma 的观测系数）

        Parameters
        ----------
        H : (state_dim or base_dim, 1) ndarray
            观测向量（例如 [r_MKT, r_SMB, r_HML, r_QMJ, 1]^T）
            若 use_ecm=True 且传入的是 base_dim，将在内部自动扩一维并将末尾设为 te_prev。
        y : float
            当期观测标量（基金当日收益）
        Q : (state_dim or base_dim, state_dim or base_dim) ndarray
            过程噪声协方差（可由外部 Q/R 估计器给出）
            若 use_ecm=True 且传入的是 base_dim，将在内部自动扩一维并给 gamma 赋默认过程噪声。
        R : float
            观测噪声方差
        te_prev : float
            上一期累计净值跟踪误差（log NAV_true - log NAV_fit）

        Returns
        -------
        z : (state_dim, 1) ndarray
            当期更新后的状态估计
        """
        # ECM: 必要时扩展 H/Q，并把 te_prev 填到 H 的最后一维
        H, Q = self._maybe_augment_H_Q(H, Q, te_prev)

        # 预测
        z_pred = self.F @ self.z
        P_pred = self.F @ self.P @ self.F.T + Q

        # 增益
        S = float(H.T @ P_pred @ H) + float(R)
        if S < self._eps:
            S = self._eps
        K = (P_pred @ H) / S  # (state_dim,1)

        # 更新
        y_pred = float(H.T @ z_pred)
        residual = float(y) - y_pred

        self.z = z_pred + K * residual

        # Joseph 更新（更稳）
        if self.use_joseph:
            I = np.eye(self.state_dim)
            KHt = K @ H.T
            self.P = (I - KHt) @ P_pred @ (I - KHt).T + K * R * K.T
            # 可选：数值对称化，进一步稳健
            # self.P = 0.5 * (self.P + self.P.T)
        else:
            self.P = (np.eye(self.state_dim) - K @ H.T) @ P_pred

        return self.z.copy()

    def set_alpha_rho(self, rho: float):
        """运行中动态调整 alpha 的 rho"""
        self.alpha_rho = float(rho)
        self.F[self.alpha_index, self.alpha_index] = self.alpha_rho

    def set_gamma_rho(self, rho: float):
        """运行中调整 ECM 系数 gamma 的 rho（仅在 use_ecm=True 时有效）"""
        if not self.use_ecm:
            return
        self.gamma_rho = float(rho)
        self.F[self.gamma_index, self.gamma_index] = self.gamma_rho

    def current_state(self) -> np.ndarray:
        return self.z.copy()

    def current_cov(self) -> np.ndarray:
        return self.P.copy()
