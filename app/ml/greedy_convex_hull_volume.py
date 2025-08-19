# greedy_convex_hull_volume.py
# -*- coding: utf-8 -*-
"""
Greedy representative selection for scenario A:
    Maximize Vol(conv({0} ∪ S)) with a relative gain threshold epsilon.
Core technique:
    Radial integral + Polar LP  (rho = 1 / h_{K°})

Input:
    X: np.ndarray of shape (N, d), each row is a d-dim vector.
    epsilon: float, relative gain threshold to stop.

Output:
    S: np.ndarray of shape (k, d), rows are chosen vectors IN THE ORDER selected.

Notes:
    - Uses sphere sampling (M directions) to estimate volume and marginal gains.
    - Dual LP per direction: maximize θ^T y s.t. A y <= 1, where A = X_S (rows are selected vectors).
    - For a candidate u, only directions violating u^T y*(θ) <= 1 need re-solving.

Author: (you)
"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
from math import pi
from math import gamma as gamma_func
from scipy.optimize import linprog


# ------------------------------- Utilities -------------------------------

def sphere_area(dim: int) -> float:
    """Area of the unit sphere S^{dim-1}."""
    return 2.0 * (pi ** (dim / 2.0)) / gamma_func(dim / 2.0)


def sample_sphere_uniform(dim: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """
    Uniformly sample M directions on S^{dim-1} via Gaussian normalization.
    Returns: (M, dim) array; each row is a unit vector θ.
    """
    G = rng.normal(size=(M, dim))
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    # Avoid division by zero in degenerate RNG cases
    norms = np.where(norms == 0.0, 1.0, norms)
    return G / norms


def dual_lp_support_theta(A: np.ndarray, theta: np.ndarray) -> Tuple[float, Optional[np.ndarray], bool]:
    """
    Solve h_{K°}(theta) = max_y theta^T y s.t. A y <= 1.
    A: (m, d) matrix, each row is a constraint vector x_i^T.
    Returns:
        h  : optimal value in [0, +inf] (np.inf if unbounded),
        y* : optimizer (d,), or None if unbounded/infeasible,
        ok : True if LP solved (bounded optimum); False if unbounded/infeasible.
    """
    d = A.shape[1]
    c = -theta.astype(float)              # maximize θ^T y == minimize -θ^T y
    A_ub = A.astype(float)                # Ay <= 1
    b_ub = np.ones(A.shape[0], dtype=float)
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * d, method="highs")
    if res.status == 0:
        y_opt = res.x
        h = float(theta @ y_opt)
        return h, y_opt, True
    elif res.status == 3:                 # unbounded
        return np.inf, None, False
    else:                                  # infeasible/other => treat as unbounded for our purpose
        return np.inf, None, False


# --------------------------- Initialization (rank) ---------------------------

def pivoted_qr_init_indices(X: np.ndarray, dim: int) -> List[int]:
    """
    Choose up to 'dim' rows (vectors) from X to quickly span R^dim using pivoted-QR on X^T.
    Returns a list of row indices.
    """
    # X: (N, d)   We run QR with pivoting on X^T (shape d x N)
    # numpy.linalg.qr with pivoting is in scipy.linalg.qr; fallback simple heuristic if unavailable.
    try:
        from scipy.linalg import qr as scipy_qr
        Q, R, piv = scipy_qr(X.T, mode="economic", pivoting=True)
        # take the first 'dim' pivot indices
        k = min(dim, len(piv))
        return list(piv[:k])
    except Exception:
        # Fallback: greedy by norm + orthogonal projection
        N = X.shape[0]
        idxs: List[int] = []
        V = np.empty((0, dim))
        for _ in range(min(dim, N)):
            # score candidates by their component orthogonal to span(V)
            best_i, best_score = -1, -1.0
            for i in range(N):
                if i in idxs:
                    continue
                v = X[i]
                if V.shape[0] > 0:
                    # project v onto span(V) via least squares
                    coef, *_ = np.linalg.lstsq(V.T, v, rcond=None)
                    proj = V.T @ coef
                    resid = v - proj
                else:
                    resid = v
                sc = float(np.linalg.norm(resid))
                if sc > best_score:
                    best_score = sc
                    best_i = i
            if best_i >= 0:
                idxs.append(best_i)
                V = X[best_i:best_i+1] if V.shape[0] == 0 else np.vstack([V, X[best_i]])
        return idxs


# ------------------------- Main greedy selection -------------------------

class RadialVolumeAccumulator:
    """
    维护径向积分体积的三要素：rho、rho**dim、其均值。
    只对少数索引做增量更新，并能直接给出体积与体积增量。
    """
    def __init__(self, dim: int, rho: np.ndarray):
        self.dim = int(dim)
        self.rho = np.asarray(rho, dtype=float)
        self.M = self.rho.size
        self.rho_pow = self.rho ** self.dim
        self.mean_rho_pow = float(self.rho_pow.mean())
        self.area_over_d = sphere_area(self.dim) / self.dim

    def volume(self) -> float:
        return self.area_over_d * self.mean_rho_pow

    def delta_from_update(self, idx: np.ndarray, new_rho_vals: np.ndarray) -> float:
        """给定要更新的一组方向及其新的 rho，返回对应的体积增量（不提交）。"""
        if np.size(idx) == 0:
            return 0.0
        new_pow = (np.asarray(new_rho_vals, float)) ** self.dim
        old_pow = self.rho_pow[idx]
        sum_diff = float((new_pow - old_pow).sum())
        return self.area_over_d * (sum_diff / self.M)

    def apply_update(self, idx: np.ndarray, new_rho_vals: np.ndarray) -> None:
        """提交更新：同步 rho、rho**dim、均值。"""
        if np.size(idx) == 0:
            return
        new_pow = (np.asarray(new_rho_vals, float)) ** self.dim
        old_pow = self.rho_pow[idx]
        sum_diff = float((new_pow - old_pow).sum())
        self.rho[idx] = np.asarray(new_rho_vals, float)
        self.rho_pow[idx] = new_pow
        self.mean_rho_pow += sum_diff / self.M
        # 体积 = self.volume()  按需读取

def select_representatives(
    X: np.ndarray,
    epsilon: float,
    *,
    M: int = 4096,
    rng_seed: int = 42,
    topk_per_iter: Optional[int] = 64,
    violation_tol: float = 1e-9,
    max_iters: Optional[int] = None,
    clip_rhopow: Optional[float] = None,
    clip_viol: Optional[float] = None,
) -> np.ndarray:
    """
    Greedy selection under Scenario A (conv({0} ∪ S)) with radial accumulator.
    """
    assert X.ndim == 2 and X.shape[0] >= 1
    N, d = X.shape
    rng = np.random.default_rng(rng_seed)

    # 去零向量
    norms = np.linalg.norm(X, axis=1)
    keep = norms > 0.0
    X = X[keep]
    if X.shape[0] == 0:
        return np.zeros((0, d), dtype=float)

    # 球面方向
    thetas = sample_sphere_uniform(d, M, rng)  # (M, d)

    # 初始化 S（补满秩）
    init_idxs = pivoted_qr_init_indices(X, d)
    S_idx: List[int] = list(init_idxs)
    S = X[S_idx]
    A = S.copy()
    used_mask = np.zeros(X.shape[0], dtype=bool)
    used_mask[S_idx] = True

    # 初始 rho 与 Y(θ)（极化 LP），并记录哪些方向当前有界
    rho = np.zeros(M, dtype=float)
    Y = np.zeros((M, d), dtype=float)
    bounded = np.zeros(M, dtype=bool)  # True: h<∞（ok）；False: 无界（rho=0）
    for m in range(M):
        if A.size == 0:
            # 没有任何约束：h=+inf, rho=0
            rho[m] = 0.0
            bounded[m] = False
            continue
        h, y_opt, ok = dual_lp_support_theta(A, thetas[m])
        if ok:
            rho[m] = 1.0 / max(h, 1e-300)
            Y[m] = y_opt
            bounded[m] = True
        else:
            rho[m] = 0.0
            bounded[m] = False

    acc = RadialVolumeAccumulator(d, rho)
    vol = acc.volume()

    iter_count = 0
    while True:
        iter_count += 1
        if max_iters is not None and iter_count > max_iters:
            break

        cand_idx = np.where(~used_mask)[0]
        if cand_idx.size == 0:
            break

        # -------- 粗筛：score = Σ ρ^(d+1) * (u^T y*(θ) - 1)_+ --------
        U = X[cand_idx]                       # (C, d)
        dots = U @ Y.T                        # (C, M)
        viol = np.maximum(dots - (1.0 + violation_tol), 0.0)   # (C, M)

        rho_pow_d1 = acc.rho_pow * acc.rho   # ρ^(d+1), shape (M,)
        if clip_rhopow is not None:
            rho_pow_d1 = np.minimum(rho_pow_d1, float(clip_rhopow))
        if clip_viol is not None:
            viol = np.minimum(viol, float(clip_viol))

        scores = (viol * rho_pow_d1[None, :]).sum(axis=1)      # (C,)
        scores = np.nan_to_num(scores, nan=0.0,
                                posinf=np.finfo(float).max/4,
                                neginf=0.0)

        # 若分数全 0，但仍存在无界方向，则用“与无界方向正向对齐度”做二级排序，避免误停
        order = None
        if np.all(scores <= 0.0) and (~bounded).any():
            ub_idx = np.where(~bounded)[0]                         # 无界方向
            # 与这些方向的正向对齐度：max(⟨u, θ⟩, 0) 求和
            align = np.maximum(U @ thetas[ub_idx].T, 0.0).sum(axis=1)  # (C,)
            order = np.argsort(-align)
        else:
            order = np.argsort(-scores)

        if topk_per_iter is not None:
            order = order[: min(topk_per_iter, order.size)]

        cand_idx_ordered = cand_idx[order]
        scores_top = scores[order]

        # 注意：只有在“没有候选”且“也不存在无界方向”时才可安全早停
        if cand_idx_ordered.size == 0 and not (~bounded).any():
            break

        # -------- 精算前K：只在“违约方向 + 当前无界且可能受新约束影响的方向”上重解 LP --------
        best_delta = -np.inf
        best_i = None
        best_idx_m = None
        best_new_rho_m = None
        best_new_Y_m = None
        best_new_bounded_flags = None

        for idx_i in cand_idx_ordered:
            u = X[idx_i]
            # 1) 违约方向
            dots_u = u @ Y.T
            mask_vio = dots_u > (1.0 + violation_tol)

            # 2) 当前无界方向里，挑“与 u 有正向对齐”的子集尝试（启发式降开销）
            if (~bounded).any():
                ub_idx = np.where(~bounded)[0]
                align_mask = (thetas[ub_idx] @ u) > 1e-12
                if np.any(align_mask):
                    add_idx = ub_idx[align_mask]
                    mask_add = np.zeros(M, dtype=bool)
                    mask_add[add_idx] = True
                    mask_vio = np.logical_or(mask_vio, mask_add)

            if not np.any(mask_vio):
                continue

            idx_m = np.where(mask_vio)[0]
            A_aug = np.vstack([A, u.reshape(1, -1)])
            new_rho_m = np.empty(idx_m.size, dtype=float)
            new_Y_m = np.empty((idx_m.size, d), dtype=float)
            new_bounded_flags = np.zeros(idx_m.size, dtype=bool)

            for j, m in enumerate(idx_m):
                h2, y2, ok2 = dual_lp_support_theta(A_aug, thetas[m])
                if ok2:                                 # 变为有界或仍有界
                    new_rho_m[j] = 1.0 / max(h2, 1e-300)
                    new_Y_m[j] = y2
                    new_bounded_flags[j] = True
                else:                                   # 仍无界
                    new_rho_m[j] = 0.0
                    new_Y_m[j] = Y[m]                  # 占位（不影响）
                    new_bounded_flags[j] = False

            delta = acc.delta_from_update(idx_m, new_rho_m)

            if delta > best_delta:
                best_delta = float(delta)
                best_i = int(idx_i)
                best_idx_m = idx_m
                best_new_rho_m = new_rho_m
                best_new_Y_m = new_Y_m
                best_new_bounded_flags = new_bounded_flags

        if best_i is None or best_delta <= 0.0:
            # 若连无界方向也无法带来正增量，则（几乎总是）可以退出
            break

        # 相对增益阈值
        rel_gain = best_delta / max(vol, 1e-300)
        if rel_gain < epsilon:
            break

        # -------- 接受最佳候选：提交更新 --------
        used_mask[best_i] = True
        S_idx.append(best_i)
        S = np.vstack([S, X[best_i]])
        A = S.copy()

        # 同步方向：rho/rho^d via accumulator；Y；bounded 标志
        acc.apply_update(best_idx_m, best_new_rho_m)
        Y[best_idx_m] = best_new_Y_m
        bounded[best_idx_m] = best_new_bounded_flags
        vol = acc.volume()

    return S