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

"""

from __future__ import annotations
from typing import Tuple, List, Optional
import numpy as np
from math import pi
from math import gamma as gamma_func
from scipy.optimize import linprog
import logging

# --------- numeric guard for 1/h -----------
H_MIN = 1e-12  # unify small floor for h to avoid overflow in 1/h


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
    Solve h_{K°}(theta) = max_y θ^T y s.t. A y <= 1.
    A: (m, d) matrix, each row is a constraint vector x_i^T.
    Returns:
        h  : optimal value in [0, +inf] (np.inf if unbounded),
        y* : optimizer (d,), or None if unbounded/infeasible,
        ok : True if LP solved (bounded optimum); False if unbounded/infeasible.
    """
    d = A.shape[1]
    c = -theta.astype(float)           # maximize θ^T y == minimize -θ^T y
    A_ub = A.astype(float)             # Ay <= 1
    b_ub = np.ones(A.shape[0], dtype=float)
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * d, method="highs")
    if res.status == 0:
        y_opt = res.x
        h = float(theta @ y_opt)
        return h, y_opt, True
    elif res.status == 3:              # unbounded
        return np.inf, None, False
    else:                              # infeasible/other => treat as unbounded for our purpose
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
        self.rho_pow = self.rho ** self.dim           # rho^d
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


# ------------------------- Logger helper -------------------------

def _get_logger(logger: Optional[logging.Logger], debug: bool) -> logging.Logger:
    if logger is not None:
        return logger
    log = logging.getLogger(__name__)
    log.propagate = True
    log.setLevel(logging.DEBUG if debug else logging.NOTSET)
    return log


# ------------------------- Main greedy selection -------------------------

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
    debug: bool = False,
    logger: Optional[logging.Logger] = None,
    log_topk: int = 5,
    diversity_beta: float = 1.5
) -> np.ndarray:
    """
    Greedy selection under Scenario A (conv({0} ∪ S)) with radial accumulator.
    Set debug=True (or pass a logger) to print per-iteration diagnostics.
    
    Returns
    -------
    idx : np.ndarray, shape (k,), dtype=int
        被选中的向量在**输入 X（原始，未去零）**中的行索引（按加入顺序）。
    """
    log = _get_logger(logger, debug)

    assert X.ndim == 2 and X.shape[0] >= 1
    N, d = X.shape
    rng = np.random.default_rng(rng_seed)

    # --- 去零向量，并建立 “过滤后索引 -> 原始索引” 的映射 ---
    norms = np.linalg.norm(X, axis=1)
    keep = norms > 0.0
    orig_idx_map = np.arange(N)[keep]         # 过滤后第 i 行对应的原始行号
    X = X[keep]
    if X.shape[0] == 0:
        return np.zeros((0, d), dtype=float)

    # 球面方向
    thetas = sample_sphere_uniform(d, M, rng)  # (M, d)

    # 初始化 S（补满秩）
    init_idxs = pivoted_qr_init_indices(X, d)  # 这些是 “过滤后 X” 的行号
    S_idx: List[int] = list(init_idxs)         # 维护为过滤后索引
    S = X[S_idx]
    A = S.copy()
    used_mask = np.zeros(X.shape[0], dtype=bool)
    used_mask[S_idx] = True

    # 初始 rho 与 Y(θ)，并记录哪些方向当前有界
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
            rho[m] = 1.0 / max(h, H_MIN)
            Y[m] = y_opt
            bounded[m] = True
        else:
            rho[m] = 0.0
            bounded[m] = False

    acc = RadialVolumeAccumulator(d, rho)
    vol = acc.volume()

    log.debug(f"Init: N={X.shape[0]} d={d} M={M} rank_init=|S|={len(S_idx)} "
              f"bounded={bounded.sum()} unbounded={(~bounded).sum()} vol≈{vol:.6g}")

    iter_count = 0
    while True:
        iter_count += 1
        if max_iters is not None and iter_count > max_iters:
            log.debug("Stop: reached max_iters")
            break

        cand_idx = np.where(~used_mask)[0]
        if cand_idx.size == 0:
            log.debug("Stop: no remaining candidates")
            break

        # -------- 粗筛：综合打分 = 有界一阶近似 + 无界对齐度 --------
        U = X[cand_idx]                                # (C, d)
        dots = U @ Y.T                                 # (C, M)
        viol = np.maximum(dots - (1.0 + violation_tol), 0.0)   # (C, M)

        # 有界部分：Σ ρ^(d+1) * (u^T y*(θ) - 1)_+
        rho_pow_d1 = acc.rho_pow * acc.rho             # (M,)
        if clip_rhopow is not None:
            rho_pow_d1 = np.minimum(rho_pow_d1, float(clip_rhopow))
        if clip_viol is not None:
            viol = np.minimum(viol, float(clip_viol))
        score_bounded = (viol * rho_pow_d1[None, :]).sum(axis=1)   # (C,)

        # 无界部分：对齐余弦（无量纲）× λ，λ ~ median(ρ^d)
        ub_mask = ~bounded                                           # (M,)
        if np.any(ub_mask):
            U_norm = np.linalg.norm(U, axis=1, keepdims=True) + H_MIN
            cos_align = np.maximum((U @ thetas[ub_mask].T) / U_norm, 0.0)  # (C, #unbounded)
            score_unbd_raw = cos_align.sum(axis=1)                          # (C,)
            lam = float(np.median(acc.rho_pow[acc.rho_pow > 0])) if np.any(acc.rho_pow > 0) else 1.0
            score = score_bounded + lam * score_unbd_raw
            # ----- 多样性权重：惩罚与 S 同向的重合 -----
            U_unit = U / U_norm
            S_unit = S / (np.linalg.norm(S, axis=1, keepdims=True) + H_MIN)
            cos_US = U_unit @ S_unit.T
            cos_US_pos = np.clip(cos_US, 0.0, 1.0)  # 只惩罚同向
            maxcos = cos_US_pos.max(axis=1) if cos_US_pos.size else 0.0
            w_div = (1.0 - maxcos)**diversity_beta + H_MIN
            score *= w_div
        else:
            score_unbd_raw = np.zeros(U.shape[0], dtype=float)
            score = score_bounded

        score = np.nan_to_num(score, nan=0.0,
                              posinf=np.finfo(float).max/4, neginf=0.0)

        # 主序：按综合分数排序；并为“封无界潜力强”的候选预留名额
        order_main = np.argsort(-score)
        reserve_unbounded = 0 if not np.any(ub_mask) else max(5, (topk_per_iter or 64)//4)
        order_unbd = np.argsort(-score_unbd_raw)[:reserve_unbounded]

        # 合并 + 稳定去重（保留第一次出现的顺序）
        order_list = np.concatenate([order_unbd, order_main]).tolist()
        seen = set()
        order = np.array([i for i in order_list if (i not in seen and not seen.add(i))], dtype=int)

        # 截断到 topK
        if topk_per_iter is not None:
            order = order[: min(topk_per_iter, order.size)]

        cand_idx_ordered = cand_idx[order]

        # 若没有候选且也不存在无界方向，才可早停
        if debug:
            kshow = min(log_topk, cand_idx_ordered.size)
            log.debug(f"[Iter {iter_count}] candidates={cand_idx.size} "
                      f"bounded={bounded.sum()} unbounded={(~bounded).sum()} "
                      f"vol≈{vol:.6g}")
            if kshow > 0:
                top_ids = cand_idx_ordered[:kshow]
                sb = score_bounded[order][:kshow]
                st = score[order][:kshow]
                log.debug("  TopK (idx_in_X, score_bounded, total): " +
                          ", ".join([f"({int(i)}, {b:.3g}, {t:.3g})"
                                     for i, b, t in zip(top_ids, sb, st)]))

        if cand_idx_ordered.size == 0 and not np.any(ub_mask):
            log.debug("Stop: no candidate passes coarse screen and no unbounded directions")
            break

        # -------- 精算前K：在“违约方向 + 与 u 对齐的无界方向”上重解 LP --------
        best_delta = -np.inf
        best_i = None
        best_idx_m = None
        best_new_rho_m = None
        best_new_Y_m = None
        best_new_bounded_flags = None

        total_lp = 0
        for idx_i in cand_idx_ordered:
            u = X[idx_i]
            # 1) 有界：违约方向
            dots_u = u @ Y.T
            mask_vio = dots_u > (1.0 + violation_tol)
            # 2) 无界：与 u 正向对齐的子集
            if np.any(ub_mask):
                ub_idx = np.where(ub_mask)[0]
                u_norm = np.linalg.norm(u) + 1e-12
                align_mask = (thetas[ub_idx] @ u) / u_norm > 0.1
                if np.any(align_mask):
                    mask_extra = np.zeros(M, dtype=bool)
                    mask_extra[ub_idx[align_mask]] = True
                    mask_vio = np.logical_or(mask_vio, mask_extra)

            if not np.any(mask_vio):
                continue

            idx_m = np.where(mask_vio)[0]
            A_aug = np.vstack([A, u.reshape(1, -1)])
            new_rho_m = np.empty(idx_m.size, dtype=float)
            new_Y_m = np.empty((idx_m.size, d), dtype=float)
            new_bounded_flags = np.zeros(idx_m.size, dtype=bool)

            # 记录分解：有界/无界切到的数量
            if debug:
                nb_v = int((mask_vio & bounded).sum())
                nu_v = int((mask_vio & (~bounded)).sum())

            for j, m in enumerate(idx_m):
                h2, y2, ok2 = dual_lp_support_theta(A_aug, thetas[m])
                if ok2:
                    new_rho_m[j] = 1.0 / max(h2, H_MIN)
                    new_Y_m[j] = y2
                    new_bounded_flags[j] = True
                else:
                    new_rho_m[j] = 0.0
                    new_Y_m[j] = Y[m]  # placeholder
                    new_bounded_flags[j] = False
            total_lp += idx_m.size

            delta = acc.delta_from_update(idx_m, new_rho_m)

            if debug:
                log.debug(f"    cand={int(idx_i)}  LPs={idx_m.size} "
                          f"(bounded_cut={nb_v}, unbounded_try={nu_v})  Δ≈{delta:.6g}")

            if delta > best_delta:
                best_delta = float(delta)
                best_i = int(idx_i)
                best_idx_m = idx_m
                best_new_rho_m = new_rho_m
                best_new_Y_m = new_Y_m
                best_new_bounded_flags = new_bounded_flags

        if best_i is None or best_delta <= 0.0:
            log.debug("Stop: no candidate yields positive Δ in fine evaluation")
            break

        rel_gain = best_delta / max(vol, H_MIN)
        log.debug(f"[Iter {iter_count}] pick idx={best_i}  Δ≈{best_delta:.6g} "
                  f"rel_gain≈{rel_gain:.3%}  LP_solved={total_lp}")

        if rel_gain < epsilon:
            log.debug(f"Stop: relative gain {rel_gain:.3%} < epsilon {epsilon:.3%}")
            break

        # -------- 接受最佳候选：提交更新 --------
        used_mask[best_i] = True
        S_idx.append(best_i)
        S = np.vstack([S, X[best_i]])
        A = S.copy()
        # 同步方向
        acc.apply_update(best_idx_m, best_new_rho_m)
        Y[best_idx_m] = best_new_Y_m
        bounded[best_idx_m] = best_new_bounded_flags
        vol = acc.volume()

        # 全量刷新仍标记为无界的方向
        ub_rest = np.where(~bounded)[0]
        if ub_rest.size:
            new_rho_rest = np.empty(ub_rest.size, dtype=float)
            new_Y_rest   = np.empty((ub_rest.size, d), dtype=float)
            new_bounded_rest = np.zeros(ub_rest.size, dtype=bool)
            for j, m in enumerate(ub_rest):
                h3, y3, ok3 = dual_lp_support_theta(A, thetas[m])
                if ok3:
                    new_rho_rest[j] = 1.0 / max(h3, H_MIN)
                    new_Y_rest[j]   = y3
                    new_bounded_rest[j] = True
                else:
                    new_rho_rest[j] = 0.0
                    new_Y_rest[j]   = Y[m]  # keep placeholder
                    new_bounded_rest[j] = False

            # 用累加器一次性提交这批更新（可能把 rho 从 0 提到正值）
            acc.apply_update(ub_rest, new_rho_rest)
            Y[ub_rest]       = new_Y_rest
            bounded[ub_rest] = new_bounded_rest
            vol = acc.volume()
            if debug:
                n_fixed = int(new_bounded_rest.sum())
                _msg = f"    refreshed unbounded={ub_rest.size}, now fixed={n_fixed}, bounded={bounded.sum()}"
                log.debug(_msg)
        log.debug(f"    |S|={len(S_idx)}  new_vol≈{vol:.6g}")

    log.debug(f"Done. selected |S|={len(S_idx)}")

    # —— 把过滤后索引映射回“原始 X 的行号”，按加入顺序返回 ——
    selected_idx = orig_idx_map[np.array(S_idx, dtype=int)]
    return selected_idx
