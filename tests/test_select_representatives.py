# test_select_representatives.py
import numpy as np
import pytest

from app.ml.greedy_convex_hull_volume import select_representatives

RTOL = 1e-6
ATOL = 1e-6

def _row_in_array(arr: np.ndarray, row: np.ndarray, atol: float = 1e-6) -> bool:
    """Check whether a row (vector) exists in arr up to atol."""
    diffs = np.linalg.norm(arr - row.reshape(1, -1), axis=1)
    return bool(np.any(diffs <= atol))

def test_triangle_inner_points_not_selected():
    """
    2D：X 含两个正交极点 (1,0),(0,1) 和若干位于 conv({0,这两点}) 内部/边上的点。
    期望：只选两个极点，内部点不会被加入。
    """
    X = np.array([
        [1.0, 0.0],           # extreme
        [0.0, 1.0],           # extreme
        [0.5, 0.5],           # on edge between the two extremes
        [0.2, 0.1],           # inside triangle conv({0,(1,0),(0,1)})
    ], dtype=float)

    S = select_representatives(
        X, epsilon=0.02,      # 相对阈值稍高，避免数值抖动引入冗余
        M=512,
        rng_seed=123,
        topk_per_iter=16,
        debug=True
    )

    # 应至少包含两个极点
    assert _row_in_array(S, np.array([1.0, 0.0]), atol=ATOL)
    assert _row_in_array(S, np.array([0.0, 1.0]), atol=ATOL)

    # 不应包含严格位于三角形内部/边上的点（在这个场景里，增量应为 0）
    assert not _row_in_array(S, np.array([0.5, 0.5]), atol=ATOL)
    assert not _row_in_array(S, np.array([0.2, 0.1]), atol=ATOL)

    # 一般情况下只选两个（初始 pivoted-QR 即满秩且无增量）
    assert S.shape[1] == 2
    assert S.shape[0] == 2

def test_unbounded_alignment_candidate_is_selected():
    """
    2D：初始包含 (1,0),(0,1)，再给一个朝“负方向”的候选 u = (-0.6, -0.2)。
    在很多方向上 K° 无界；u 的主要作用是“封无界”，其有界违约分数可能为 0，
    但算法的无界对齐项应把它推入精算并最终被选中。
    """
    X = np.array([
        [1.0, 0.0],           # extreme
        [0.0, 1.0],           # extreme
        [-0.6, -0.2],         # candidate that helps close unbounded directions
        [0.3, 0.1],           # small interior-ish point
    ], dtype=float)

    S = select_representatives(
        X, epsilon=0.005,     # 更低阈值，允许加入“封无界”的第三个点
        M=512,
        rng_seed=42,
        topk_per_iter=16,
        debug=True
    )

    # 应包含“封无界”的候选
    assert _row_in_array(S, np.array([-0.6, -0.2]), atol=ATOL)

    # 同时也应包含两个正交极点中的至少一个（通常两者都在）
    assert _row_in_array(S, np.array([1.0, 0.0]), atol=ATOL) or \
           _row_in_array(S, np.array([0.0, 1.0]), atol=ATOL)

    # 至少应选出 3 个（两个极点 + 负向候选）
    assert S.shape[0] == 3
    assert S.shape[1] == 2

def test_larger_unbounded_candidate_is_prior():
    """
    2D：初始包含 (1,0),(0,1)，再给一个朝“负方向”的候选 u = (-0.6, -0.2)。
    以及另一个增量小一些的候选 u = (0.2, -0.2)。
    """
    X = np.array([
        [1.0, 0.0],           # extreme
        [0.0, 1.0],           # extreme
        [-0.6, -0.2],         # candidate that helps close unbounded directions
        [0.2, -0.2],
        [0.3, 0.1],           # small interior-ish point
    ], dtype=float)

    S = select_representatives(
        X, epsilon=0.005,     # 更低阈值，允许加入“封无界”的第三个点
        M=512,
        rng_seed=42,
        topk_per_iter=16,
        debug=True
    )

    # 应包含“封无界”的候选
    assert _row_in_array(S, np.array([-0.6, -0.2]), atol=ATOL)

    # 同时也应包含两个正交极点中的至少一个（通常两者都在）
    assert _row_in_array(S, np.array([1.0, 0.0]), atol=ATOL) or \
           _row_in_array(S, np.array([0.0, 1.0]), atol=ATOL)

    # 至少应选出 3 个（两个极点 + 负向候选）
    assert S.shape[0] == 4
    assert S.shape[1] == 2
    assert np.array_equal(S[3], np.array([0.2, -0.2]))

def test_larger_unbounded_candidate_is_prior():
    """
    2D：初始包含 (1,0),(0,1)，再给一个朝“负方向”的候选 u = (-0.6, -0.2)。
    以及另一个增量小一些的候选 u = (-0.5, -0.1)。
    """
    X = np.array([
        [1.0, 0.0],           # extreme
        [0.0, 1.0],           # extreme
        [-0.5, -0.1],         # candidate that helps close unbounded directions
        [-0.6, -0.2],
        [0.3, 0.1],           # small interior-ish point
    ], dtype=float)

    S = select_representatives(
        X, epsilon=0.005,     # 更低阈值，允许加入“封无界”的第三个点
        M=512,
        rng_seed=42,
        topk_per_iter=16,
        debug=True
    )

    # 应包含“封无界”的候选
    assert _row_in_array(S, np.array([-0.6, -0.2]), atol=ATOL)

    # 同时也应包含两个正交极点中的至少一个（通常两者都在）
    assert _row_in_array(S, np.array([1.0, 0.0]), atol=ATOL) or \
           _row_in_array(S, np.array([0.0, 1.0]), atol=ATOL)

    # 至少应选出 3 个（两个极点 + 负向候选）
    assert S.shape[0] == 3
    assert S.shape[1] == 2
    assert np.array_equal(S[2], np.array([-0.6, -0.2]))


def test_reproducibility_with_fixed_seed():
    """
    同一随机种子应得到一致的输出（方向采样一致）。
    """
    rng_seed = 999
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [-0.6, -0.2],
        [0.3, 0.1],
    ], dtype=float)

    S1 = select_representatives(
        X, epsilon=0.01, M=512, rng_seed=rng_seed, topk_per_iter=16, debug=False
    )
    S2 = select_representatives(
        X, epsilon=0.01, M=512, rng_seed=rng_seed, topk_per_iter=16, debug=False
    )

    assert S1.shape == S2.shape
    # 顺序也应一致（按“加入顺序”返回）
    assert np.allclose(S1, S2, rtol=RTOL, atol=ATOL)
