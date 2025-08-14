# tests/test_cov_packer.py
import json
import numpy as np
import pytest

from app.utils.cov_packer import pack_covariance, unpack_covariance
from app.service.portfolio_opt import compute_diverge
from app import create_app
from app.config import Config

@pytest.fixture
def app():
    app = create_app(config_class=Config)
    yield app

def make_spd_matrix(n: int, seed: int = 7) -> np.ndarray:
    """
    构造 n x n 的对称正定矩阵（近似协方差矩阵）。
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n  # 保证对称正定
    return cov


@pytest.mark.parametrize("n", [3, 5, 15])
def test_pack_unpack_roundtrip_basic(n):
    cov = make_spd_matrix(n, seed=42)

    buf, meta_json = pack_covariance(cov)

    # 基础元信息检查
    meta = json.loads(meta_json)
    assert meta["n"] == n
    assert meta["dtype"] == "float32"

    cov_unpacked = unpack_covariance(buf, meta_json)

    # 维度、codes、一致性检查
    assert cov_unpacked.shape == (n, n)

    # 对称&dtype
    assert np.allclose(cov_unpacked, cov_unpacked.T, atol=0, rtol=0)
    assert cov_unpacked.dtype == np.float32


def test_lower_triangle_equivalence():
    """
    仅下三角被打包：上三角即使被篡改，也不应影响打包结果。
    """
    n = 6
    cov = make_spd_matrix(n, seed=123)

    # 人为修改上三角，保持下三角不变
    cov_asym = cov.copy()
    iu = np.triu_indices(n, k=1)
    cov_asym[iu] = cov_asym[iu] + 123.456  # 上三角扰动

    codes = [f"C{i}" for i in range(n)]

    packed_sym = pack_covariance(cov, codes, sig=4)
    packed_asym = pack_covariance(cov_asym, codes, sig=4)

    # 解包后，两者应完全一致（因为仅使用下三角）
    cov_sym, _ = unpack_covariance(packed_sym)
    cov_asym_recovered, _ = unpack_covariance(packed_asym)

    assert np.allclose(cov_sym, cov_asym_recovered, atol=0, rtol=0)


def test_nonsquare_raises():
    cov_rect = np.zeros((3, 5), dtype=float)
    codes = ["A", "B", "C", "D", "E"]
    with pytest.raises(AssertionError):
        _ = pack_covariance(cov_rect, codes, sig=4)

def test_compute_diverge(app):
    portfolio_id = 1
    trade_date = "2025-08-13"
    current_w = {
        "008114.OF": 0.07680478,
        "006342.OF": 4e-8,
        "110003.OF": 0.0,
        "019702.OF": 0.0,
        "011041.OF": 0.0,
        "020466.OF": 0.00236983,
        "019311.OF": 4e-8,
        "270004.OF": 0.02067287,
        "002236.OF": 0.00686659,
        "006712.OF": 4e-8,
        "018732.OF": 0.02849554,
        "019918.OF": 0.108306,
        "Au99.99.SGE": 0.29684363,
        "020602.OF": 0.0537684,
        "H11004.CSI": 0.40587236
    }
    target_w = {
        "008114.OF": 0.05711707,
        "006342.OF": 4e-8,
        "110003.OF": 0.0,
        "019702.OF": 0.0,
        "011041.OF": 0.0,
        "020466.OF": 0.00056829,
        "019311.OF": 4e-8,
        "270004.OF": 0.00916007,
        "002236.OF": 0.00203395,
        "006712.OF": 4e-8,
        "018732.OF": 0.02208354,
        "019918.OF": 0.07996696,
        "Au99.99.SGE": 0.37924829,
        "020602.OF": 0.03962734,
        "H11004.CSI": 0.41019449
    }
    
    diverge = compute_diverge(portfolio_id, trade_date, current_w, target_w)
    print(diverge)
