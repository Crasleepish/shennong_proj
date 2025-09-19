# app/utils/cov_packer.py
import base64, json, zlib
import numpy as np
from typing import List, Tuple, Dict, Any


def pack_covariance(cov: np.ndarray) -> str:
    """
    将 NxN 协方差矩阵打包为 JSON 字符串（仅存下三角，float32）
    """
    cov = np.asarray(cov)
    assert cov.ndim == 2 and cov.shape[0] == cov.shape[1], "cov must be square"
    n = cov.shape[0]

    # 仅取下三角
    tri_idx = np.tril_indices(n)
    tri_vec = cov[tri_idx].astype(np.float32, copy=False)

    # 压缩&编码
    raw = tri_vec.tobytes()

    meta = {
        "dtype": str(tri_vec.dtype),
        "n": n,
    }
    return raw, json.dumps(meta, separators=(",", ":"))  # 紧凑 JSON

def unpack_covariance(buf: bytes, meta_json: str) -> Tuple[np.ndarray, List[str]]:
    """
    解包为 (cov_matrix, codes)
    """
    meta = json.loads(meta_json)
    n = int(meta["n"])
    tri_vec = np.frombuffer(buf, dtype=meta["dtype"])

    cov = np.zeros((n, n), dtype=np.float32)
    tri_idx = np.tril_indices(n)
    cov[tri_idx] = tri_vec
    cov[(tri_idx[1], tri_idx[0])] = tri_vec  # 对称补齐
    return cov
