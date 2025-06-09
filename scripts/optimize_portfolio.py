# 文件路径建议：app/scripts/optimize_portfolio.py

import pandas as pd
from typing import List, Tuple
from app.ml.view_builder import build_view_matrix
from app.ml.inference import get_softprob_dict, get_label_to_ret


def load_fund_betas(codes: List[str]) -> pd.DataFrame:
    """
    从 output/fund_factors.csv 中读取指定 code 的因子暴露数据。
    返回包含 ['code', 'MKT', 'SMB', 'HML', 'QMJ'] 的 DataFrame。
    """
    df = pd.read_csv("output/fund_factors.csv")
    df = df[df["code"].isin(codes)].reset_index(drop=True)
    return df[["code", "MKT", "SMB", "HML", "QMJ"]]


def build_bl_views(codes: List[str], trade_date: str) -> Tuple:
    """
    构造 Black-Litterman 所需的观点：P, q, omega, code_list。
    """
    df_beta = load_fund_betas(codes)
    softprob_dict = get_softprob_dict(trade_date)
    label_to_ret = get_label_to_ret()

    P, q, omega, code_list = build_view_matrix(
        df_beta=df_beta,
        softprob_dict=softprob_dict,
        label_to_ret=label_to_ret,
        asset_codes=codes,
        top_k=min(10, len(codes))
    )
    return P, q, omega, code_list
