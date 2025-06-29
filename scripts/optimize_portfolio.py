# 文件路径建议：app/scripts/optimize_portfolio.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from typing import List
import numpy as np
from app.data_fetcher.factor_data_reader import FactorDataReader
from app.data_fetcher.csi_index_data_fetcher import CSIIndexDataFetcher
from app.dao.fund_info_dao import FundHistDao
from app import create_app
from app.ml.black_litterman_opt_util import load_fund_betas, compute_prior_mu_sigma, build_bl_views, compute_bl_posterior, optimize_mean_variance

app = create_app()

def optimize(asset_source_map: dict, code_factors_map: dict, trade_date: str, window: int = 20, view_codes: List[str] = None):
    factor_data_reader = FactorDataReader()
    csi_index_data_fetcher = CSIIndexDataFetcher()
    # 1. 根据不同数据来源构造资产净值矩阵
    factor_codes = [code for code, src in asset_source_map.items() if src == "factor"]
    index_codes = [code for code, src in asset_source_map.items() if src == "index"]
    hist_codes = [code for code, src in asset_source_map.items() if src == "hist"]

    net_value_df = pd.DataFrame()

    if factor_codes:
        df_beta = load_fund_betas(factor_codes).set_index("code")[["MKT", "SMB", "HML", "QMJ"]]
        df_factors = factor_data_reader.read_daily_factors()[["MKT", "SMB", "HML", "QMJ"]].dropna()
        for code in factor_codes:
            beta = df_beta.loc[code].values
            cumret = df_factors.values @ beta
            net_value_df[code] = (1 + pd.Series(cumret, index=df_factors.index)).cumprod()

    for code in index_codes:
        df = csi_index_data_fetcher.get_data_by_code_and_date(code=code)
        df = df[["date", "close"]].dropna().set_index("date")
        df = df.sort_index()
        net_value_df[code] = df["close"]

    dao = FundHistDao._instance
    for code in hist_codes:
        df = dao.select_dataframe_by_code(code)
        df = df[["date", "net_value"]].dropna().set_index("date").sort_index()
        net_value_df[code] = df["net_value"]

    net_value_df = net_value_df.dropna(how="any").sort_index()
    fund_codes = list(asset_source_map.keys())

    # 2. 构造先验收益与协方差矩阵（使用净值曲线）
    mu_prior, Sigma, code_list_mu = compute_prior_mu_sigma(net_value_df, window=window, method="linear")

    # 3. 构造观点（P, q, omega）（仅使用 view_codes 子集）
    if not view_codes:
        view_asset_source_map = asset_source_map
        view_code_factors_map = code_factors_map
    else:
        view_asset_source_map = {code: asset_source_map[code] for code in view_codes if code in asset_source_map}
        view_code_factors_map = {code: code_factors_map[code] for code in view_codes if code in code_factors_map}
    P, q, omega, code_list_view = build_bl_views(view_asset_source_map, view_code_factors_map, trade_date)

    # 对齐 mu_prior 和 Sigma 的顺序与 fund_codes 保持一致
    # code_index_map = {code: i for i, code in enumerate(code_list_mu)}
    fund_indices = [code_list_mu.index(code) for code in fund_codes]
    mu_prior_full = mu_prior[fund_indices]
    Sigma_full = Sigma[np.ix_(fund_indices, fund_indices)]

    # 提取观点相关子集，将先验mu和Sigma调整与顺序与code_list_view一致
    view_indices = [fund_codes.index(code) for code in code_list_view]
    mu_prior_view = mu_prior_full[view_indices]
    Sigma_view = Sigma_full[np.ix_(view_indices, view_indices)]

    # 计算后验收益率（仅观点子集）
    mu_post_view = compute_bl_posterior(
        mu_prior=mu_prior_view,
        Sigma=Sigma_view,
        P=P,
        q=q,
        omega=omega,
        tau=0.2
    )

    # 将后验结果更新到完整序列中，顺序与fund_codes保持一致
    mu_post_full = mu_prior_full.copy()
    for i, idx in enumerate(view_indices):
        mu_post_full[idx] = mu_post_view[i]

    # 4. Max Sharpe 组合优化
    # weights, expected_return, expected_vol = optimize_max_sharpe(mu_post_full, Sigma_full)
    weights, expected_return, expected_vol = optimize_mean_variance(mu_post_full, Sigma_full, 0.0006)

    return {
        'weights': dict(zip(fund_codes, weights)),
        'expected_return': expected_return,
        'expected_volatility': expected_vol,
        'sharpe_ratio': expected_return / expected_vol
    }

if __name__ == '__main__':
    with app.app_context():
        # 创建测试数据
        asset_source_map = {
            'H11004.CSI': 'index',
            'Au99.99.SGE': 'index',
            '008114.OF': 'factor',
            '020602.OF': 'factor',
            '019918.OF': 'factor', 
            '002236.OF': 'factor',
            '019311.OF': 'factor',
            '006712.OF': 'factor',
            '011041.OF': 'factor',
            '110003.OF': 'factor',
            '019702.OF': 'factor',
            '006342.OF': 'factor',
            '020466.OF': 'factor',
            '018732.OF': 'factor',
        }
        code_factors_map = {
            "H11004.CSI": ["10YBOND"], 
            "Au99.99.SGE": ["GOLD"],
            "008114.OF": ["MKT", "SMB", "HML", "QMJ"],
            "020602.OF": ["MKT", "SMB", "HML", "QMJ"],
            "019918.OF": ["MKT", "SMB", "HML", "QMJ"],
            "002236.OF": ["MKT", "SMB", "HML", "QMJ"],
            "019311.OF": ["MKT", "SMB", "HML", "QMJ"],
            "006712.OF": ["MKT", "SMB", "HML", "QMJ"],
            "011041.OF": ["MKT", "SMB", "HML", "QMJ"],
            "110003.OF": ["MKT", "SMB", "HML", "QMJ"],
            "019702.OF": ["MKT", "SMB", "HML", "QMJ"],
            '006342.OF': ["MKT", "SMB", "HML", "QMJ"],
            '020466.OF': ["MKT", "SMB", "HML", "QMJ"],
            '018732.OF': ["MKT", "SMB", "HML", "QMJ"],
        }
        trade_date = '2025-06-20'
        window = 20
        # view_codes = ['H11004.CSI', 'Au99.99.SGE', '008114.OF', '020602.OF', '019918.OF', '002236.OF', '019311.OF', '006712.OF', '011041.OF', '110003.OF', '019702.OF', '006342.OF']
        portfolio_plan = optimize(asset_source_map, code_factors_map, trade_date, window, None)
        print(portfolio_plan)
    