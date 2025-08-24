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
from app.ml.black_litterman_opt_util import load_fund_betas, compute_prior_mu_sigma, compute_prior_mu_fixed_window, build_bl_views, compute_bl_posterior, optimize_mean_variance
from app.service.portfolio_opt import optimize
from app.service.portfolio_assets_service import get_portfolio_assets

app = create_app()


if __name__ == '__main__':
    with app.app_context():
        # 创建测试数据
        portfolio_id = 1
        asset_info = get_portfolio_assets(portfolio_id)
        asset_source_map = asset_info["asset_source_map"]
        code_factors_map = asset_info["code_factors_map"]
        view_codes = asset_info["view_codes"]
        trade_date = '2025-08-13'
        window = 20
        portfolio_plan = optimize(asset_source_map, code_factors_map, trade_date, window, view_codes)
        print(portfolio_plan)
    