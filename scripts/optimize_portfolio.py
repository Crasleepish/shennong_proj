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
from app.service.portfolio_opt import optimize
from app.service.portfolio_assets_service import get_portfolio_assets

app = create_app()


if __name__ == '__main__':
    with app.app_context():
        # 创建测试数据
        portfolio_id = 2
        asset_info = get_portfolio_assets(portfolio_id)
        asset_source_map = asset_info["asset_source_map"]
        code_factors_map = asset_info["code_factors_map"]
        view_codes = asset_info["view_codes"]
        params = asset_info["params"]
        if params is None or "post_view_tau" not in params or "alpha" not in params or "variance" not in params:
            raise Exception("Invalid params, please set post_view_tau and alpha and variance in params")
        post_view_tau = float(params["post_view_tau"])
        variance = float(params["variance"])
        alpha = float(params["alpha"])
        trade_date = '2025-08-22'
        window = 20
        portfolio_plan = optimize(asset_source_map, code_factors_map, trade_date, post_view_tau, variance, window, view_codes)
        print(portfolio_plan)
    