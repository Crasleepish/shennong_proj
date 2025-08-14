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

app = create_app()


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
            '270004.OF': 'cash',
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
        view_codes = ["H11004.CSI", "Au99.99.SGE", "008114.OF", "020602.OF", "019918.OF", "002236.OF", "019311.OF", "006712.OF", "011041.OF", "110003.OF", "019702.OF", "006342.OF", "020466.OF", "018732.OF"]
        trade_date = '2025-08-13'
        window = 20
        # view_codes = ['H11004.CSI', 'Au99.99.SGE', '008114.OF', '020602.OF', '019918.OF', '002236.OF', '019311.OF', '006712.OF', '011041.OF', '110003.OF', '019702.OF', '006342.OF']
        portfolio_plan = optimize(asset_source_map, code_factors_map, trade_date, window, view_codes)
        print(portfolio_plan)
    