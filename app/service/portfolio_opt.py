from typing import List
import numpy as np
import os
import pandas as pd
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
from datetime import datetime

def optimize_portfolio_realtime():
    """
    实时组合优化主流程：完成从数据获取、因子计算、预测、优化的全过程。
    """
    # Step 1: 获取全市场实时行情数据
    stock_rt, index_rt = fetch_realtime_market_data()

    # Step 2: 计算各股票组合收益率
    portfolio_returns = compute_portfolio_returns(stock_rt)

    # Step 3: 计算实时因子收益率（MKT, SMB, HML, QMJ）
    realtime_factors = calculate_intraday_factors(portfolio_returns, index_rt)

    # Step 4: 估算实时指数数据
    index_to_etf = {
        "H11001.CSI": "511010.SH",  # 中证综合债 → 国债ETF
        "H11004.CSI": "511260.SH",  # 中证10债 → 十年国债ETF
        "Au99.99.SGE": "518880.SH",  # 黄金 → 黄金ETF
    }
    est_index_values = estimate_intraday_index_value(index_to_etf)

    # Step 5: 读取历史因子收益率（用于特征构造）
    additonal_factor_df, additonal_map = build_real_time_date(realtime_factors, est_index_values)

    # today = datetime.strftime(TradeCalendarReader.get_trade_dates(end=datetime.strftime(datetime.today(), "%Y%m%d"))[-1], "%Y-%m-%d")
    # additonal_factor_df = pd.DataFrame([{
    #     "MKT": 0.012,
    #     "SMB": -0.004,
    #     "HML": 0.006,
    #     "QMJ": -0.001
    # }], index=[today])
    # additonal_factor_df.index = pd.to_datetime(additonal_factor_df.index, format='%Y-%m-%d').date
    # additonal_factor_df.index.name = "date"

    # additonal_df = pd.DataFrame([{
    #     "index_code": "H11004.CSI",
    #     "date": today,
    #     "open": 105.55,
    #     "close": 105.55,
    #     "high": 105.55,
    #     "low": 105.55,
    #     "change_percent": 0.1,
    #     "change": 10
    # }])
    # additonal_df["date"] = pd.to_datetime(additonal_df["date"], format='%Y-%m-%d').dt.date
    # additonal_map = {"H11004.CSI": additonal_df}

    # Step 6: 执行组合优化，输出最优资产权重
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
    trade_date = datetime.strftime(TradeCalendarReader.get_trade_dates(end=datetime.strftime(datetime.today(), "%Y%m%d"))[-1], "%Y-%m-%d")
    portfolio_plan = optimize_allocation(additonal_factor_df, additonal_map, asset_source_map, code_factors_map, trade_date)

    # 输出或保存优化结果
    output_optimized_portfolio(portfolio_plan)

    return portfolio_plan

# --- 各子函数定义区域 ---

def fetch_realtime_market_data():
    """获取全市场股票实时行情数据与指数行情数据"""
    from app.data_fetcher.stock_data_reader import StockDataReader
    from app.data_fetcher.index_data_reader import IndexDataReader

    stock_reader = StockDataReader()
    index_reader = IndexDataReader()

    try:
        stock_rt = stock_reader.fetch_realtime_prices()  # 返回字段包含 stock_code, close, vol, amount
        index_rt = index_reader.fetch_realtime_prices("000985.CSI")  # 中证全指
    except Exception as e:
        import logging
        logging.exception("实时行情数据获取失败，终止组合优化流程")
        raise RuntimeError("实时数据获取失败") from e

    return stock_rt, index_rt

def compute_portfolio_returns(market_data):
    """基于实时行情数据，计算预设组合的收益率（用于构建风格因子）"""

    from app.data_fetcher.stock_data_reader import StockDataReader

    # 获取 bt_result 下最新日期目录
    bt_root = "bt_result"
    all_dates = sorted([d for d in os.listdir(bt_root) if os.path.isdir(os.path.join(bt_root, d))])
    latest_date_dir = os.path.join(bt_root, all_dates[-1])

    # 所需组合名称列表
    target_portfolios = [
        "portfolio_OP_B_H_portfolio.csv",
        "portfolio_OP_S_H_portfolio.csv",
        "portfolio_OP_B_M_portfolio.csv",
        "portfolio_OP_S_M_portfolio.csv",
        "portfolio_OP_B_L_portfolio.csv",
        "portfolio_OP_S_L_portfolio.csv",
        "portfolio_BM_B_H_portfolio.csv",
        "portfolio_BM_S_H_portfolio.csv",
        "portfolio_BM_B_M_portfolio.csv",
        "portfolio_BM_S_M_portfolio.csv",
        "portfolio_BM_B_L_portfolio.csv",
        "portfolio_BM_S_L_portfolio.csv",
    ]

    # 获取实时行情 close
    price_map_rt = market_data.set_index("stock_code")["close"].to_dict()

    # 获取昨日收盘价（缓存数据）
    stock_reader = StockDataReader()
    price_df_yesterday = stock_reader.fetch_latest_close_prices_from_cache()
    price_map_yesterday = price_df_yesterday.set_index("stock_code")["close"].to_dict()

    result = {}

    for fname in target_portfolios:
        fpath = os.path.join(latest_date_dir, fname)
        if not os.path.exists(fpath):
            continue

        df = pd.read_csv(fpath, index_col=0)
        if df.empty:
            continue
        latest_row = df.iloc[-1]

        weights = latest_row[latest_row > 0]
        codes = weights.index

        try:
            yesterday_value = sum([weights[code] * price_map_yesterday.get(code, 0) for code in codes])
            today_value = sum([weights[code] * price_map_rt.get(code, 0) for code in codes])
            portfolio_ret = today_value / yesterday_value - 1
            result[fname.replace(".csv", "")] = portfolio_ret
        except Exception as e:
            import logging
            logging.warning(f"组合收益计算失败：{fname}，错误：{e}")

    return pd.Series(result)

def calculate_intraday_factors(portfolio_returns, index_rt):
    """根据组合收益率计算实时风格因子（如 MKT、SMB、HML、QMJ）"""
    import pandas as pd
    from app.data_fetcher.index_data_reader import IndexDataReader

    def _mean_diff(group1_prefix: str, group2_prefix: str) -> float:
        g1 = portfolio_returns[[k for k in portfolio_returns.index if k.startswith(group1_prefix)]]
        g2 = portfolio_returns[[k for k in portfolio_returns.index if k.startswith(group2_prefix)]]
        return g1.mean() - g2.mean() if not g1.empty and not g2.empty else float("nan")

    # 获取中证全指昨日收盘价
    index_reader = IndexDataReader()
    index_close_df = index_reader.fetch_latest_close_prices_from_cache("000985.CSI")
    pre_close = index_close_df.loc[0, "close"]
    today_close = index_rt.loc[0, "close"]

    factors = {
        "MKT": today_close / pre_close - 1,
        "SMB": _mean_diff("portfolio_BM_S_", "portfolio_BM_B_"),
        "HML": float("nan"),
        "QMJ": float("nan"),
    }

    # 特殊处理 HML 和 QMJ 的 H/L 分组
    hml_high = portfolio_returns[[k for k in portfolio_returns.index if k.startswith("portfolio_BM_") and k.endswith("_H_portfolio")]].mean()
    hml_low = portfolio_returns[[k for k in portfolio_returns.index if k.startswith("portfolio_BM_") and k.endswith("_L_portfolio")]].mean()
    qmj_high = portfolio_returns[[k for k in portfolio_returns.index if k.startswith("portfolio_OP_") and k.endswith("_H_portfolio")]].mean()
    qmj_low = portfolio_returns[[k for k in portfolio_returns.index if k.startswith("portfolio_OP_") and k.endswith("_L_portfolio")]].mean()

    factors["HML"] = hml_high - hml_low if not pd.isna(hml_high) and not pd.isna(hml_low) else float("nan")
    factors["QMJ"] = qmj_high - qmj_low if not pd.isna(qmj_high) and not pd.isna(qmj_low) else float("nan")

    return pd.Series(factors)


def estimate_intraday_index_value(index_to_etf: dict[str, str]) -> dict[str, float]:
    """
    根据ETF价格推估盘中指数点数。

    :param index_to_etf: 映射字典 {index_code: etf_code}
    :return: {index_code: estimated_intraday_value, latest_close_price}
    """
    from app.data_fetcher.index_data_reader import IndexDataReader
    from app.data_fetcher.etf_data_reader import EtfDataReader

    result = {}
    index_reader = IndexDataReader()
    etf_reader = EtfDataReader()
    current_trade_date = TradeCalendarReader.get_trade_dates(end=datetime.strftime(datetime.today(), "%Y%m%d"))[-1]

    for index_code, etf_code in index_to_etf.items():
        # 读取指数昨日收盘点数
        idx_df = index_reader.fetch_latest_close_prices_from_cache(index_code)
        if idx_df.empty:
            continue
        idx_close = idx_df.loc[0, "close"]
        idx_date = pd.to_datetime(idx_df.loc[0, "date"])

        # 读取ETF昨日收盘价
        etf_df = etf_reader.fetch_latest_close_prices_from_cache(etf_code)
        if etf_df.empty:
            continue
        etf_close = etf_df.loc[0, "close"]
        etf_date = pd.to_datetime(etf_df.loc[0, "date"])

        # 获取ETF实时价和日期
        etf_rt = etf_reader.fetch_realtime_prices(etf_code)
        if etf_rt.empty:
            continue
        rt_price = etf_rt.loc[0, "close"]
        rt_date = pd.to_datetime(etf_rt.loc[0, "date"]) if "date" in etf_rt.columns else current_trade_date

        # 判断是否为同一交易日：若是则说明指数尚未更新，直接返回昨日指数
        if rt_date.date() == idx_date.date():
            result[index_code] = (idx_close, idx_close)
        else:
            ratio = rt_price / etf_close if etf_close else float("nan")
            result[index_code] = (idx_close * ratio, idx_close)

    return result

def build_real_time_date(intraday_factors, est_index_values):
    """
    根据实时行情构造出实时因子与估算的实时指数的数据，用于后续模型预测
    
    返回:
    - additional_factor_df: 根据当前日盘中因子收益率构造的 DataFrame
    - additional_map: dict[index_code, DataFrame] 包含估算盘中点的指数数据
    """
    from datetime import datetime
    from app.service.portfolio_opt import calculate_intraday_factors, compute_portfolio_returns, fetch_realtime_market_data
    from app.data_fetcher.trade_calender_reader import TradeCalendarReader

    current_trade_date = TradeCalendarReader.get_trade_dates(end=datetime.strftime(datetime.today(), "%Y%m%d"))[-1]
    additional_factor_df = pd.DataFrame([intraday_factors.values], columns=intraday_factors.index, index=[current_trade_date])
    additional_factor_df.index = pd.to_datetime(additional_factor_df.index, format='%Y-%m-%d').date
    additional_factor_df.index.name = "date"

    additional_map = {}
    for index_code, (value, pre_value) in est_index_values.items():
        df = pd.DataFrame({
            "index_code": [index_code],
            "date": [current_trade_date],
            "open": [value],
            "close": [value],
            "high": [value],
            "low": [value],
            "change_percent": value / pre_value - 1,
            "change": value - pre_value
        })
        df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d').dt.date
        additional_map[index_code] = df

    return additional_factor_df, additional_map


def optimize_allocation(additional_factor_df: pd.DataFrame, additional_map: dict[str, pd.DataFrame], asset_source_map: dict, code_factors_map: dict, trade_date: str, view_codes: List[str] = None):
    """根据预测收益执行组合优化，输出最优权重
        additional_factor_df: pd.DataFrame - 额外的因子数据
        additional_map: dict[str, pd.DataFrame] - 额外的数据字典
        asset_source_map: dict - 资产源映射
        code_factors_map: dict - 资产因子映射
        trade_date: str - 交易日, 格式为 YYYY-MM-DD
        view_codes: List[str] - 需要结合观点的资产代码
    """

    from app.data_fetcher.factor_data_reader import FactorDataReader
    from app.data_fetcher.csi_index_data_fetcher import CSIIndexDataFetcher
    from app.ml.dataset_builder import DatasetBuilder
    from app.dao.fund_info_dao import FundHistDao
    from app.ml.black_litterman_opt_util import load_fund_betas, compute_prior_mu_sigma, build_bl_views, compute_bl_posterior, optimize_mean_variance

    factor_data_reader = FactorDataReader(additional_df=additional_factor_df)
    csi_index_data_fetcher = CSIIndexDataFetcher(additional_map=additional_map)
    dataset_builder = DatasetBuilder(additional_factor_df=additional_factor_df, additional_map=additional_map)
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
    mu_prior, Sigma, code_list_mu = compute_prior_mu_sigma(net_value_df, window=20, method="linear")

    # 3. 构造观点（P, q, omega）（仅使用 view_codes 子集）
    if not view_codes:
        view_asset_source_map = asset_source_map
        view_code_factors_map = code_factors_map
    else:
        view_asset_source_map = {code: asset_source_map[code] for code in view_codes if code in asset_source_map}
        view_code_factors_map = {code: code_factors_map[code] for code in view_codes if code in code_factors_map}
    P, q, omega, code_list_view = build_bl_views(view_asset_source_map, view_code_factors_map, trade_date, dataset_builder)

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

def output_optimized_portfolio(portfolio_plan):
    """保存或打印最终最优组合权重"""
    print("最优组合方案: %s" % portfolio_plan)
    return
