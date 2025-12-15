from typing import List
import numpy as np
import os
import pandas as pd
from app.ai.gold_view_llm import GoldViewLLM
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
from datetime import datetime
from app.models.service_models import PortfolioWeights
from app.database import get_db
import json
import logging
from tqdm import tqdm

from app.data_fetcher.factor_data_reader import FactorDataReader
from app.data_fetcher.csi_index_data_fetcher import CSIIndexDataFetcher
from app.dao.fund_info_dao import FundHistDao
from numba import njit
from vectorbt.portfolio.enums import Direction, OrderStatus, NoOrder, CallSeqType, SizeType
from vectorbt.portfolio import nb
from app.backtest.backtest_engine import run_backtest as run_backtest_engine
from app.backtest.backtest_engine import BacktestConfig
from app.dao.betas_dao import FundBetaDao
from app.service.portfolio_crud import query_latest_portfolio_by_id
from app.data_fetcher import CalendarFetcher
from app.ml.black_litterman_opt_util import load_fund_betas, load_fund_const, compute_prior_mu_sigma, compute_prior_mu_fixed_window, build_bl_views, compute_bl_posterior, optimize_mean_variance
from app.service.portfolio_crud import query_weights_by_date, store_portfolio, query_cov_matrix_by_date
from app.service.portfolio_assets_service import get_portfolio_assets
from app.ml.factor_black_litterman import compute_asset_mu_from_factor_bl
from app.ml.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)

def _load_betas_by_code(code):
    df = FundBetaDao.select_by_code_date(code, None)
    df = df.set_index("date", drop=True)
    return df[["MKT", "SMB", "HML", "QMJ"]]

def optimize(asset_source_map: dict, code_factors_map: dict, trade_date: str, 
             post_view_tau: float,
             variance: float, window: int = 20, view_codes: List[str] = None, view_var_scale: float = 0.7, prior_mix: float = 0.3):
    factor_data_reader = FactorDataReader()
    csi_index_data_fetcher = CSIIndexDataFetcher()
    dataset_builder = DatasetBuilder()
    # 1. 根据不同数据来源使用不同的处理方式，factor/index, 使用Black Litterman 模型进行计算，cash使用滚动一年平均的收益进行计算，hist为占位符暂无计算方案
    factor_codes = [code for code, src in asset_source_map.items() if src == "factor"]
    index_codes = [code for code, src in asset_source_map.items() if src == "index"]
    hist_codes = [code for code, src in asset_source_map.items() if src == "hist"]
    cash_codes = [code for code, src in asset_source_map.items() if src == "cash"]

    fund_codes = list(asset_source_map.keys())  # 参与优化的资产列表
    net_value_df = pd.DataFrame()
    # 加载各资产的收益净值曲线
    if factor_codes:
        df_beta = load_fund_betas(factor_codes, trade_date, lookback_days=365)
        df_factors = factor_data_reader.read_daily_factors(end=trade_date)[["MKT", "SMB", "HML", "QMJ"]].dropna()
        for code in factor_codes:
            beta = df_beta.loc[code].values
            cumret = df_factors.values @ beta
            net_value_df[code] = (1 + pd.Series(cumret, index=df_factors.index)).cumprod()

    for code in index_codes:
        df = csi_index_data_fetcher.get_data_by_code_and_date(code=code, end=trade_date)
        df = df[["date", "close"]].dropna().set_index("date")
        df = df.sort_index()
        net_value_df[code] = df["close"]

    dao = FundHistDao._instance
    for code in hist_codes:
        df = dao.select_dataframe_by_code(code, end_date=trade_date)
        df = df[["date", "net_value"]].dropna().set_index("date").sort_index()
        net_value_df[code] = df["net_value"]

    for code in cash_codes:
        df = dao.select_dataframe_by_code(code, end_date=trade_date)
        df = df[["date", "net_value"]].dropna().set_index("date").sort_index()
        net_value_df[code] = df["net_value"]

    net_value_df = net_value_df.ffill().sort_index()

    # 2. 构造先验收益与协方差矩阵（使用净值曲线）
    mu_dummy, Sigma_full, code_list_mu = compute_prior_mu_sigma(
        net_value_df,
        horizon_days=window,
        lookback_years=8.0,
        method="linear",
        asset_source_map=asset_source_map,

    )
    fund_indices = [code_list_mu.index(code) for code in fund_codes]
    mu_dummy_aligned = mu_dummy[fund_indices]
    Sigma_full = Sigma_full[np.ix_(fund_indices, fund_indices)]
    mu_dict = {code: mu_dummy_aligned[i] for i, code in enumerate(fund_codes)}

    # 从code_factors_map找出10Ybond对应的指数代码
    ten_year_bond_codes = [code for code, factors in code_factors_map.items() if "10YBOND" in factors]
    if not ten_year_bond_codes:
        raise ValueError("10Ybond must have at least one asset code")
    ten_year_bond_index_code = ten_year_bond_codes[0]

    # ====== 新增：因子层 BL → 资产 μ_post ======
    # ten_year_bond_index_code 这里需要你按照实际代码填，例如 "H11006.CSI" 之类
    if post_view_tau > 0:
        mu_post_full, code_list_mu_post, mu_factor_post_dict = compute_asset_mu_from_factor_bl(
            asset_source_map=asset_source_map,
            code_factors_map=code_factors_map,
            asset_codes=view_codes,
            trade_date=trade_date,
            dataset_builder=dataset_builder,          # 你现有用于 get_softprob_dict 的实例
            ten_year_bond_index_code=ten_year_bond_index_code,       # ★ 真实的 10Y 指数代码
            horizon_days=20,
            lookback_years=8.0,
            tau=post_view_tau,
            alpha_10ybond=0.3,
            beta_lookback_days=250,
            view_var_scale=view_var_scale,
            prior_mix=prior_mix,
        )

        # 现在顺序是code_list_mu_post，与fund_codes对齐
        code_idx_map = {code: i for i, code in enumerate(code_list_mu_post)}
        for code in fund_codes:
            if code in code_idx_map:
                mu_dict[code] = mu_post_full[code_idx_map[code]]

    # 计算现金类资产的平均收益仅用滚动最近一年的数据进行计算，由于计算方式与其它资产不同，这里单独处理
    cash_net_value_df = net_value_df[cash_codes].dropna()
    cash_mu_series = compute_prior_mu_fixed_window(cash_net_value_df, window=window, lookback_days=252, method="linear")
    for code in cash_codes:
        mu_dict[code] = cash_mu_series[code]
    
    # === 使用大语言模型分析黄金期货结构，得到期望收益率，覆盖mu_post_full中的黄金收益率
    try:
        # 从code_factors_map找出GOLD对应的指数代码
        gold_codes = [code for code, factors in code_factors_map.items() if "GOLD" in factors]
        if not gold_codes:
            raise ValueError("GOLD must have at least one asset code")
        gold_code = gold_codes[0]
        gold_idx = fund_codes.index(gold_code)
        gv = GoldViewLLM()
        res = gv.generate_view(gold_code, pd.to_datetime(trade_date).date())
        if res.expected_return is not None and res.view != "no_view":
            mu_dict[gold_code] = res.expected_return
        else:
            logger.warning(f"大模型未返回 {gold_code} 的观点预测，使用先验收益率进行优化")
    except Exception as e:
        logger.error(e)
        # 大模型无法回溯历史数据进行分析，直接用先验

    mu_post_full = np.array([mu_dict[code] for code in fund_codes], dtype=float)

    # === α 调整（一次性水平项，不进入协方差）,这里的α指的是因子归因分析后剩下的常数项 ===
    fund_const = load_fund_const(factor_codes, trade_date)
    const_dict = fund_const.rolling(window=window, min_periods=window).sum().dropna(how='all').mean().to_dict()
    const_dict = _robust_clip_const(const_dict)
    lambda_alpha = 0.2  # 可调：0.1~0.3 较稳健

    # 非factor类型资产没有const，默认为0
    alpha_day = np.array([const_dict.get(code, 0.0) for code in fund_codes], dtype=float)

    # 只调整 μ，不调整 Σ，避免“alpha 一言堂”的波动放大
    mu_post_full = mu_post_full + lambda_alpha * alpha_day
    
    # 4. Max Sharpe 组合优化
    # weights, expected_return, expected_vol = optimize_max_sharpe(mu_post_full, Sigma_full)
    weights, expected_return, expected_vol = optimize_mean_variance(mu_post_full, Sigma_full, variance)

    return {
        'codes': fund_codes,
        'weights': dict(zip(fund_codes, weights)),
        'expected_return': expected_return,
        'expected_volatility': expected_vol,
        'sharpe_ratio': expected_return / expected_vol,
        'cov_matrix': Sigma_full,
    }

def optimize_portfolio_realtime(portfolio_id: int):
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

    # Step 5: 构造出实时因子数据及实时资产收益数据（用于特征构造）
    additonal_factor_df, additonal_map = build_real_time_date(realtime_factors, est_index_values)

    # Step 6: 执行组合优化，输出最优资产权重
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
    trade_date = datetime.strftime(TradeCalendarReader.get_trade_dates(end=datetime.strftime(datetime.today(), "%Y%m%d"))[-1], "%Y-%m-%d")
    portfolio_plan = optimize_allocation(additonal_factor_df, additonal_map, asset_source_map, code_factors_map, trade_date, post_view_tau, variance, view_codes)

    # 输出或保存优化结果
    w_smooth = output_optimized_portfolio(portfolio_id, portfolio_plan, alpha)

    return w_smooth

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

    # 所需组合名称列表
    target_portfolios = [
        "bm_BH_weights.csv",
        "bm_BL_weights.csv",
        "bm_BM_weights.csv",
        "bm_SH_weights.csv",
        "bm_SL_weights.csv",
        "bm_SM_weights.csv",
        "qmj_BH_weights.csv",
        "qmj_BL_weights.csv",
        "qmj_BM_weights.csv",
        "qmj_SH_weights.csv",
        "qmj_SL_weights.csv",
        "qmj_SM_weights.csv"
    ]

    # 获取实时行情 close
    price_map_rt = market_data.set_index("stock_code")["close"].to_dict()

    # 获取昨日收盘价（缓存数据）
    today = datetime.now().date()
    latest_trade_date = TradeCalendarReader.get_trade_dates(end=today.strftime("%Y-%m-%d"))[-2]  #T-1日期
    adjust_trade_date = TradeCalendarReader.get_trade_dates(end=today.strftime("%Y-%m-%d"))[-1]  #T日
    stock_reader = StockDataReader()
    price_df_yesterday = stock_reader.fetch_latest_close_prices_from_cache(latest_trade_date=latest_trade_date, adjust_trade_date=adjust_trade_date)
    price_map_yesterday = price_df_yesterday.set_index("stock_code")["close"].to_dict()

    # 实时数据中不包含停牌退市数据，根据历史数据补充
    for yesterday_key in price_map_yesterday:
        if yesterday_key not in price_map_rt:
            price_map_rt[yesterday_key] = price_map_yesterday[yesterday_key]

    result = {}

    for fname in target_portfolios:
        fpath = os.path.join(bt_root, fname)
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
            result[fname.replace("_weights.csv", "")] = portfolio_ret
        except Exception as e:
            import logging
            logging.warning(f"组合收益计算失败：{fname}，错误：{e}")

    return pd.Series(result)

def calculate_intraday_factors(portfolio_returns, index_rt):
    """根据组合收益率计算实时风格因子（如 MKT、SMB、HML、QMJ）"""
    import pandas as pd
    from app.data_fetcher.index_data_reader import IndexDataReader

    def _mean_diff(factor_group1: List[str], factor_group2: List[str]) -> float:
        g1 = portfolio_returns[factor_group1]
        g2 = portfolio_returns[factor_group2]
        return g1.mean() - g2.mean() if not g1.empty and not g2.empty else float("nan")

    # 获取中证全指昨日收盘价
    today = datetime.now().date()
    latest_trade_date = TradeCalendarReader.get_trade_dates(end=today.strftime("%Y-%m-%d"))[-2]  #T-1日期
    index_reader = IndexDataReader()
    index_close_df = index_reader.fetch_latest_close_prices("000985.CSI", latest_trade_date=latest_trade_date)
    pre_close = index_close_df.loc[0, "close"]
    today_close = index_rt.loc[0, "close"]

    factors = {
        "MKT": today_close / pre_close - 1,
        "SMB": _mean_diff(["bm_SH", "bm_SM", "bm_SL", "qmj_SH", "qmj_SM", "qmj_SL"], ["bm_BH", "bm_BM", "bm_BL", "qmj_BH", "qmj_BM", "qmj_BL"]),
        "HML": _mean_diff(["bm_BH", "bm_SH"], ["bm_BL", "bm_SL"]),
        "QMJ": _mean_diff(["qmj_BH", "qmj_SH"], ["qmj_BL", "qmj_SL"]),
    }

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
    today = datetime.now().date()
    latest_trade_date = TradeCalendarReader.get_trade_dates(end=today.strftime("%Y-%m-%d"))[-2]  #T-1日期
    index_reader = IndexDataReader()
    etf_reader = EtfDataReader()
    current_trade_date = TradeCalendarReader.get_trade_dates(end=datetime.strftime(datetime.today(), "%Y%m%d"))[-1]

    for index_code, etf_code in index_to_etf.items():
        # 读取指数昨日收盘点数
        idx_df = index_reader.fetch_latest_close_prices(index_code, latest_trade_date=latest_trade_date)
        if idx_df.empty:
            continue
        idx_close = idx_df.loc[0, "close"]
        idx_date = pd.to_datetime(idx_df.loc[0, "date"])

        # 读取ETF昨日收盘价
        etf_df = etf_reader.fetch_latest_close_prices(etf_code, latest_trade_date=latest_trade_date)
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
        if idx_date.date() != etf_date.date():
            logging.error(f"指数和ETF数据未同步，请确保二者最新数据更新到同一日期")
            raise Exception("指数和ETF数据未同步")
        if rt_date.date() == idx_date.date():
            # 若是同一交易日则说明实时行情已收盘，直接返回最新指数
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


def optimize_allocation(additional_factor_df: pd.DataFrame, additional_map: dict[str, pd.DataFrame], asset_source_map: dict, code_factors_map: dict, 
                        trade_date: str, post_view_tau: float, variance: float, view_codes: List[str] = None, window: int = 20, view_var_scale: float = 0.7, prior_mix: float = 0.3):
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
    from app.dao.fund_info_dao import FundHistDao
    from app.ml.black_litterman_opt_util import load_fund_betas, compute_prior_mu_sigma, compute_prior_mu_fixed_window, build_bl_views, compute_bl_posterior, optimize_mean_variance

    factor_data_reader = FactorDataReader(additional_df=additional_factor_df)
    csi_index_data_fetcher = CSIIndexDataFetcher(additional_map=additional_map)
    dataset_builder = DatasetBuilder(additional_factor_df=additional_factor_df, additional_map=additional_map)
    # 1. 根据不同数据来源使用不同的处理方式，factor/index, 使用Black Litterman 模型进行计算，cash使用滚动一年平均的收益进行计算，hist为占位符暂无计算方案
    factor_codes = [code for code, src in asset_source_map.items() if src == "factor"]
    index_codes = [code for code, src in asset_source_map.items() if src == "index"]
    hist_codes = [code for code, src in asset_source_map.items() if src == "hist"]
    cash_codes = [code for code, src in asset_source_map.items() if src == "cash"]

    fund_codes = list(asset_source_map.keys())  # 参与优化的资产列表
    net_value_df = pd.DataFrame()
    # 加载各资产的收益净值曲线
    if factor_codes:
        df_beta = load_fund_betas(factor_codes, trade_date, lookback_days=365)
        df_factors = factor_data_reader.read_daily_factors(end=trade_date)[["MKT", "SMB", "HML", "QMJ"]].dropna()
        for code in factor_codes:
            beta = df_beta.loc[code].values
            cumret = df_factors.values @ beta
            net_value_df[code] = (1 + pd.Series(cumret, index=df_factors.index)).cumprod()

    for code in index_codes:
        df = csi_index_data_fetcher.get_data_by_code_and_date(code=code, end=trade_date)
        df = df[["date", "close"]].dropna().set_index("date")
        df = df.sort_index()
        net_value_df[code] = df["close"]

    dao = FundHistDao._instance
    for code in hist_codes:
        df = dao.select_dataframe_by_code(code, end_date=trade_date)
        df = df[["date", "net_value"]].dropna().set_index("date").sort_index()
        net_value_df[code] = df["net_value"]

    for code in cash_codes:
        df = dao.select_dataframe_by_code(code, end_date=trade_date)
        df = df[["date", "net_value"]].dropna().set_index("date").sort_index()
        net_value_df[code] = df["net_value"]

    net_value_df = net_value_df.ffill().sort_index()

    # 2. 构造先验收益与协方差矩阵（使用净值曲线）
    mu_dummy, Sigma_full, code_list_mu = compute_prior_mu_sigma(
        net_value_df,
        horizon_days=window,
        lookback_years=8.0,
        method="linear",
        asset_source_map=asset_source_map,

    )
    fund_indices = [code_list_mu.index(code) for code in fund_codes]
    mu_dummy_aligned = mu_dummy[fund_indices]
    Sigma_full = Sigma_full[np.ix_(fund_indices, fund_indices)]
    mu_dict = {code: mu_dummy_aligned[i] for i, code in enumerate(fund_codes)}

    # 从code_factors_map找出10Ybond对应的指数代码
    ten_year_bond_codes = [code for code, factors in code_factors_map.items() if "10YBOND" in factors]
    if not ten_year_bond_codes:
        raise ValueError("10Ybond must have at least one asset code")
    ten_year_bond_index_code = ten_year_bond_codes[0]

    # ====== 新增：因子层 BL → 资产 μ_post ======
    # ten_year_bond_index_code 这里需要你按照实际代码填，例如 "H11006.CSI" 之类
    if post_view_tau > 0:
        mu_post_full, code_list_mu_post, mu_factor_post_dict = compute_asset_mu_from_factor_bl(
            asset_source_map=asset_source_map,
            code_factors_map=code_factors_map,
            asset_codes=view_codes,
            trade_date=trade_date,
            dataset_builder=dataset_builder,          # 你现有用于 get_softprob_dict 的实例
            ten_year_bond_index_code=ten_year_bond_index_code,       # ★ 真实的 10Y 指数代码
            horizon_days=20,
            lookback_years=8.0,
            tau=post_view_tau,
            alpha_10ybond=0.3,
            beta_lookback_days=250,
            view_var_scale=view_var_scale,
            prior_mix=prior_mix,
        )

        # 现在顺序是code_list_mu_post，与fund_codes对齐
        code_idx_map = {code: i for i, code in enumerate(code_list_mu_post)}
        for code in fund_codes:
            if code in code_idx_map:
                mu_dict[code] = mu_post_full[code_idx_map[code]]

    # 计算现金类资产的平均收益仅用滚动最近一年的数据进行计算，由于计算方式与其它资产不同，这里单独处理
    cash_net_value_df = net_value_df[cash_codes].dropna()
    cash_mu_series = compute_prior_mu_fixed_window(cash_net_value_df, window=window, lookback_days=252, method="linear")
    for code in cash_codes:
        mu_dict[code] = cash_mu_series[code]
    
    # === 使用大语言模型分析黄金期货结构，得到期望收益率，覆盖mu_post_full中的黄金收益率
    try:
        # 从code_factors_map找出GOLD对应的指数代码
        gold_codes = [code for code, factors in code_factors_map.items() if "GOLD" in factors]
        if not gold_codes:
            raise ValueError("GOLD must have at least one asset code")
        gold_code = gold_codes[0]
        gold_idx = fund_codes.index(gold_code)
        gv = GoldViewLLM()
        res = gv.generate_view(gold_code, pd.to_datetime(trade_date).date())
        if res.expected_return is not None and res.view != "no_view":
            mu_dict[gold_code] = res.expected_return
        else:
            logger.warning(f"大模型未返回 {gold_code} 的观点预测，使用先验收益率进行优化")
    except Exception as e:
        logger.error(e)
        # 大模型无法回溯历史数据进行分析，直接用先验

    mu_post_full = np.array([mu_dict[code] for code in fund_codes], dtype=float)

    # === α 调整（一次性水平项，不进入协方差）,这里的α指的是因子归因分析后剩下的常数项 ===
    fund_const = load_fund_const(factor_codes, trade_date)
    const_dict = fund_const.rolling(window=window, min_periods=window).sum().dropna(how='all').mean().to_dict()
    const_dict = _robust_clip_const(const_dict)
    lambda_alpha = 0.2  # 可调：0.1~0.3 较稳健

    # 非factor类型资产没有const，默认为0
    alpha_day = np.array([const_dict.get(code, 0.0) for code in fund_codes], dtype=float)

    # 只调整 μ，不调整 Σ，避免“alpha 一言堂”的波动放大
    mu_post_full = mu_post_full + lambda_alpha * alpha_day
    
    # 4. Max Sharpe 组合优化
    # weights, expected_return, expected_vol = optimize_max_sharpe(mu_post_full, Sigma_full)
    weights, expected_return, expected_vol = optimize_mean_variance(mu_post_full, Sigma_full, variance)

    return {
        'codes': fund_codes,
        'weights': dict(zip(fund_codes, weights)),
        'expected_return': expected_return,
        'expected_volatility': expected_vol,
        'sharpe_ratio': expected_return / expected_vol,
        'cov_matrix': Sigma_full,
    }

def output_optimized_portfolio(portfolio_id: int, portfolio_plan: dict, alpha: float = 0.1):
    """保存或打印最终最优组合权重，并将 ewma 平滑后的结果写入数据库"""
    # 获取当前和前一交易日
    trade_dates = TradeCalendarReader.get_trade_dates(end=datetime.today().strftime("%Y%m%d"))
    if len(trade_dates) < 2:
        raise ValueError("交易日不足，无法执行权重平滑")

    prev_date = trade_dates[-2]
    today_date = trade_dates[-1]

    # 查询昨日权重
    w_prev = query_weights_by_date(date=prev_date, portfolio_id=portfolio_id)["weights"]

    # 当前权重
    w_today = portfolio_plan["weights"]

    # 合并资产列表
    all_assets = set(w_today.keys()).union(w_prev.keys())

    # 计算平滑权重
    w_smooth = {
        code: round(alpha * w_today.get(code, 0.0) + (1 - alpha) * w_prev.get(code, 0.0), 8)
        for code in all_assets
    }

    cov_matrix = portfolio_plan['cov_matrix']
    codes = portfolio_plan['codes']
    additional_assets = [code for code in all_assets if code not in portfolio_plan['codes']]

    # 将cov_matrix和codes按合并后的合并资产列表扩展，cov_matrix多出来的位置填充0
    additional_size = len(additional_assets)
    cov_matrix = np.pad(cov_matrix, ((0, additional_size), (0, additional_size)), 'constant', constant_values=0.0)
    codes = codes + additional_assets

    store_portfolio(portfolio_id, pd.to_datetime("today").strftime("%Y-%m-%d"), w_today, w_smooth, cov_matrix, codes)
    return w_smooth


factor_data_reader = FactorDataReader()
csi_index_data_fetcher = CSIIndexDataFetcher()

def build_price_df(asset_source_map: dict, start: str, end: str) -> pd.DataFrame:
    """
    构造组合资产的净值曲线：factor资产使用因子暴露生成，index资产使用真实指数行情
    """
    df_factors = factor_data_reader.read_daily_factors(start=start, end=end)[["MKT", "SMB", "HML", "QMJ"]].dropna()

    net_value_df = pd.DataFrame(index=df_factors.index)
    dao = FundHistDao._instance
    for code, src in asset_source_map.items():
        if src == "factor":
            beta_df = _load_betas_by_code(code)
            if beta_df.empty:
                logger.warning(f"⚠️ {code} 因子暴露缺失，跳过")
                continue
            beta_df = beta_df.reindex(df_factors.index).bfill()
            ret = (beta_df * df_factors).sum(axis=1)
            net_value_df[code] = (1 + pd.Series(ret, index=df_factors.index)).cumprod()
        elif src == "index":
            df = csi_index_data_fetcher.get_data_by_code_and_date(code=code)
            df = df[["date", "close"]].dropna().set_index("date").sort_index()
            net_value_df[code] = df["close"]
        elif src == "hist":
            df = dao.select_dataframe_by_code(code)
            df = df[["date", "net_value"]].dropna().set_index("date").sort_index()
            net_value_df[code] = df["net_value"]
        elif src == "cash":
            df = dao.select_dataframe_by_code(code)
            df = df[["date", "net_value"]].dropna().set_index("date").sort_index()
            net_value_df[code] = df["net_value"]
        else:
            logger.warning(f"未知资产类型 {code}: {src}")

    net_value_df = net_value_df.dropna(how='all')
    net_value_df = net_value_df / net_value_df.iloc[0]
    return net_value_df.ffill()


def optimize_portfolio_history(portfolio_id: int, start_date: str = None, end_date: str = None):
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

    if not end_date:
        end_date = CalendarFetcher().get_trade_date(end=pd.to_datetime("today").strftime("%Y%m%d"), format="%Y-%m-%d", limit=1, ascending=False)[0]

    if not start_date:
        latest_portfolio = query_latest_portfolio_by_id(portfolio_id)
        logging.info("🔁 获取最后一次优化的组合权重")
        if latest_portfolio.empty:
            raise ValueError("没有历史优化权重")
        prev_weights = latest_portfolio["weights_ewma"]
            
        # start_date = (pd.to_datetime(latest_portfolio["date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = CalendarFetcher().get_next_trade_date(pd.to_datetime(latest_portfolio["date"]).strftime("%Y%m%d"), format="%Y-%m-%d")
        logging.info(f"使用最后一次优化的日期的T+1日 {start_date} 作为起始日期")
    else:
        latest_portfolio = query_latest_portfolio_by_id(portfolio_id, start_date)
        if latest_portfolio.empty:
            prev_weights = None
            logger.warning(f"⚠️ {portfolio_id} 没有历史优化权重")
        else:
            prev_weights = latest_portfolio["weights_ewma"]

    all_dates = CalendarFetcher().get_trade_date(start=start_date.replace("-", ""), end=end_date.replace("-", ""), format="%Y-%m-%d", ascending=True)

    logging.info("📊 开始每日 optimize + ewma 平滑 + 入库")
    for dt in tqdm(all_dates):
        try:
            optimize_result = optimize(
                asset_source_map=asset_source_map,
                code_factors_map=code_factors_map,
                trade_date=dt,
                post_view_tau=post_view_tau,
                variance=variance,
                window=20,
                view_codes=view_codes
            )

            w_today = optimize_result["weights"]
            cov_matrix = optimize_result["cov_matrix"]
            codes = optimize_result["codes"]
            
            if prev_weights is None:
                prev_weights = w_today
            
            all_codes = set(w_today.keys()).union(prev_weights.keys())

            w_ewma = {
                code: round(alpha * w_today.get(code, 0.0) + (1 - alpha) * prev_weights.get(code, 0.0), 8)
                for code in all_codes
            }

            additional_assets = [code for code in all_codes if code not in optimize_result["codes"]]
            # 将cov_matrix和codes按合并后的合并资产列表扩展，cov_matrix多出来的位置填充0
            additional_size = len(additional_assets)
            cov_matrix = np.pad(cov_matrix, ((0, additional_size), (0, additional_size)), 'constant', constant_values=0.0)
            codes = codes + additional_assets

            prev_weights = w_ewma.copy()

            store_portfolio(portfolio_id, dt, w_today, w_ewma, cov_matrix, codes)

        except Exception as e:
            logger.warning(f"⚠️ {dt} 调仓失败: {e}")
            continue

def compute_diverge(portfolio_id: int, trade_date: str, current_w: dict, target_w: dict) -> float:
    """
    计算当前组合和目标组合之间的偏离度（跟踪误差）
    TE = sqrt((w - w*)^T Σ (w - w*))

    :param portfolio_id: 组合 ID
    :param trade_date: 日期（格式 'YYYY-MM-DD'）
    :param current_w: 当前组合权重 dict[asset] = weight
    :param target_w: 目标组合权重 dict[asset] = weight
    :return: 跟踪误差（Tracking Error）
    """
    codes, cov = query_cov_matrix_by_date(trade_date, portfolio_id)

    # 将当前和目标权重映射到 codes 顺序的向量（没有的设为 0）
    current_vec = np.array([current_w.get(code, 0.0) for code in codes])
    target_vec = np.array([target_w.get(code, 0.0) for code in codes])

    # 计算差异向量
    diff = current_vec - target_vec

    # TE = sqrt(diff.T * Σ * diff)
    tracking_error = np.sqrt(diff.T @ cov @ diff)

    return tracking_error

def _robust_clip_const(const_dict: dict) -> dict:
    """
    输入 const_dict（{code: const}），将所有 value 按 (均值 ± 3*标准差) 做裁剪，
    返回裁剪后的新 dict。不改变原 dict。

    说明：
    - 均值与标准差使用 np.nanmean / np.nanstd 计算（忽略不可转为 float 或 NaN 的值）。
    - 对于无法转成 float 的值（如 None、字符串非数值），原样保留。
    """
    if not const_dict:
        return {}

    # 收集数值，无法转 float 的记为 NaN（后续在 mean/std 中忽略）
    vals = []
    for v in const_dict.values():
        try:
            vals.append(float(v))
        except Exception:
            vals.append(np.nan)

    # 若全是 NaN，则直接返回原 dict
    if not np.isfinite(np.nanmean(vals)):
        return dict(const_dict)

    mu = float(np.nanmean(vals))
    sigma = float(np.nanstd(vals))
    lo, hi = mu - 2.0 * sigma, mu + 2.0 * sigma

    # 逐项裁剪；不可转 float 的值原样返回
    clipped = {}
    for k, v in const_dict.items():
        try:
            x = float(v)
            # np.clip 对 NaN 会返回 NaN，这里保持一致
            clipped[k] = float(np.clip(x, lo, hi)) if np.isfinite(x) else x
        except Exception:
            clipped[k] = v

    return clipped