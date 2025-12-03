import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime
from tqdm import tqdm
import os
import logging

from app import create_app
from scripts.optimize_portfolio import optimize
from app.data_fetcher.factor_data_reader import FactorDataReader
from app.data_fetcher.csi_index_data_fetcher import CSIIndexDataFetcher
from app.dao.fund_info_dao import FundHistDao
from numba import njit
from vectorbt.portfolio.enums import Direction, OrderStatus, NoOrder, CallSeqType, SizeType
from vectorbt.portfolio import nb
from app.database import get_db
from app.models.service_models import PortfolioWeights
import json
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
from app.backtest.backtest_engine import run_backtest as run_backtest_engine
from app.backtest.backtest_engine import BacktestConfig
from app.dao.betas_dao import FundBetaDao
from app.service.portfolio_crud import query_weights_by_date, store_portfolio
from app.service.portfolio_assets_service import get_portfolio_assets
from app.service.portfolio_opt import compute_diverge

app = create_app()
logger = logging.getLogger(__name__)
factor_data_reader = FactorDataReader()
csi_index_data_fetcher = CSIIndexDataFetcher()
sell_fee_rate = 0.0005
slippage_rate = 0.0
portfolio_id = 1

# 资产配置
asset_info = get_portfolio_assets(portfolio_id)
asset_source_map = asset_info["asset_source_map"]
code_factors_map = asset_info["code_factors_map"]
for code, src in asset_source_map.items():
    if src == "factor":
        code_factors_map[code] = ["MKT", "SMB", "HML", "QMJ"]

view_codes = asset_info["view_codes"]

params = asset_info["params"]
if params is None or "post_view_tau" not in params or "alpha" not in params or "variance" not in params:
    raise Exception("Invalid params, please set post_view_tau and alpha and variance in params")
post_view_tau = float(params["post_view_tau"])
view_var_scale= float(params["view_var_scale"])
prior_mix=float(params["prior_mix"])
variance = float(params["variance"])
alpha = float(params["alpha"])

def load_fund_betas(code):
    df = FundBetaDao.select_by_code_date(code, None)
    df = df.set_index("date", drop=True)
    return df[["MKT", "SMB", "HML", "QMJ"]]

def load_latest_fund_betas(codes):
    one_year_ago = (pd.to_datetime("today") - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    df = FundBetaDao.get_latest_fund_betas(fund_type_list=["股票型"], invest_type_list=["被动指数型", "增强指数型"], found_date_limit=one_year_ago)
    df = df.set_index("code", drop=True)
    df = df[df.index.isin(codes)]
    return df[["MKT", "SMB", "HML", "QMJ"]]


def build_price_df(asset_source_map: dict, start: str, end: str) -> pd.DataFrame:
    """
    构造组合资产的净值曲线：factor资产使用因子暴露生成，index资产使用真实指数行情
    """
    df_factors = factor_data_reader.read_daily_factors(start=start, end=end)[["MKT", "SMB", "HML", "QMJ"]].dropna()

    net_value_df = pd.DataFrame(index=df_factors.index)
    dao = FundHistDao._instance
    for code, src in asset_source_map.items():
        if src == "factor":
            beta_df = load_fund_betas(code)
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
    net_value_df = net_value_df / net_value_df.bfill().iloc[0]
    return net_value_df.ffill()


def run_backtest(start="2019-12-22", end="2024-12-22", window=20):
    out_dir = f"./fund_portfolio_bt_result/{datetime.today().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    with app.app_context():
        print("🔁 开始构建价格数据")
        price_df = build_price_df(asset_source_map, start, end)
        all_dates = price_df.index
        assets = list(asset_source_map.keys())
        weights_dict = {d: {} for d in all_dates}

        prev_weights = None  # 上一日的平滑权重

        print("📊 开始每日 optimize + ewma 平滑 + 入库")
        for dt in tqdm(all_dates):
            try:
                portfolio_plan = optimize(
                    asset_source_map=asset_source_map,
                    code_factors_map=code_factors_map,
                    trade_date=dt.strftime('%Y-%m-%d'),
                    post_view_tau = post_view_tau,
                    variance = variance,
                    window=window,
                    view_codes=view_codes,
                    view_var_scale= 0.7, prior_mix=0.3
                )

                w_today = portfolio_plan["weights"]
                cov_matrix = portfolio_plan["cov_matrix"]
                codes = portfolio_plan["codes"]

                # 如果第一天，从数据库尝试读取前一日平滑值
                if prev_weights is None:
                    trade_dates = TradeCalendarReader.get_trade_dates(end=dt.strftime("%Y-%m-%d"))
                    if len(trade_dates) >= 2:
                        prev_trade_date = trade_dates[-2]
                    else:
                        raise ValueError("交易日不足，无法执行权重平滑")
                    
                    prev_weights = query_weights_by_date(prev_trade_date, portfolio_id)["weights"]

                all_codes = set(w_today.keys()).union(prev_weights.keys())
                w_ewma = {
                    code: round(alpha * w_today.get(code, 0.0) + (1 - alpha) * prev_weights.get(code, 0.0), 8)
                    for code in all_codes
                }

                additional_assets = [code for code in all_codes if code not in portfolio_plan["codes"]]
                # 将cov_matrix和codes按合并后的合并资产列表扩展，cov_matrix多出来的位置填充0]
                additional_size = len(additional_assets)
                cov_matrix = np.pad(cov_matrix, ((0, additional_size), (0, additional_size)), 'constant', constant_values=0.0)
                codes = codes + additional_assets

                weights_dict[dt] = pd.Series(w_ewma)
                prev_weights = w_ewma.copy()
                
                store_portfolio(portfolio_id, dt, w_today, w_ewma, cov_matrix, codes)

            except Exception as e:
                logger.warning(f"⚠️ {dt.strftime('%Y-%m-%d')} 调仓失败: {e}")
                continue

        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index')
        weights_df = weights_df.dropna(how='all').fillna(0)
        price_df = price_df.loc[weights_df.index]

        

        # === 处理权重序列 ===
        d = 0.005  # 偏差百分比阈值
        rebalance_dates = [weights_df.index[0]]
        prev_weight = weights_df.iloc[0]

        for date, current_weight in weights_df.iloc[1:].iterrows():
            avg_ratio = compute_diverge(portfolio_id=portfolio_id, trade_date=date, current_w=prev_weight, target_w=current_weight)

            if avg_ratio > d:
                rebalance_dates.append(date)
                prev_weight = current_weight

        rebalance_dates = pd.DatetimeIndex(rebalance_dates)
        weights_df = weights_df.loc[rebalance_dates]
        weights_df.to_csv(os.path.join(out_dir, "portfolio_weights.csv"))

        print("🏁 开始构建 Portfolio")

        cfg = BacktestConfig(
            init_cash=100_000_000,
            buy_fee=0.0,
            sell_fee=sell_fee_rate,
            slippage=slippage_rate,
            cash_sharing=True
        )
        result = run_backtest_engine(weights_df, price_df, cfg)

        # 5. 日换手率计算函数
        def compute_turnover_rate(portfolio, n: int = 1) -> pd.Series:
            """
            计算 n 日平均换手率（n=1 为日换手率），适用于 portfolio.value() 返回 Series 的情况。
            """
            orders = portfolio.orders.records_readable.copy()

            # 计算每笔交易金额
            if 'trade_value' not in orders.columns:
                orders['trade_value'] = orders['Size'].abs() * orders['Price']

            # 确保 Timestamp 是 datetime 类型
            orders['date'] = pd.to_datetime(orders['Timestamp'])
            daily_trade_value = orders.groupby('date')['trade_value'].sum()

            # 获取组合每日市值（Series）
            portfolio_value = portfolio.value()
            portfolio_value.index = pd.to_datetime(portfolio_value.index)

            # 对齐后计算换手率
            aligned = pd.concat([daily_trade_value, portfolio_value], axis=1, join='inner')
            aligned.columns = ['turnover_amt', 'portfolio_value']
            aligned['turnover'] = aligned['turnover_amt'] / aligned['portfolio_value']

            # 返回滚动换手率
            return aligned['turnover'].rolling(n).mean()

        # 6. 结果输出
        # stats = pf.stats()
        turnover_rate = compute_turnover_rate(result["pf"], n=1)
        mean_turnover = turnover_rate[1:].mean() #去掉建仓首日的换手率

        # print(stats)
        print(f"\n🔄 日均换手率: {mean_turnover:.4f}")

        result["nav"].to_csv(os.path.join(out_dir, "portfolio_value.csv"))
        result["returns"].to_csv(os.path.join(out_dir, "daily_returns.csv"))
        # stats.to_csv(os.path.join(out_dir, "stats.csv"))
        turnover_rate.to_csv(os.path.join(out_dir, "daily_turnover.csv"))
        result["nav"].vbt.plot(title="混合资产经债线").write_html(os.path.join(out_dir, "value_plot.html"))

        print(f"✅ 回测完成，结果已保存至：{out_dir}")

from app.service.portfolio_crud import query_all_weights_by_date

def run_backtest_using_db_weights(
    *,
    portfolio_id: int,
    start: str,
    end: str,
    d_threshold: float = 0.005,     # 偏差阈值，超过则触发调仓
    ensure_trade_calendar_union: bool = True  # 价格按权重日期并齐，缺失用前值（0收益）
):
    """
    使用数据库中已存的平滑权重（weights_ewma）+ 真实资产净值进行回测。
    - 仅使用 PortfolioWeights.weights_ewma
    - 回测时间区间由 start/end 指定（含端点）
    - 若某天某资产缺 NAV：按“0收益”处理 → 前值填充（ffill）
    - 调仓频率：当偏差超过阈值时调仓；当资产代码集合发生变化时必调仓
    - 价格来源：对涉及到的全部资产构建 asset_source_map：
        "H11004.CSI" 与 "Au99.99.SGE" → "index"
        "270004.OF" → "cash"
        其他代码 → "hist"
      build_price_df 会为 "hist" 返回复权累计净值（无需额外处理）

    导出：
    - portfolio_weights.csv  （实际参与回测的调仓日权重）
    - portfolio_value.csv    （组合净值）
    - daily_returns.csv      （日收益率）
    - daily_turnover.csv     （日换手率）
    - value_plot.html        （净值曲线）
    - weights_from_db.csv    （DB读出的完整日度权重明细，便于核对）
    """
    # === 0) 输出目录 ===
    out_dir = f"./fund_portfolio_bt_result/{datetime.today().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    # === 1) 从 DB 读取权重（仅 weights_ewma）===
    print("🗃️ 从数据库读取权重（weights_ewma）...")
    df_w_all = query_all_weights_by_date(
        portfolio_id=portfolio_id,
        start_date=start,
        end_date=end,
        fill_missing_zero=True  # 缺失资产权重=0，便于对齐
    )
    if df_w_all.empty:
        raise ValueError(f"在区间 {start}~{end} 内未读取到任何权重记录（portfolio_id={portfolio_id}）。")

    # 保存原始（从DB读出）的日度权重表，便于核对
    df_w_all.sort_index().to_csv(os.path.join(out_dir, "weights_from_db.csv"))
    print(f"✅ 已导出 weights_from_db.csv，共 {len(df_w_all)} 行。")

    # === 2) 构造 asset_source_map（只覆盖回测区间出现过的资产代码）===
    all_codes = set(df_w_all.columns)
    def _code_to_type(code: str) -> str:
        if code in {"H11004.CSI", "Au99.99.SGE"}:
            return "index"
        if code == "270004.OF":
            return "cash"
        return "hist"

    asset_source_map = {code: _code_to_type(code) for code in all_codes}

    # === 3) 构建真实净值 price_df，并按“0收益”规则对齐 ===
    print("💰 构建资产净值曲线（build_price_df）...")
    price_df = build_price_df(asset_source_map, start, end)
    if price_df is None or price_df.empty:
        raise ValueError("price_df 为空，无法回测。")

    # 仅保留权重涉及到的资产列
    missing_cols = all_codes - set(price_df.columns)
    if missing_cols:
        # 对于缺失价格的资产，新建列并用 NaN，后续 ffill
        for c in missing_cols:
            price_df[c] = np.nan
        price_df = price_df[df_w_all.columns]  # 列顺序与权重对齐
    else:
        price_df = price_df[df_w_all.columns]

    # 将价格索引限制在 DB 权重的日期范围内的交集
    # 注意：根据你的规则，若某日某资产无 NAV，0 收益 → ffill
    if ensure_trade_calendar_union:
        # 以“权重日期”为主，强制把价格 reindex 到权重日期上，再 ffill
        price_df = price_df.reindex(df_w_all.index).sort_index()

    # 前向填充：缺失价格用最近一次价格（即 0 收益）
    price_df = price_df.ffill()

    # 若起始日仍有 NaN（前一日完全没有历史价格），可用首个非 NaN 值向后填（避免回测引擎报错）
    price_df = price_df.fillna(method="bfill", axis=0)

    # 再次校验
    if price_df.isna().any().any():
        # 仍存在 NaN，说明某些资产在整个区间都没有价格；把其对应权重强制置 0
        cols_all_nan = [c for c in price_df.columns if price_df[c].isna().all()]
        if cols_all_nan:
            logger.warning(f"这些资产在区间内没有任何价格数据，将在权重中置零并从价格中剔除: {cols_all_nan}")
            df_w_all[cols_all_nan] = 0.0
            price_df = price_df.drop(columns=cols_all_nan)

    # 再次对齐列
    shared_cols = [c for c in df_w_all.columns if c in price_df.columns]
    df_w_all = df_w_all[shared_cols]
    price_df = price_df[shared_cols]

    # === 4) 处理权重序列 → 选择调仓日 ===
    print("🧭 选择调仓日（偏差阈值 & 代码集合变化必调仓）...")
    rebalance_dates = []
    prev_weight = None
    prev_cols_set = None

    for date, w_row in df_w_all.iterrows():
        w_row = w_row.fillna(0.0)

        # 第一天必调仓
        if prev_weight is None:
            rebalance_dates.append(date)
            prev_weight = w_row
            prev_cols_set = set(w_row.index[w_row.values != 0.0])
            continue

        # 1) 若代码集合变化：必调仓
        curr_cols_set = set(w_row.index[w_row.values != 0.0])
        if curr_cols_set != prev_cols_set:
            rebalance_dates.append(date)
            prev_weight = w_row
            prev_cols_set = curr_cols_set
            continue

        # 2) 偏差阈值：使用你已有的 compute_diverge（保持逻辑一致）
        avg_ratio = compute_diverge(
            portfolio_id=portfolio_id,
            trade_date=date,
            current_w=prev_weight,
            target_w=w_row
        )
        if avg_ratio > d_threshold:
            rebalance_dates.append(date)
            prev_weight = w_row
            prev_cols_set = curr_cols_set

    rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    weights_df = df_w_all.loc[rebalance_dates].copy()

    # 导出用于回测的调仓权重
    weights_df.to_csv(os.path.join(out_dir, "portfolio_weights.csv"))
    print(f"✅ 调仓日共 {len(weights_df)} 个，已导出 portfolio_weights.csv。")

    # 对齐 price_df 的时间索引（引擎通常需要完整的日度价格序列）
    # price_df 已经 reindex 到 df_w_all.index 并 ffill，这里确保区间完整：
    price_df = price_df.loc[df_w_all.index.min(): df_w_all.index.max()]

    # === 5) 运行回测引擎 ===
    print("🚀 运行回测引擎...")
    cfg = BacktestConfig(
        init_cash=100_000_000,
        buy_fee=0.0,
        sell_fee=sell_fee_rate,
        slippage=slippage_rate,
        cash_sharing=True
    )
    result = run_backtest_engine(weights_df, price_df, cfg)

    # === 7) 导出结果 ===
    result["nav"].to_csv(os.path.join(out_dir, "portfolio_value.csv"))
    result["returns"].to_csv(os.path.join(out_dir, "daily_returns.csv"))
    result["nav"].vbt.plot(title="组合净值曲线").write_html(os.path.join(out_dir, "value_plot.html"))

    print(f"✅ 回测完成，结果已保存至：{out_dir}")

    return {
        "out_dir": out_dir,
        "rebalance_dates": rebalance_dates,
        "nav": result["nav"],
        "returns": result["returns"],
    }


if __name__ == '__main__':
    # run_backtest_using_db_weights(portfolio_id=2, start='2025-09-02', end='2025-09-25')
    run_backtest(start="2019-10-22", end="2025-10-22", window=20)