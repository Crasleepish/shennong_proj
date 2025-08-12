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

app = create_app()
logger = logging.getLogger(__name__)
factor_data_reader = FactorDataReader()
csi_index_data_fetcher = CSIIndexDataFetcher()
sell_fee_rate = 0.0005
slippage_rate = 0.0
portfolio_id = 1
alpha = 0.1

# 资产配置
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
}
for code, src in asset_source_map.items():
    if src == "factor":
        code_factors_map[code] = ["MKT", "SMB", "HML", "QMJ"]

view_codes = ["H11004.CSI", "Au99.99.SGE", "008114.OF", "020602.OF", "019918.OF", "002236.OF", "019311.OF", "006712.OF", "011041.OF", "110003.OF", "019702.OF", "006342.OF", "020466.OF", "018732.OF"]
        
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
    net_value_df = net_value_df / net_value_df.iloc[0]
    return net_value_df.ffill()


def run_backtest(start="2025-08-07", end="2025-08-08", window=20):
    out_dir = f"./fund_portfolio_bt_result/{datetime.today().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    with app.app_context():
        print("🔁 开始构建价格数据")
        price_df = build_price_df(asset_source_map, start, end)
        all_dates = price_df.index
        assets = list(asset_source_map.keys())
        weights_df = pd.DataFrame(index=all_dates, columns=assets)

        prev_weights = None  # 上一日的平滑权重

        print("📊 开始每日 optimize + ewma 平滑 + 入库")
        for dt in tqdm(all_dates):
            try:
                portfolio_plan = optimize(
                    asset_source_map=asset_source_map,
                    code_factors_map=code_factors_map,
                    trade_date=dt.strftime('%Y-%m-%d'),
                    window=window,
                    view_codes=view_codes
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
                    
                    prev_weights = query_weights_by_date(prev_trade_date)

                all_codes = set(w_today.keys()).union(prev_weights.keys())
                w_ewma = {
                    code: round(alpha * w_today.get(code, 0.0) + (1 - alpha) * prev_weights.get(code, 0.0), 8)
                    for code in all_codes
                }

                weights_df.loc[dt] = pd.Series(w_ewma)
                prev_weights = w_ewma.copy()
                
                store_portfolio(portfolio_id, dt, w_today, w_ewma, cov_matrix, codes)

            except Exception as e:
                logger.warning(f"⚠️ {dt.strftime('%Y-%m-%d')} 调仓失败: {e}")
                continue

        weights_df = weights_df.infer_objects(copy=False).dropna(how='all').fillna(0)
        price_df = price_df.loc[weights_df.index]

        # === 处理权重序列 ===
        d = 0.12  # 偏差百分比阈值
        rebalance_dates = [weights_df.index[0]]
        prev_weight = weights_df.iloc[0]

        for date, current_weight in weights_df.iloc[1:].iterrows():
            denom = prev_weight.replace(0, np.nan)
            denom = denom.where(denom >= 0.03, np.nan)
            ratio = ((prev_weight - current_weight).clip(lower=0) / denom).replace([np.inf, -np.inf], np.nan)
            avg_ratio = ratio.mean(skipna=True)

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



if __name__ == '__main__':
    run_backtest()