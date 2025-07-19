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

app = create_app()
logger = logging.getLogger(__name__)
factor_data_reader = FactorDataReader()
csi_index_data_fetcher = CSIIndexDataFetcher()
fee_rate = 0.00025
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
        
def load_fund_betas(codes):
    df = pd.read_csv("output/fund_factors.csv")
    df = df[df["code"].isin(codes)].reset_index(drop=True)
    return df[["code", "MKT", "SMB", "HML", "QMJ"]]


def build_price_df(asset_source_map: dict, start: str, end: str) -> pd.DataFrame:
    """
    构造组合资产的净值曲线：factor资产使用因子暴露生成，index资产使用真实指数行情
    """
    df_factors = factor_data_reader.read_daily_factors(start=start, end=end)[["MKT", "SMB", "HML", "QMJ"]].dropna()

    net_value_df = pd.DataFrame(index=df_factors.index)
    dao = FundHistDao._instance
    for code, src in asset_source_map.items():
        if src == "factor":
            beta_df = load_fund_betas([code]).set_index("code")
            if code not in beta_df.index:
                logger.warning(f"⚠️ {code} 因子暴露缺失，跳过")
                continue
            beta = beta_df.loc[code].values
            ret = df_factors.values @ beta
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


def run_backtest(start="2025-07-15", end="2025-07-17", window=20):
    out_dir = f"./fund_portfolio_bt_result/{datetime.today().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    with app.app_context(), get_db() as db:
        print("🔁 开始构建价格数据")
        price_df = build_price_df(asset_source_map, start, end)
        all_dates = price_df.index
        assets = list(asset_source_map.keys())
        weights_df = pd.DataFrame(index=all_dates, columns=assets)

        prev_weights = None  # 上一日的平滑权重

        print("📊 开始每日 optimize + ewma 平滑 + 入库")
        for dt in tqdm(all_dates):
            try:
                w_today = optimize(
                    asset_source_map=asset_source_map,
                    code_factors_map=code_factors_map,
                    trade_date=dt.strftime('%Y-%m-%d'),
                    window=window,
                    view_codes=view_codes
                )["weights"]

                # 如果第一天，从数据库尝试读取前一日平滑值
                if prev_weights is None:
                    trade_dates = TradeCalendarReader.get_trade_dates(end=dt.strftime("%Y-%m-%d"))
                    if len(trade_dates) >= 2:
                        prev_trade_date = trade_dates[-2]
                    else:
                        raise ValueError("交易日不足，无法执行权重平滑")
                    prev_row = db.query(PortfolioWeights).filter_by(
                        portfolio_id=portfolio_id,
                        date=prev_trade_date
                    ).first()
                    prev_weights = json.loads(prev_row.weights_ewma) if prev_row else w_today

                all_codes = set(w_today.keys()).union(prev_weights.keys())
                w_ewma = {
                    code: round(alpha * w_today.get(code, 0.0) + (1 - alpha) * prev_weights.get(code, 0.0), 8)
                    for code in all_codes
                }

                weights_df.loc[dt] = pd.Series(w_ewma)
                prev_weights = w_ewma.copy()

                # 入库
                pw = PortfolioWeights(
                    portfolio_id=portfolio_id,
                    date=pd.Timestamp(dt),
                    weights=json.dumps(w_today),
                    weights_ewma=json.dumps(w_ewma)
                )
                db.merge(pw)

            except Exception as e:
                logger.warning(f"⚠️ {dt.strftime('%Y-%m-%d')} 调仓失败: {e}")
                continue
        db.commit()  # ✅ 提交所有权重记录

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

        init_cash = 10_000_000

        # 1. 构造 segment_mask
        segment_mask = np.full((price_df.shape[0], 1), False, dtype=bool)
        for date in weights_df.index:
            segment_mask[price_df.index.get_loc(date), 0] = True

        # 2. 计算 order_sizes（目标持仓变动量）
        order_sizes = pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)
        current_positions = pd.Series(0.0, index=price_df.columns)
        current_cash = init_cash

        for dt in weights_df.index:
            target_weight = weights_df.loc[dt]
            prices = price_df.loc[dt]

            # 计算总资产
            asset_value = (current_positions * prices).sum()
            total_value = asset_value + current_cash

            # 金额 -> 目标股数
            target_position = np.floor((target_weight * total_value) / prices)
            target_position = target_position.fillna(0)

            diff_position = target_position - current_positions
            diff_value = diff_position * prices

            # 考虑手续费和滑点
            sell_cash = -diff_value[diff_value < 0].sum() * (1 - fee_rate - slippage_rate)
            buy_cash_needed = diff_value[diff_value > 0].sum() * (1 + fee_rate + slippage_rate)

            while sell_cash + current_cash < buy_cash_needed:
                scale = (sell_cash + current_cash) / buy_cash_needed
                target_position = np.floor(target_position * scale)
                diff_position = target_position - current_positions
                diff_value = diff_position * prices
                sell_cash = -diff_value[diff_value < 0].sum() * (1 - fee_rate - slippage_rate)
                buy_cash_needed = diff_value[diff_value > 0].sum() * (1 + fee_rate + slippage_rate)

            order_sizes.loc[dt] = diff_position
            current_positions = target_position
            current_cash = current_cash + sell_cash - buy_cash_needed

        # 3. 自定义 order_func
        @njit(cache=True)
        def pre_group_func_nb(c):
            order_value_out = np.empty(c.group_len, dtype=np.float64)
            return (order_value_out,)

        @njit(cache=True)
        def pre_segment_func_nb(c, order_value_out, size, price, size_type, direction):
            for col in range(c.from_col, c.to_col):
                c.last_val_price[col] = nb.get_col_elem_nb(c, col, price)
            nb.sort_call_seq_nb(c, size, size_type, direction, order_value_out)
            return ()

        @njit(cache=True)
        def order_func_nb(c, size, price, size_type, direction, fees, slippage):
            if nb.get_elem_nb(c, size) == 0:
                return nb.NoOrder
            print(">>>generate order: idx=", c.i, 
              ", col=", c.col, 
              ", size=", nb.get_elem_nb(c, size),
              ", price=", nb.get_elem_nb(c, price), 
              ", size_type=", nb.get_elem_nb(c, size_type),
              ", direction=", nb.get_elem_nb(c, direction),
              ", fees=", nb.get_elem_nb(c, fees),
              ", slippage=", nb.get_elem_nb(c, slippage))
            return nb.order_nb(
                size=nb.get_elem_nb(c, size),
                price=nb.get_elem_nb(c, price),
                size_type=nb.get_elem_nb(c, size_type),
                direction=nb.get_elem_nb(c, direction),
                fees=nb.get_elem_nb(c, fees),
                slippage=nb.get_elem_nb(c, slippage)
            )
        # 4. 构造 Portfolio
        pf = vbt.Portfolio.from_order_func(
            price_df,
            order_func_nb,
            vbt.Rep('size'),
            vbt.Rep('price'),
            vbt.Rep('size_type'),
            vbt.Rep('direction'),
            vbt.Rep('fees'),
            vbt.Rep('slippage'),
            segment_mask=segment_mask,
            pre_group_func_nb=pre_group_func_nb,
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(vbt.Rep('size'), vbt.Rep('price'), vbt.Rep('size_type'), vbt.Rep('direction')),
            broadcast_named_args=dict(
                price=price_df,
                size=order_sizes,
                size_type=SizeType.Amount,
                direction=Direction.Both,
                fees=fee_rate,
                slippage=slippage_rate
            ),
            cash_sharing=True,
            group_by=True,
            init_cash=init_cash
        )

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
        logger.info("Order detail:")
        logger.info(pf.orders.records_readable)
        # stats = pf.stats()
        turnover_rate = compute_turnover_rate(pf, n=1)
        mean_turnover = turnover_rate[1:].mean() #去掉建仓首日的换手率

        # print(stats)
        print(f"\n🔄 日均换手率: {mean_turnover:.4f}")

        pf.value().to_csv(os.path.join(out_dir, "portfolio_value.csv"))
        pf.returns().to_csv(os.path.join(out_dir, "daily_returns.csv"))
        # stats.to_csv(os.path.join(out_dir, "stats.csv"))
        turnover_rate.to_csv(os.path.join(out_dir, "daily_turnover.csv"))
        pf.value(group_by=True).vbt.plot(title="混合资产经债线").write_html(os.path.join(out_dir, "value_plot.html"))

        print(f"✅ 回测完成，结果已保存至：{out_dir}")



if __name__ == '__main__':
    run_backtest()