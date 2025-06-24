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

    # Step 4: 读取历史因子收益率（用于特征构造）
    historical_factors = load_historical_factor_returns()

    # Step 5: 拼接特征，执行预测（模型推理）
    predicted_returns = predict_future_returns(realtime_factors, historical_factors)

    # Step 6: 执行组合优化，输出最优资产权重
    optimal_weights = optimize_allocation(predicted_returns)

    # 输出或保存优化结果
    output_optimized_portfolio(optimal_weights)


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
    import os
    import pandas as pd
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

def load_historical_factor_returns():
    """读取历史因子收益率数据，用于补充模型预测特征"""
    # TODO: 从数据库中读取因子历史收益率（或净值），格式与因子预测模型输入一致
    pass

def predict_future_returns(realtime_factors, historical_factors):
    """拼接历史与实时因子数据，调用模型预测未来因子收益"""
    # TODO: 拼接特征，加载模型，执行推理，输出未来收益预测值
    pass

def optimize_allocation(predicted_returns):
    """根据预测收益执行组合优化，输出最优权重"""
    # TODO: 构造 Black-Litterman 输入，CVaR 优化，得到 w*
    pass

def output_optimized_portfolio(weights):
    """保存或打印最终最优组合权重"""
    # TODO: 可写入数据库、缓存、控制台等
    pass
