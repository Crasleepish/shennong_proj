import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys
import os
import akshare as ak

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.dao.fund_info_dao import FundInfoDao
from app.data.helper import get_fund_daily_return, get_index_daily_return
from app import create_app

app = create_app()

# 假设输入数据格式：
# df_rf: DataFrame, 包含两列 ['date', 'daily_return']（无风险利率）
# df_list: 包含N个DataFrame的列表，每个DataFrame两列 ['date', 'daily_return']（资产收益率）

def max_sharpe_portfolio_with_return_constraint(df_list, df_rf, min_annual_return=0.08):
    """
    不允许做空且年化收益率≥min_annual_return的条件下，最大化夏普比率
    
    参数:
        df_list: 包含N个资产历史收益率的DataFrame列表
        df_rf: 无风险利率的DataFrame
        min_annual_return: 最低要求的年化收益率（默认8%）
    
    返回:
        dict: 最优权重、年化收益率、年化波动率、夏普比率
    """
    # 检查输入
    if len(df_list) < 2:
        raise ValueError("至少需要2项资产")
    
    # 1. 合并数据（对齐日期）
    df_merged = df_rf.rename(columns={'daily_return': 'rf'})
    for i, df in enumerate(df_list):
        df_merged = df_merged.merge(
            df.rename(columns={'daily_return': f'asset_{i}'}),
            on='date', how='inner'
        )
    
    # 2. 提取收益率数据（去掉日期和无风险利率列）
    df_merged = df_merged.ffill().dropna()
    returns = df_merged.drop(columns=['date', 'rf']).values
    rf_daily = df_merged['rf'].mean()  # 平均无风险利率
    
    # 3. 计算预期收益率和协方差矩阵
    mu = np.mean(returns, axis=0)  # 各资产平均收益率
    cov = np.cov(returns, rowvar=False)  # 协方差矩阵
    
    # 4. 定义优化问题（最大化夏普比率 = 最小化负夏普比率）
    def negative_sharpe_ratio(weights):
        port_return = np.dot(weights, mu)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return -(port_return - rf_daily) / port_vol  # 负号因为最小化
    
    n_assets = len(mu)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
        {'type': 'ineq', 'fun': lambda w: 252 * np.dot(w, mu) - min_annual_return}  # 年化收益率≥8%
    ]
    bounds = tuple((0, 1) for _ in range(n_assets))  # 不允许做空
    
    # 初始猜测（等权重）
    init_weights = np.ones(n_assets) / n_assets
    
    # 优化
    result = minimize(
        negative_sharpe_ratio,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    if not result.success:
        raise ValueError("优化失败: " + result.message)
    
    optimal_weights = result.x
    
    # 计算组合性能（年化）
    annual_return = 252 * np.dot(optimal_weights, mu)
    annual_vol = np.sqrt(252) * np.sqrt(np.dot(optimal_weights.T, np.dot(cov, optimal_weights)))
    sharpe = (annual_return - 252 * rf_daily) / annual_vol
    
    return {
        'weights': optimal_weights,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe
    }

def prepare_date(code_list: list, start_date: str, end_date: str):
    shibor_data = ak.rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="1月")
    rf_daily = (
        shibor_data[["报告日", "利率"]]
        .rename(columns={"报告日": "date", "利率": "rf_rate"})
        .assign(date=lambda x: pd.to_datetime(x["date"]),
                daily_return=lambda x: x["rf_rate"] / 100 / 252)  # 年化利率转日利率
        .reset_index(drop=True)
        .sort_values("date")[["date", "daily_return"]]
    )
    df_list = []
    for fund_code in code_list:
        fund = get_fund_daily_return(fund_code)
        if not fund.empty:
            fund['daily_return'] = fund['change_percent']
            fund = fund.loc[start_date:end_date]
            fund = fund[["date", "daily_return"]]
            fund = fund.reset_index(drop=True)
            df_list.append(fund)
    return df_list, rf_daily

def prepare_index_date(index_list: list, start_date: str, end_date: str):
    index_df_list = []
    for index_code in index_list:
        index_daily_return = get_index_daily_return(index_code)
        if not index_daily_return.empty:
            index_daily_return = index_daily_return.loc[start_date:end_date]
            index_df_list.append(index_daily_return)
    return index_df_list


# ============= 示例调用 =============
if __name__ == '__main__':
    with app.app_context():
        # code_list = ["000218", "003376", "005561", "240016"]
        code_list = ["000218", "017838", "012708", "019408", "019918", "019162", "240016", "007937"]
        df_list, df_rf = prepare_date(code_list, start_date="2022-06-01", end_date="2024-12-31")
        index_list = [] #"000919", "399631"
        index_df_list = prepare_index_date(index_list, start_date="2019-06-01", end_date="2023-06-01")
        df_list = df_list + index_df_list
        
        # 调用函数
        result = max_sharpe_portfolio_with_return_constraint(df_list=df_list, df_rf=df_rf, min_annual_return=0.08)
        
        # 打印结果
        print("最优权重:", result['weights'])
        print("年化收益率:", result['annual_return'])
        print("年化波动率:", result['annual_volatility'])
        print("夏普比率:", result['sharpe_ratio'])