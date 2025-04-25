import pandas as pd
import numpy as np
import os
from typing import Dict

## DEPRECATED

#ME / W-FRI
sample_rate = "W-FRI"

# --------------------------
# 数据加载与预处理
# --------------------------
def load_and_resample_data(folder_path: str, index_file: str) -> Dict[str, pd.DataFrame]:
    """加载所有组合的日收益率CSV并转换为月收益率，同时加载指数数据"""
    resampled_data = {}
    
    # 1. 加载中证全指日收益率
    index_daily = pd.read_csv(
        index_file,
        parse_dates=["date"],
        index_col="date"
    )
    # 转换为周收益率（每周五截止，且至少1个交易日）
    index_resampled = (
        index_daily.assign(n_trading_days=1)  # 标记每日计数
        .resample("W-FRI")
        .apply(lambda x: (1 + x["daily_return"]).prod() - 1 if len(x) > 0 else np.nan)
        .to_frame("group")
        .dropna()
    )
    index_resampled = index_resampled.dropna()
    # 转换为月收益率（复利累积）
    resampled_data["INDEX"] = index_resampled
    
    # 2. 加载组合日收益率并转换
    for filename in file_list:
        if filename.endswith(".csv"):
            # 解析组合类型
            parts = filename.split("_")
            factor_type = parts[1]  # BM/OP/VLT
            size = parts[2]         # B/S
            level = parts[3]        # H/M/L
            
            # 读取日数据
            filepath = os.path.join(folder_path, filename)
            daily_df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
            daily_df = daily_df.rename(columns={daily_df.columns[0]: "group"})
            
            # 转换为月收益率
            resampled_df = (1 + daily_df).resample(sample_rate).prod() - 1
            
            # 存储到字典
            key = f"{factor_type}_{size}_{level}"
            resampled_data[key] = resampled_df
            
    return resampled_data

# --------------------------
# 因子计算逻辑（月度）
# --------------------------
def calculate_monthly_factors(resampled_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """计算五因子收益率（MKT/SMB/HML/QMJ/VOL）"""
    # 合并所有数据
    all_dfs = []
    for key, df in resampled_data.items():
        df = df.rename(columns={"group": key})
        all_dfs.append(df)
    merged_df = pd.concat(all_dfs, axis=1, join="inner")
    
    # 初始化结果DataFrame
    factors_df = pd.DataFrame(index=merged_df.index)
    
    # 1. 市场因子（MKT）：中证全指月收益
    factors_df["MKT"] = merged_df["INDEX"]  # 取第一列（daily_return）
    
    # 2. 市值因子（SMB）
    def calc_smb(factor: str) -> pd.Series:
        small = [col for col in merged_df.columns 
                if col.startswith(f"{factor}_S_")]
        big = [col for col in merged_df.columns 
              if col.startswith(f"{factor}_B_")]
        return merged_df[small].mean(axis=1) - merged_df[big].mean(axis=1)
    
    factors_df["SMB"] = (
        calc_smb("BM") + calc_smb("OP") + calc_smb("VLT")
    ) / 3
    
    # 3. 价值因子（HML）
    high_bm = merged_df[[col for col in merged_df.columns 
                       if "_H" in col and "BM_" in col]].mean(axis=1)
    low_bm = merged_df[[col for col in merged_df.columns 
                      if "_L" in col and "BM_" in col]].mean(axis=1)
    factors_df["HML"] = high_bm - low_bm
    
    # 4. 质量因子（QMJ）
    high_op = merged_df[[col for col in merged_df.columns 
                       if "_H" in col and "OP_" in col]].mean(axis=1)
    low_op = merged_df[[col for col in merged_df.columns 
                      if "_L" in col and "OP_" in col]].mean(axis=1)
    factors_df["QMJ"] = high_op - low_op
    
    # 5. 波动率因子（VOL）
    low_vol = merged_df[[col for col in merged_df.columns 
                       if "_L" in col and "VLT_" in col]].mean(axis=1)
    high_vol = merged_df[[col for col in merged_df.columns 
                        if "_H" in col and "VLT_" in col]].mean(axis=1)
    factors_df["VOL"] = low_vol - high_vol
    
    return factors_df.dropna()

# --------------------------
# 主函数
# --------------------------
def main(portfolio_folder: str, index_file: str, output_path: str):
    # 加载并转换数据
    resampled_data = load_and_resample_data(portfolio_folder, index_file)
    
    # 计算因子
    factors_df = calculate_monthly_factors(resampled_data)
    
    # 输出结果
    factors_df.to_csv(output_path)
    print(f"因子数据已保存至：{output_path}")

if __name__ == "__main__":
    file_list = ["portfolio_BM_B_H_daily_returns.csv",
        "portfolio_BM_S_H_daily_returns.csv",
        "portfolio_BM_S_M_daily_returns.csv",
        "portfolio_BM_B_M_daily_returns.csv",
        "portfolio_BM_B_L_daily_returns.csv",
        "portfolio_BM_S_L_daily_returns.csv",
        "portfolio_OP_B_H_daily_returns.csv",
        "portfolio_OP_B_M_daily_returns.csv",
        "portfolio_OP_S_M_daily_returns.csv",
        "portfolio_OP_B_L_daily_returns.csv",
        "portfolio_OP_S_L_daily_returns.csv",
        "portfolio_OP_S_H_daily_returns.csv",
        "portfolio_VLT_B_H_daily_returns.csv",
        "portfolio_VLT_S_H_daily_returns.csv",
        "portfolio_VLT_B_L_daily_returns.csv",
        "portfolio_VLT_S_L_daily_returns.csv"]
    main(
        portfolio_folder="./result", 
        index_file="./result/csi_index_zzqz.csv", 
        output_path=f"./output/factors{sample_rate}.csv"
    )