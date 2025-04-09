import pandas as pd
import numpy as np
import os

# 1. 定义文件路径和组合名称（按市值 + 动量分组）
file_map = {
    "MOM_S_L": r"result/portfolio_MOM_S_L_daily_returns.csv",
    "MOM_B_L": r"result/portfolio_MOM_B_L_daily_returns.csv",
    "MOM_S_H": r"result/portfolio_MOM_S_H_daily_returns.csv",
    "MOM_B_H": r"result/portfolio_MOM_B_H_daily_returns.csv"
}

# 2. 读取所有CSV文件并合并
dfs = []
for group, path in file_map.items():
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"group": group})
    df = df.set_index("date")
    dfs.append(df)

merged_df = pd.concat(dfs, axis=1, join="inner")  # 按日期对齐

# 3. 将每日收益率转换为月度收益率（复利）
monthly_returns = merged_df.resample("M").apply(lambda x: (1 + x).prod() - 1)

# 4. 计算 MOM 因子
mom_s = monthly_returns["MOM_S_H"] - monthly_returns["MOM_S_L"]
mom_b = monthly_returns["MOM_B_H"] - monthly_returns["MOM_B_L"]

# 动量因子为小盘+大盘的平均
mom = ((mom_s + mom_b) / 2).rename("MOM")

# 5. 保存结果
mom.to_csv("MOM_monthly.csv", header=True)
print("月度 MOM 因子数据已保存至 MOM_monthly.csv")
