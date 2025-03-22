import pandas as pd
import numpy as np

# 1. 定义文件路径和组合名称
file_map = {
    "BM_S_L": r"../result/portofolio_BM_S_L_daily_returns.csv",
    "BM_B_L": r"../result/portofolio_BM_B_L_daily_returns.csv",
    "BM_S_M": r"../result/portofolio_BM_S_M_daily_returns.csv",
    "BM_B_M": r"../result/portofolio_BM_B_M_daily_returns.csv",
    "BM_S_H": r"../result/portofolio_BM_S_H_daily_returns.csv",
    "BM_B_H": r"../result/portofolio_BM_B_H_daily_returns.csv",
}

# 2. 读取所有CSV文件并合并
dfs = []
for group, path in file_map.items():
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"group": group})
    df = df.set_index("date")
    dfs.append(df)

merged_df = pd.concat(dfs, axis=1, join="inner")  # 确保日期对齐

# 3. 将每日回报转换为月度回报（复利累积）
monthly_returns = merged_df.resample("M").apply(lambda x: (1 + x).prod() - 1)

# 4. 计算SMB_BM（按BM分组）
# 低BM分组：S_L - B_L
smb_bm_l = monthly_returns["BM_S_L"] - monthly_returns["BM_B_L"]
# 中BM分组：S_M - B_M
smb_bm_m = monthly_returns["BM_S_M"] - monthly_returns["BM_B_M"]
# 高BM分组：S_H - B_H
smb_bm_h = monthly_returns["BM_S_H"] - monthly_returns["BM_B_H"]

# SMB_BM为三组平均值
smb_bm = (smb_bm_l + smb_bm_m + smb_bm_h) / 3
smb_bm = smb_bm.rename("SMB_BM")

# 5. 输出到CSV
smb_bm.to_csv("SMB_BM_monthly.csv", header=True)
print("月度SMB_BM数据已保存至 SMB_BM_monthly.csv")