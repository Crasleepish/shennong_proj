# app/scripts/plot_fund_fit.py
# -*- coding: utf-8 -*-

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 依赖你项目里已存在的方法/DAO （与 beta_estimator.py 一致）===
from app.data.helper import get_fund_daily_return
from app.dao.stock_info_dao import MarketFactorsDao
from app.dao.betas_dao import FundBetaDao
from app import create_app

FACTOR_COLS = ["MKT", "SMB", "HML", "QMJ"]  # 四因子
BETA_COLS = ["MKT_beta", "SMB_beta", "HML_beta", "QMJ_beta", "const"]        # β 向量含截距 α
app = create_app()

def load_data(fund_code: str, start: str | None, end: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    读取基金日收益、因子日收益、基金动态beta（含alpha）。
    返回:
      fund_ret_df: ['date','daily_return']
      factor_df:   ['date','MKT','SMB','HML','QMJ']
      beta_df:     ['date','MKT','SMB','HML','QMJ','const']  # 每日Kalman估计
    """
    fund_ret_df = get_fund_daily_return(fund_code, start, end)
    # 保障列名 & 类型
    fund_ret_df = fund_ret_df.reset_index() if 'date' not in fund_ret_df.columns else fund_ret_df
    fund_ret_df['date'] = pd.to_datetime(fund_ret_df['date'])
    fund_ret_df = fund_ret_df[['date', 'change_percent']].dropna()

    factor_df = MarketFactorsDao().select_dataframe_by_date(start, end)
    factor_df = factor_df.reset_index() if 'date' not in factor_df.columns else factor_df
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    factor_df = factor_df[['date'] + FACTOR_COLS].dropna()
    factor_df = factor_df.set_index("date", drop=False)

    beta_df = FundBetaDao.select_all_by_code_date(fund_code, start, end)

    beta_df = beta_df.reset_index() if 'date' not in beta_df.columns else beta_df
    beta_df['date'] = pd.to_datetime(beta_df['date'])
    beta_df = beta_df.set_index("date", drop=False)

    need_cols = ['date'] + FACTOR_COLS + ["const"]
    missing = [c for c in need_cols if c not in beta_df.columns]
    if missing:
        raise RuntimeError(f"FundBetaDao 返回缺少列: {missing}。请确认表/DAO输出。")
    beta_df = beta_df[need_cols].dropna()

    return fund_ret_df, factor_df, beta_df


def build_curves(fund_ret_df: pd.DataFrame, factor_df: pd.DataFrame, beta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    合并数据并计算两条累计净值曲线：
      1) 实际曲线：由 daily_return 累乘
      2) 拟合曲线：r_fit = β·factor + α，再累乘
    返回:
      merged: 包含 date / daily_return / 因子 / β / α / r_fit
      nav_real: 实际累计净值（起点归一到1.0）
      nav_fit:  拟合累计净值（起点归一到1.0）
    """
    # 先把因子与β合并（按同一日期）
    df = pd.merge(factor_df, beta_df, left_index=True, right_index=True, how='inner', suffixes=('', '_beta')).sort_index()
    # 再与基金真实日收益合并，取三者共有区间
    df = pd.merge(df, fund_ret_df, left_index=True, right_index=True, how='inner').sort_index()

    # 计算拟合日收益：r_fit = sum_i beta_i * factor_i + alpha
    # 注意 β 随时间变（动态），此处为逐日点乘
    beta_mat = df[[c for c in BETA_COLS]].to_numpy(dtype=float)  # [MKT,SMB,HML,QMJ,const]
    factor_mat = df[FACTOR_COLS].to_numpy(dtype=float)          # [MKT,SMB,HML,QMJ]

    r_fit = (beta_mat[:, :4] * factor_mat).sum(axis=1)
    df['r_fit'] = r_fit.astype(float)

    # 实际与拟合累计净值（归一到1.0）
    nav_real = (1.0 + df['change_percent'].astype(float)).cumprod()
    nav_fit = (1.0 + df['r_fit']).cumprod()

    # 起点统一归一（避免首日不是1.0）
    if len(nav_real) > 0:
        nav_real = nav_real / nav_real.iloc[0]
    if len(nav_fit) > 0:
        nav_fit = nav_fit / nav_fit.iloc[0]

    return df, nav_real, nav_fit


def plot_curves(df: pd.DataFrame, nav_real: pd.Series, nav_fit: pd.Series, fund_code: str, out: str | None):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, nav_real, label='真实累计净值（归一）', linewidth=1.8)
    plt.plot(df.index, nav_fit, label='拟合累计净值（β×因子+α）', linestyle='--', linewidth=1.8)
    plt.title(f'基金 {fund_code}：真实 vs 拟合 累计净值')
    plt.xlabel('日期')
    plt.ylabel('累计净值（起点=1.00）')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if out:
        plt.savefig(out, dpi=150)
        print(f"[OK] 已保存图像到 {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="绘制基金真实与拟合（β×因子+α）累计净值曲线")
    parser.add_argument("fund_code", type=str, help="基金代码")
    parser.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD（可选）")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD（可选）")
    parser.add_argument("--out", type=str, default=None, help="保存路径（如 plot.png），不传则窗口展示")
    args = parser.parse_args()

    fund_ret_df, factor_df, beta_df = load_data(args.fund_code, args.start, args.end)
    if fund_ret_df.empty:
        raise SystemExit("未获取到基金日收益数据。")
    if factor_df.empty:
        raise SystemExit("未获取到因子数据。")
    if beta_df.empty:
        raise SystemExit("未获取到基金动态β数据，请先回填/更新 Kalman β。")

    df, nav_real, nav_fit = build_curves(fund_ret_df, factor_df, beta_df)

    if df.empty:
        raise SystemExit("数据合并后为空，请检查日期区间与数据源。")

    plot_curves(df, nav_real, nav_fit, args.fund_code, args.out)


if __name__ == "__main__":
    main()
