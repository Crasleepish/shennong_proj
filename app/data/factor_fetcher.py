from app.models.stock_models import MarketFactors
from app.dao.stock_info_dao import MarketFactorsDao
from app.backtest.portfolio_BM_S_L import backtest_strategy as portfolio_BM_S_L
from app.backtest.portfolio_BM_B_L import backtest_strategy as portfolio_BM_B_L
from app.backtest.portfolio_BM_S_M import backtest_strategy as portfolio_BM_S_M
from app.backtest.portfolio_BM_B_M import backtest_strategy as portfolio_BM_B_M
from app.backtest.portfolio_BM_S_H import backtest_strategy as portfolio_BM_S_H
from app.backtest.portfolio_BM_B_H import backtest_strategy as portfolio_BM_B_H
from app.backtest.portfolio_OP_S_L import backtest_strategy as portfolio_OP_S_L
from app.backtest.portfolio_OP_B_L import backtest_strategy as portfolio_OP_B_L
from app.backtest.portfolio_OP_S_M import backtest_strategy as portfolio_OP_S_M
from app.backtest.portfolio_OP_B_M import backtest_strategy as portfolio_OP_B_M
from app.backtest.portfolio_OP_S_H import backtest_strategy as portfolio_OP_S_H
from app.backtest.portfolio_OP_B_H import backtest_strategy as portfolio_OP_B_H
from app.backtest.portfolio_VLT_S_L import backtest_strategy as portfolio_VLT_S_L
from app.backtest.portfolio_VLT_B_L import backtest_strategy as portfolio_VLT_B_L
from app.backtest.portfolio_VLT_S_H import backtest_strategy as portfolio_VLT_S_H
from app.backtest.portfolio_VLT_B_H import backtest_strategy as portfolio_VLT_B_H
from app.backtest.portfolio_LIQ_X_H import backtest_strategy as portfolio_LIQ_X_H
from app.backtest.portfolio_LIQ_X_L import backtest_strategy as portfolio_LIQ_X_L
from app.backtest.portfolio_ALL import backtest_strategy as portfolio_ALL
from app.data.helper import get_index_daily_return
import os
from datetime import datetime
import pandas as pd

import logging

logger = logging.getLogger(__name__)

output_dir = r"./bt_result"

RET_DATE_COL = "date"
RET_VAL_COL = "group"

def format_date(date_str):
    """
    将日期字符串转换为 YYYYMMDD 格式的字符串。
    
    参数:
        date_str (str): 日期字符串，格式为 "YYYY-MM-DD"
        
    返回:
        str: YYYYMMDD 格式的日期字符串，如果解析失败则返回 "invaliddate"
    """
    try:
        # 尝试解析日期字符串（格式：YYYY-MM-DD）
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # 转换为 YYYYMMDD 格式
        return date_obj.strftime("%Y%m%d")
    except ValueError:
        # 如果解析失败，返回 "invaliddate"
        return "invaliddate"

def safe_value(val):
    return None if pd.isna(val) else val

def safe_get(val):
    return 0.0 if val is None or pd.isna(val) else val

class FactorFetcher:
    def __init__(self):
        self.market_factors_dao = MarketFactorsDao._instance

    def compute_and_store_daily_factors(self, end_date: str, base_dir: str):
        folder = os.path.join(base_dir, end_date.replace('-', ''))
        if not os.path.exists(folder):
            logger.warn("目录不存在: %s， 创建", folder)
            os.makedirs(folder)
            return

        logger.info("开始处理目录: %s", folder)

        # 收集所有组合数据
        returns_dict = {}
        for file in os.listdir(folder):
            if file.startswith("portfolio_") and file.endswith("_daily_returns.csv"):
                path = os.path.join(folder, file)
                df = pd.read_csv(path, usecols=[RET_DATE_COL, RET_VAL_COL], parse_dates=[RET_DATE_COL])
                df = df.rename(columns={RET_VAL_COL: file.replace(".csv", "")})
                returns_dict[file] = df.set_index(RET_DATE_COL)

        if not returns_dict:
            logger.warning("未找到组合收益率文件")
            return

        # 合并所有组合收益率数据
        merged_df = pd.concat(returns_dict.values(), axis=1, join='inner')
        merged_df.index.name = "date"

        # 计算每日因子
        def calc_smb(factor: str, merged_df: pd.DataFrame) -> pd.Series:
            small = [col for col in merged_df.columns if col.startswith(f"portfolio_{factor}_S_")]
            big = [col for col in merged_df.columns if col.startswith(f"portfolio_{factor}_B_")]
            return merged_df[small].mean(axis=1) - merged_df[big].mean(axis=1)

        high_bm = merged_df[[col for col in merged_df.columns if col.startswith("portfolio_BM_") and col.endswith("_H_daily_returns")]].mean(axis=1)
        low_bm = merged_df[[col for col in merged_df.columns if col.startswith("portfolio_BM_") and col.endswith("_L_daily_returns")]].mean(axis=1)

        high_op = merged_df[[col for col in merged_df.columns if col.startswith("portfolio_OP_") and col.endswith("_H_daily_returns")]].mean(axis=1)
        low_op = merged_df[[col for col in merged_df.columns if col.startswith("portfolio_OP_") and col.endswith("_L_daily_returns")]].mean(axis=1)

        high_vol = merged_df[[col for col in merged_df.columns if col.startswith("portfolio_VLT_") and col.endswith("_H_daily_returns")]].mean(axis=1)
        low_vol = merged_df[[col for col in merged_df.columns if col.startswith("portfolio_VLT_") and col.endswith("_L_daily_returns")]].mean(axis=1)

        high_liq = merged_df[[col for col in merged_df.columns if col.startswith("portfolio_LIQ_") and col.endswith("_H_daily_returns")]].mean(axis=1)
        low_liq = merged_df[[col for col in merged_df.columns if col.startswith("portfolio_LIQ_") and col.endswith("_L_daily_returns")]].mean(axis=1)

        factors_df = pd.DataFrame(index=merged_df.index)
        factors_df["MKT"] = merged_df["portofolio_ALL_daily_returns"]
        factors_df["SMB"] = calc_smb("BM", merged_df) #SMB仅相对HML保持纯净
        factors_df["HML"] = high_bm - low_bm
        factors_df["QMJ"] = high_op - low_op
        factors_df["VOL"] = low_vol - high_vol
        factors_df["LIQ"] = low_liq - high_liq

        logger.info("共计算出 %d 条因子记录", len(factors_df))

        # 入库
        for date, row in factors_df.iterrows():
            record = MarketFactors(
                date=date,
                MKT=safe_value(row["MKT"]),
                SMB=safe_value(row["SMB"]),
                HML=safe_value(row["HML"]),
                QMJ=safe_value(row["QMJ"]),
                VOL=safe_value(row["VOL"]),
                LIQ=safe_value(row["LIQ"]),
            )
            self.market_factors_dao.upsert_one(record)

        logger.info("入库完成: %d 条记录", len(factors_df))


    def fetch_all(self, start_date: str, end_date: str, progress_callback=None):
        logger.info("Starting fetching market index from %s to %s", start_date, end_date)
        portfolio_ALL(start_date, end_date)

        if progress_callback:
            progress_callback(5.5, 100)

        logger.info("Starting run BM_S_L strategy from %s to %s", start_date, end_date)
        portfolio_BM_S_L(start_date, end_date)

        if progress_callback:
            progress_callback(11.1, 100)

        logger.info("Starting run BM_B_L strategy from %s to %s", start_date, end_date)
        portfolio_BM_B_L(start_date, end_date)

        if progress_callback:
            progress_callback(16.6, 100)

        logger.info("Starting run BM_S_M strategy from %s to %s", start_date, end_date)
        portfolio_BM_S_M(start_date, end_date)

        if progress_callback:
            progress_callback(22.2, 100)

        logger.info("Starting run BM_B_M strategy from %s to %s", start_date, end_date)
        portfolio_BM_B_M(start_date, end_date)

        if progress_callback:
            progress_callback(27.7, 100)

        logger.info("Starting run BM_S_H strategy from %s to %s", start_date, end_date)
        portfolio_BM_S_H(start_date, end_date)

        if progress_callback:
            progress_callback(33.3, 100)

        logger.info("Starting run BM_B_H strategy from %s to %s", start_date, end_date)
        portfolio_BM_B_H(start_date, end_date)

        if progress_callback:
            progress_callback(38.8, 100)

        logger.info("Starting run OP_S_L strategy from %s to %s", start_date, end_date)
        portfolio_OP_S_L(start_date, end_date)

        if progress_callback:
            progress_callback(44.4, 100)

        logger.info("Starting run OP_B_L strategy from %s to %s", start_date, end_date)
        portfolio_OP_B_L(start_date, end_date)

        if progress_callback:
            progress_callback(50.0, 100)

        logger.info("Starting run OP_S_M strategy from %s to %s", start_date, end_date)
        portfolio_OP_S_M(start_date, end_date)

        if progress_callback:
            progress_callback(55.5, 100)

        logger.info("Starting run OP_B_M strategy from %s to %s", start_date, end_date)
        portfolio_OP_B_M(start_date, end_date)

        if progress_callback:
            progress_callback(61.1, 100)

        logger.info("Starting run OP_S_H strategy from %s to %s", start_date, end_date)
        portfolio_OP_S_H(start_date, end_date)

        if progress_callback:
            progress_callback(66.6, 100)

        logger.info("Starting run OP_B_H strategy from %s to %s", start_date, end_date)
        portfolio_OP_B_H(start_date, end_date)

        if progress_callback:
            progress_callback(72.2, 100)

        logger.info("Starting run VLT_S_L strategy from %s to %s", start_date, end_date)
        portfolio_VLT_S_L(start_date, end_date)

        if progress_callback:
            progress_callback(77.7, 100)

        logger.info("Starting run VLT_B_L strategy from %s to %s", start_date, end_date)
        portfolio_VLT_B_L(start_date, end_date)

        if progress_callback:
            progress_callback(83.3, 100)

        logger.info("Starting run VLT_S_H strategy from %s to %s", start_date, end_date)
        portfolio_VLT_S_H(start_date, end_date)

        if progress_callback:
            progress_callback(88.8, 100)

        logger.info("Starting run VLT_B_H strategy from %s to %s", start_date, end_date)
        portfolio_VLT_B_H(start_date, end_date)

        if progress_callback:
            progress_callback(94.4, 100)

        portfolio_LIQ_X_H(start_date, end_date)

        if progress_callback:
            progress_callback(95, 100)

        portfolio_LIQ_X_L(start_date, end_date)

        if progress_callback:
            progress_callback(99.0, 100)

        self.compute_and_store_daily_factors(end_date, output_dir)

        if progress_callback:
            progress_callback(100, 100)

factor_fetcher = FactorFetcher()