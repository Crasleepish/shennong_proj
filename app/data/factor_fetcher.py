import os
import pandas as pd
from datetime import datetime
from app.data_fetcher import CSIIndexDataFetcher
from app.backtest.portfolio_driver import build_all_portfolios
from app.dao.stock_info_dao import MarketFactorsDao
from app.models.stock_models import MarketFactors

import logging

logger = logging.getLogger(__name__)

output_dir = r"./bt_result"

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

    def compute_and_store_daily_factors(self, start_date: str, end_date: str, output_dir: str):
        logger.info("开始读取回测组合数据目录: %s", output_dir)

        returns_dict = {}

        # 加载 bm 组合
        for size in ["S", "B"]:
            for bm in ["L", "M", "H"]:
                file = f"bm_{size}{bm}_daily_returns.csv"
                path = os.path.join(output_dir, file)
                if not os.path.exists(path):
                    logger.warning("未找到文件: %s", file)
                    continue
                df = pd.read_csv(path, parse_dates=["date"])[["date", "value"]]
                col_name = f"bm_{size}{bm}"
                df = df.rename(columns={"value": col_name}).set_index("date")
                returns_dict[col_name] = df

        # 加载 qmj 组合
        for size in ["S", "B"]:
            for q in ["L", "M", "H"]:
                file = f"qmj_{size}{q}_daily_returns.csv"
                path = os.path.join(output_dir, file)
                if not os.path.exists(path):
                    logger.warning("未找到文件: %s", file)
                    continue
                df = pd.read_csv(path, parse_dates=["date"])[["date", "value"]]
                col_name = f"qmj_{size}{q}"
                df = df.rename(columns={"value": col_name}).set_index("date")
                returns_dict[col_name] = df

        if not returns_dict:
            logger.warning("未加载到任何组合数据")
            return

        # 合并组合收益率
        merged_df = pd.concat(returns_dict.values(), axis=1).dropna()
        merged_df = merged_df.sort_index()
        merged_df.index.name = "date"

        # 加载中证全指收益率作为 MKT 因子
        index_df = CSIIndexDataFetcher().get_data_by_code_and_date(
            "000985.CSI", start=start_date, end=end_date, fields=["date", "change_percent"]
        )
        index_df = index_df.rename(columns={"change_percent": "MKT"})
        index_df = index_df.set_index("date", drop=True)
        index_df["MKT"] = index_df["MKT"] / 100.0
        merged_df = merged_df.join(index_df, how="inner")

        # SMB = 小盘 - 大盘（平均）
        smb_bm = (
            merged_df[["bm_SL", "bm_SM", "bm_SH"]].mean(axis=1) -
            merged_df[["bm_BL", "bm_BM", "bm_BH"]].mean(axis=1)
        )
        smb_qmj = (
            merged_df[["qmj_SL", "qmj_SM", "qmj_SH"]].mean(axis=1) -
            merged_df[["qmj_BL", "qmj_BM", "qmj_BH"]].mean(axis=1)
        )
        smb = (smb_bm + smb_qmj) / 2

        # HML = 高 BM - 低 BM（平均小盘和大盘）
        hml = (
            (merged_df["bm_SH"] + merged_df["bm_BH"]) / 2 -
            (merged_df["bm_SL"] + merged_df["bm_BL"]) / 2
        )

        # QMJ = 高质量 - 低质量（平均小盘和大盘）
        qmj = (
            (merged_df["qmj_SH"] + merged_df["qmj_BH"]) / 2 -
            (merged_df["qmj_SL"] + merged_df["qmj_BL"]) / 2
        )

        factors_df = pd.DataFrame({
            "MKT": merged_df["MKT"],
            "SMB": smb,
            "HML": hml,
            "QMJ": qmj
        }, index=merged_df.index)

        logger.info("因子计算完成，共 %d 条记录", len(factors_df))

        # 写入数据库
        for date, row in factors_df.iterrows():
            record = MarketFactors(
                date=date,
                MKT=safe_value(row["MKT"]),
                SMB=safe_value(row["SMB"]),
                HML=safe_value(row["HML"]),
                QMJ=safe_value(row["QMJ"]),
                VOL=None,  # 已弃用
                LIQ=None   # 已弃用
            )
            self.market_factors_dao.upsert_one(record)

        logger.info("因子入库完成: %d 条记录", len(factors_df))


    def fetch_all(self, start_date: str, end_date: str, progress_callback=None):
        """
        Parameters:
        - start_date (str): The start date of the data to fetch, in 'YYYY-MM-DD' format.
        - end_date (str): The end date of the data to fetch, in 'YYYY-MM-DD' format.
        - progress_callback (callable, optional): A callback function to report the progress of the fetch operation.
                                    The callback function should accept two float arguments: the current progress
                                    and the total progress. Defaults to None.
        """
        logger.info("Starting fetching market index from %s to %s", start_date, end_date)
        build_all_portfolios(start_date, end_date)

        self.compute_and_store_daily_factors(start_date=start_date, end_date=end_date, output_dir=output_dir)

        if progress_callback:
            progress_callback(100, 100)

factor_fetcher = FactorFetcher()