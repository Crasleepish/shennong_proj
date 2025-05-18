import akshare as ak
import pandas as pd
import logging
from app.database import get_db
from app.data_fetcher.base_fetcher import BaseFetcher
from app.models.macro_models import (
    SocialFinancing,
    OfficialPmi,
    LprRate,
    CpiYearly,
    MoneySupply,
    FxReserves,
    GoldReserve,
    BondYield
)

logger = logging.getLogger(__name__)


class MacroDataFetcher:
    """
    负责从 Akshare 获取指定宏观指标，并统一入库。
    """

    @staticmethod
    def fetch_social_financing() -> pd.DataFrame:
        raw = ak.macro_china_shrzgm()
        df = BaseFetcher.rename_columns(raw, {
            "月份": "date",
            "社会融资规模增量": "total"
        })[["date", "total"]]

        df = BaseFetcher.standardize_dates(df, "date", format="%Y%m")
        BaseFetcher.log_shape(df, "社融")

        with get_db() as db:
            BaseFetcher.write_to_db(df, SocialFinancing, db)
        return df

    @staticmethod
    def fetch_pmi() -> pd.DataFrame:
        raw = ak.macro_china_pmi_yearly()
        df = BaseFetcher.rename_columns(raw, {
            "日期": "date",
            "今值": "value"
        })[["date", "value"]]

        df = BaseFetcher.standardize_dates(df, "date")
        BaseFetcher.log_shape(df, "官方PMI")

        with get_db() as db:
            BaseFetcher.write_to_db(df, OfficialPmi, db)
        return df

    @staticmethod
    def fetch_lpr_rate() -> pd.DataFrame:
        raw = ak.macro_china_lpr()
        df = BaseFetcher.rename_columns(raw, {
            "TRADE_DATE": "date",
            "LPR1Y": "lpr_1y"
        })[["date", "lpr_1y"]]

        df = BaseFetcher.standardize_dates(df, "date")
        BaseFetcher.log_shape(df, "LPR利率")

        with get_db() as db:
            BaseFetcher.write_to_db(df, LprRate, db)
        return df

    @staticmethod
    def fetch_cpi_yearly() -> pd.DataFrame:
        raw = ak.macro_china_cpi_yearly()
        df = BaseFetcher.rename_columns(raw, {
            "日期": "date",
            "今值": "cpi"
        })[["date", "cpi"]]

        df = BaseFetcher.standardize_dates(df, "date")
        BaseFetcher.log_shape(df, "CPI年率")

        with get_db() as db:
            BaseFetcher.write_to_db(df, CpiYearly, db)
        return df

    @staticmethod
    def fetch_money_supply() -> pd.DataFrame:
        raw = ak.macro_china_money_supply()
        df = BaseFetcher.rename_columns(raw, {
            "月份": "date",
            "货币(M1)-数量(亿元)": "m1",
            "货币(M1)-同比增长": "m1_yoy",
            "货币(M1)-环比增长": "m1_mom",
            "货币和准货币(M2)-数量(亿元)": "m2",
            "货币和准货币(M2)-同比增长": "m2_yoy",
            "货币和准货币(M2)-环比增长": "m2_mom"
        })[["date", "m1", "m1_yoy", "m1_mom", "m2", "m2_yoy", "m2_mom"]]

        df["date"] = df["date"].str.replace("年", "-").str.replace("月份", "")
        df = BaseFetcher.standardize_dates(df, "date")
        BaseFetcher.log_shape(df, "货币供应量")

        with get_db() as db:
            BaseFetcher.write_to_db(df, MoneySupply, db)
        return df

    @staticmethod
    def fetch_fx_reserves() -> pd.DataFrame:
        raw = ak.macro_china_fx_reserves_yearly()
        df = BaseFetcher.rename_columns(raw, {
            "日期": "date",
            "今值": "reserve"
        })[["date", "reserve"]]

        df = BaseFetcher.standardize_dates(df, "date")
        BaseFetcher.log_shape(df, "外汇储备")

        with get_db() as db:
            BaseFetcher.write_to_db(df, FxReserves, db)
        return df

    @staticmethod
    def fetch_gold() -> pd.DataFrame:
        raw = ak.macro_china_fx_gold()
        df = BaseFetcher.rename_columns(raw, {
            "月份": "date",
            "黄金储备-数值": "gold_reserve",
            "黄金储备-同比": "gold_yoy",
            "黄金储备-环比": "gold_mom"
        })[["date", "gold_reserve", "gold_yoy", "gold_mom"]]

        df["date"] = df["date"].str.replace("年", "-").str.replace("月份", "")
        df = BaseFetcher.standardize_dates(df, "date")
        BaseFetcher.log_shape(df, "黄金储备")

        with get_db() as db:
            BaseFetcher.write_to_db(df, GoldReserve, db)
        return df
    
    @staticmethod
    def fetch_bond_yield() -> pd.DataFrame:
        raw = ak.bond_zh_us_rate(start_date="19901219")
        df = BaseFetcher.rename_columns(raw, {
            "日期": "date",
            "中国国债收益率2年": "cn_2y",
            "中国国债收益率5年": "cn_5y",
            "中国国债收益率10年": "cn_10y",
            "中国国债收益率30年": "cn_30y",
            "中国国债收益率10年-2年": "cn_10y_2y"
        })[["date", "cn_2y", "cn_5y", "cn_10y", "cn_30y", "cn_10y_2y"]]

        df = BaseFetcher.standardize_dates(df, "date")
        BaseFetcher.log_shape(df, "中国国债收益率")

        with get_db() as db:
            BaseFetcher.write_to_db(df, BondYield, db)
        return df

    @staticmethod
    def fetch_all() -> None:
        """批量调用所有 fetch_* 方法"""
        MacroDataFetcher.fetch_social_financing()
        MacroDataFetcher.fetch_pmi()
        MacroDataFetcher.fetch_lpr_rate()
        MacroDataFetcher.fetch_cpi_yearly()
        MacroDataFetcher.fetch_money_supply()
        MacroDataFetcher.fetch_fx_reserves()
        MacroDataFetcher.fetch_gold()
        MacroDataFetcher.fetch_bond_yield()

        logger.info("所有宏观数据抓取与入库完成 ✅")
