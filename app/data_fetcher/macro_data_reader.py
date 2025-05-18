import pandas as pd
from sqlalchemy.orm import Session
from app.database import get_db
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
from app.models.calendar_model import TradeCalendar
from app.data_fetcher.trade_calender_reader import TradeCalendarReader
import logging

logger = logging.getLogger(__name__)


class MacroDataReader:
    """
    从数据库中读取宏观经济数据，并整理为月度对齐的 DataFrame。
    """

    @staticmethod
    def read_all_macro_data(start: str = None, end: str = None) -> pd.DataFrame:
        """
        拉取所有宏观变量并合并为一个 DataFrame，并按月取中位数聚合。
        :param start: 起始日期（YYYY-MM-DD）
        :param end: 截止日期（YYYY-MM-DD）
        :return: 合并后的 DataFrame（index 为月份，列为指标名）
        """
        with get_db() as db:
            dfs = []

            def fetch(model, db_col_name, col_name):
                query = db.query(model)
                if start:
                    query = query.filter(model.date >= pd.to_datetime(start))
                if end:
                    query = query.filter(model.date <= pd.to_datetime(end))

                df = pd.read_sql(query.statement, db.bind)
                if not df.empty:
                    df = df[["date", db_col_name]].rename(columns={db_col_name: col_name})
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date").sort_index()
                    dfs.append(df)

            # 实际调用 fetch 拉取各字段
            fetch(SocialFinancing, "total", "社融")
            fetch(OfficialPmi, "value", "PMI")
            fetch(LprRate,"lpr_1y", "LPR1Y")
            fetch(CpiYearly, "cpi", "CPI")
            fetch(MoneySupply, "m1", "M1")
            fetch(MoneySupply, "m1_yoy", "M1YOY")
            fetch(MoneySupply, "m2", "M2")
            fetch(MoneySupply, "m2_yoy", "M2YOY")
            fetch(FxReserves,"reserve", "外储")
            fetch(GoldReserve, "gold_reserve", "黄金")
            fetch(BondYield, "cn_2y", "BOND_2Y")
            fetch(BondYield, "cn_10y", "BOND_10Y")

        # 合并所有 DataFrame（自然日索引）
        df_merged = pd.concat(dfs, axis=1)
        df_merged = df_merged.sort_index()

        # 构建自然日历索引，先用自然日填充
        natural_index = pd.date_range(
            start=df_merged.index.min() if not start else pd.to_datetime(start),
            end=df_merged.index.max() if not end else pd.to_datetime(end),
            freq="D"
        )
        df_daily = df_merged.reindex(natural_index).ffill()

        # 聚合为月度数据：每月的中位数
        df_monthly = df_daily.resample("M").median()
        df_monthly.index.name = "date"

        logger.info("读取并按月聚合宏观数据完成，共 %d 月，%d 列", *df_monthly.shape)
        return df_monthly