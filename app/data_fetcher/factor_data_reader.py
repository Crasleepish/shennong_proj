import pandas as pd
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.stock_models import MarketFactors
import logging

logger = logging.getLogger(__name__)


class FactorDataReader:
    """
    从数据库中读取每日因子收益数据（如 MKT、SMB、HML、QMJ 等）。
    输出为按日对齐的 DataFrame，index 为日期。
    支持拼接额外提供的数据（如盘中实时因子）。
    """

    def __init__(self, additional_df: pd.DataFrame = None):
        """
        初始化 FactorDataReader

        :param additional_df: 额外添加的数据（DataFrame），index 为日期，列为因子名
        """
        self.additional_df = additional_df

    def read_daily_factors(self, start: str = None, end: str = None) -> pd.DataFrame:
        """
        读取每日因子收益数据（按日对齐，不做聚合），并附加额外数据（如盘中值）
        :param start: 起始日期（YYYY-MM-DD）
        :param end: 截止日期（YYYY-MM-DD）
        :return: DataFrame，index 为日期，列为因子名
        """
        with get_db() as db:
            dfs = []

            def fetch(col_name: str, alias: str):
                query = db.query(MarketFactors.date, getattr(MarketFactors, col_name).label(alias))
                if start:
                    query = query.filter(MarketFactors.date >= pd.to_datetime(start))
                if end:
                    query = query.filter(MarketFactors.date <= pd.to_datetime(end))
                df = pd.read_sql(query.statement, db.bind)
                if not df.empty:
                    df = df.set_index("date").sort_index()
                    dfs.append(df)

            factor_fields = ["MKT", "SMB", "HML", "QMJ"]
            for field in factor_fields:
                fetch(field, field)

            if not dfs:
                logger.warning("未读取到任何因子数据")
                df_merged = pd.DataFrame()
            else:
                df_merged = pd.concat(dfs, axis=1).sort_index()

            if self.additional_df is not None and not self.additional_df.empty:
                df_merged = pd.concat([df_merged, self.additional_df], axis=0)

            df_merged = df_merged.sort_index()
            df_merged.index.name = "date"
            return df_merged.dropna(how="all")

    def read_factor_nav(self, start: str = None, end: str = None) -> pd.DataFrame:
        """
        返回以下字段：
        - MKT_NAV：MKT 的累计净值曲线
        - SMB_NAV：SMB 的累计净值曲线
        - HML_NAV：HML 的累计净值曲线
        - QMJ_NAV：QMJ 的累计净值曲线
        """
        df_factors = self.read_daily_factors(start=start, end=end)
        if df_factors.empty:
            return pd.DataFrame()

        df_nav = (1 + df_factors).cumprod()
        return df_nav.rename(columns={
            "MKT": "MKT_NAV",
            "SMB": "SMB_NAV",
            "HML": "HML_NAV",
            "QMJ": "QMJ_NAV"
        }).dropna()