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
    """

    @staticmethod
    def read_daily_factors(start: str = None, end: str = None) -> pd.DataFrame:
        """
        读取每日因子收益数据（按日对齐，不做聚合）。
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

            # 需要读取的所有字段
            factor_fields = ["MKT", "SMB", "HML", "QMJ"]
            for field in factor_fields:
                fetch(field, field)

            if not dfs:
                logger.warning("未读取到任何因子数据")
                return pd.DataFrame()

            df_merged = pd.concat(dfs, axis=1).sort_index()
            df_merged.index.name = "date"

            logger.info("读取因子数据完成，共 %d 行，%d 列", *df_merged.shape)
            return df_merged.dropna(how="all")

