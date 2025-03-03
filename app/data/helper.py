from app.dao.stock_info_dao import StockHistAdjDao
import pandas as pd

def get_prices_df() -> pd.DataFrame:
    """
    返回股票历史价格数据，DataFrame 的索引为交易日（datetime64[ns]），列为股票代码，
    值为收盘价（或者其它价格，根据需求）。
    """
    stock_hist_adj_dao = StockHistAdjDao._instance
    df_all = stock_hist_adj_dao.select_dataframe_all()
    return df_all.pivot(index="date", columns="stock_code", values="close")