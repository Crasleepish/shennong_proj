# app/ml/preprocess.py

import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging

logger = logging.getLogger(__name__)

def select_features_vif_pca(df_X: pd.DataFrame, vif_thresh: float = 10.0, pca_var: float = 0.95) -> pd.DataFrame:
    """
    迭代剔除 VIF > 阈值的特征，然后执行 PCA 降维。
    :param df_X: 原始特征 DataFrame
    :param vif_thresh: VIF 阈值（如 >10 即剔除）
    :param pca_var: PCA 保留的解释方差比例（如 0.95）
    :return: 降维后的特征 DataFrame
    """
    df_X = df_X.dropna(axis=1).copy()
    df_X = df_X.loc[:, df_X.std() > 0]

    logger.info("开始迭代 VIF 剔除，共 %d 个特征", df_X.shape[1])

    # 迭代剔除 VIF 过高的变量
    while True:
        vif_df = pd.DataFrame()
        vif_df["feature"] = df_X.columns
        vif_df["VIF"] = [
            variance_inflation_factor(df_X.values, i)
            for i in range(df_X.shape[1])
        ]

        max_vif = vif_df["VIF"].max()
        if max_vif <= vif_thresh:
            break

        to_drop = vif_df.sort_values("VIF", ascending=False).iloc[0]["feature"]
        logger.info("剔除高 VIF 特征: %s (VIF=%.2f)", to_drop, max_vif)
        df_X = df_X.drop(columns=[to_drop])

    logger.info("VIF 筛选后剩余 %d 个特征", df_X.shape[1])

    return df_X