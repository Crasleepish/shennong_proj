# app/ml/dataset_builder.py

import pandas as pd
import numpy as np
from app.features.macro_feature_pipeline import MacroFeaturePipeline
from app.features.factor_feature_pipeline import FactorFeaturePipeline
from app.features.macro_feature_builder import MacroFeatureBuilder
from app.features.factor_feature_builder import FactorFeatureBuilder
from app.data_fetcher.macro_data_reader import MacroDataReader
from app.data_fetcher.factor_data_reader import FactorDataReader
from app.features.feature_assembler import FeatureAssembler
from app.ml.preprocess import select_features_vif_pca

class DatasetBuilder:
    def __init__(self):
        self.macro_feature_plan = {
            # "社融": [
            #     {"func": MacroFeatureBuilder.yoy_growth, "suffix": "yoy"},
            #     {"func": MacroFeatureBuilder.mom_growth, "suffix": "mom"},
            #     {"func": MacroFeatureBuilder.rolling_mean, "suffix": "avg3", "kwargs": {"window": 3}},
            # ],
            "PMI": [
                {"func": MacroFeatureBuilder.value_minus, "suffix": "diff_50", "kwargs": {"series2": pd.Series(50, index=pd.date_range("2000-01", "2100-01", freq="M"))}},
                {"func": MacroFeatureBuilder.rolling_slope, "suffix": "trend12", "kwargs": {"window": 12}},
            ],
            # "外储": [
            #     {"func": MacroFeatureBuilder.yoy_growth, "suffix": "yoy"},
            #     {"func": MacroFeatureBuilder.rolling_log_slope, "suffix": "trend3", "kwargs": {"window": 3}},
            # ],
            # "黄金": [
            #     {"func": MacroFeatureBuilder.yoy_growth, "suffix": "yoy"},
            #     {"func": MacroFeatureBuilder.rolling_log_slope, "suffix": "trend3", "kwargs": {"window": 3}},
            # ],
            "CPI": [
                {"func": MacroFeatureBuilder.raw, "suffix": "raw"},
                {"func": MacroFeatureBuilder.rolling_slope, "suffix": "trend12", "kwargs": {"window": 12}},
            ],
            ("BOND_10Y", "CPI"): [
                {"func": MacroFeatureBuilder.value_minus, "suffix": "real_rate"},
                {"func": MacroFeatureBuilder.diff_yoy, "suffix": "diff_yoy"},
            ],
            ("M1YOY", "M2YOY"): [
                {"func": MacroFeatureBuilder.value_minus, "suffix": "diff"},
            ],
            ("BOND_10Y", "BOND_2Y"): [
                {"func": MacroFeatureBuilder.value_minus, "suffix": "diff"},
                {"func": MacroFeatureBuilder.diff_yoy, "suffix": "diff_yoy"},
            ]
        }

        self.factor_feature_plan = {
            "MKT": [
                {"func": FactorFeatureBuilder.rolling_zscore, "suffix": "zscore", "kwargs": {"window": 2000}},
                {"func": FactorFeatureBuilder.mean_change, "suffix": "ma10_rate", "kwargs": {"period": 10}},
                {"func": FactorFeatureBuilder.mean_change, "suffix": "ma60_rate", "kwargs": {"period": 60}},
                {"func": FactorFeatureBuilder.volatility, "suffix": "vol20", "kwargs": {"window": 20}},
                {"func": FactorFeatureBuilder.rsi, "suffix": "rsi28", "kwargs": {"window": 28}},
                {"func": FactorFeatureBuilder.regression_slope_on_zscore, "suffix": "slope_z20", "kwargs": {"zscore_window": 20, "reg_window": 20}},
                {"func": FactorFeatureBuilder.regression_slope_diff_on_zscore, "suffix": "slope_diff_z20", "kwargs": {"zscore_window": 20, "reg_window": 20}},
                {"func": FactorFeatureBuilder.days_since_rsi_extreme, "suffix": "since_rsi_gt_70", "kwargs": {"window": 14, "threshold": 70, "mode": "gt"}},
                {"func": FactorFeatureBuilder.days_since_rsi_extreme, "suffix": "since_rsi_lt_30", "kwargs": {"window": 14, "threshold": 30, "mode": "lt"}},
                {"func": FactorFeatureBuilder.backward_cum_return, "suffix": "cum_ret_20", "kwargs": {"days": 20}},
                {"func": FactorFeatureBuilder.backward_cum_return, "suffix": "cum_ret_60", "kwargs": {"days": 60}},
            ],
            "SMB": [
                {"func": FactorFeatureBuilder.rolling_zscore, "suffix": "zscore", "kwargs": {"window": 2000}},
                {"func": FactorFeatureBuilder.mean_change, "suffix": "ma10_rate", "kwargs": {"period": 10}},
                {"func": FactorFeatureBuilder.mean_change, "suffix": "ma60_rate", "kwargs": {"period": 60}},
                {"func": FactorFeatureBuilder.volatility, "suffix": "vol20", "kwargs": {"window": 20}},
                {"func": FactorFeatureBuilder.rsi, "suffix": "rsi28", "kwargs": {"window": 28}},
                {"func": FactorFeatureBuilder.regression_slope_on_zscore, "suffix": "slope_z20", "kwargs": {"zscore_window": 20, "reg_window": 20}},
                {"func": FactorFeatureBuilder.regression_slope_diff_on_zscore, "suffix": "slope_diff_z20", "kwargs": {"zscore_window": 20, "reg_window": 20}},
                {"func": FactorFeatureBuilder.days_since_rsi_extreme, "suffix": "since_rsi_gt_70", "kwargs": {"window": 14, "threshold": 70, "mode": "gt"}},
                {"func": FactorFeatureBuilder.days_since_rsi_extreme, "suffix": "since_rsi_lt_30", "kwargs": {"window": 14, "threshold": 30, "mode": "lt"}},
                {"func": FactorFeatureBuilder.backward_cum_return, "suffix": "cum_ret_20", "kwargs": {"days": 20}},
                {"func": FactorFeatureBuilder.backward_cum_return, "suffix": "cum_ret_60", "kwargs": {"days": 60}},
            ],
            "HML": [
                {"func": FactorFeatureBuilder.rolling_zscore, "suffix": "zscore", "kwargs": {"window": 2000}},
                {"func": FactorFeatureBuilder.mean_change, "suffix": "ma10_rate", "kwargs": {"period": 10}},
                {"func": FactorFeatureBuilder.mean_change, "suffix": "ma60_rate", "kwargs": {"period": 60}},
                {"func": FactorFeatureBuilder.volatility, "suffix": "vol20", "kwargs": {"window": 20}},
                {"func": FactorFeatureBuilder.rsi, "suffix": "rsi28", "kwargs": {"window": 28}},
                {"func": FactorFeatureBuilder.regression_slope_on_zscore, "suffix": "slope_z20", "kwargs": {"zscore_window": 20, "reg_window": 20}},
                {"func": FactorFeatureBuilder.regression_slope_diff_on_zscore, "suffix": "slope_diff_z20", "kwargs": {"zscore_window": 20, "reg_window": 20}},
                {"func": FactorFeatureBuilder.days_since_rsi_extreme, "suffix": "since_rsi_gt_70", "kwargs": {"window": 14, "threshold": 70, "mode": "gt"}},
                {"func": FactorFeatureBuilder.days_since_rsi_extreme, "suffix": "since_rsi_lt_30", "kwargs": {"window": 14, "threshold": 30, "mode": "lt"}},
                {"func": FactorFeatureBuilder.backward_cum_return, "suffix": "cum_ret_20", "kwargs": {"days": 20}},
                {"func": FactorFeatureBuilder.backward_cum_return, "suffix": "cum_ret_60", "kwargs": {"days": 60}},
            ],
            "QMJ": [
                {"func": FactorFeatureBuilder.rolling_zscore, "suffix": "zscore", "kwargs": {"window": 2000}},
                {"func": FactorFeatureBuilder.mean_change, "suffix": "ma10_rate", "kwargs": {"period": 10}},
                {"func": FactorFeatureBuilder.mean_change, "suffix": "ma60_rate", "kwargs": {"period": 60}},
                {"func": FactorFeatureBuilder.volatility, "suffix": "vol20", "kwargs": {"window": 20}},
                {"func": FactorFeatureBuilder.rsi, "suffix": "rsi28", "kwargs": {"window": 28}},
                {"func": FactorFeatureBuilder.regression_slope_on_zscore, "suffix": "slope_z20", "kwargs": {"zscore_window": 20, "reg_window": 20}},
                {"func": FactorFeatureBuilder.regression_slope_diff_on_zscore, "suffix": "slope_diff_z20", "kwargs": {"zscore_window": 20, "reg_window": 20}},
                {"func": FactorFeatureBuilder.days_since_rsi_extreme, "suffix": "since_rsi_gt_70", "kwargs": {"window": 14, "threshold": 70, "mode": "gt"}},
                {"func": FactorFeatureBuilder.days_since_rsi_extreme, "suffix": "since_rsi_lt_30", "kwargs": {"window": 14, "threshold": 30, "mode": "lt"}},
                {"func": FactorFeatureBuilder.backward_cum_return, "suffix": "cum_ret_20", "kwargs": {"days": 20}},
                {"func": FactorFeatureBuilder.backward_cum_return, "suffix": "cum_ret_60", "kwargs": {"days": 60}},
            ],
            ("SMB", "HML"): [
                {"func": FactorFeatureBuilder.value_minus, "suffix": "diff_smb_hml"},
                {"func": FactorFeatureBuilder.regression_slope_on_zscore_of_diff, "suffix": "slope_z20", "kwargs": {"zscore_window": 20, "reg_window": 20}},
            ]
        }

        self.feature_assembler = FeatureAssembler(
            macro_feature_plan=self.macro_feature_plan,
            factor_feature_plan=self.factor_feature_plan
        )

    @staticmethod
    def calc_forward_return(df: pd.DataFrame, days: int) -> pd.DataFrame:
        """
        计算未来 days 天的复合收益率（从 t+1 到 t+days）
        :param df: 日度收益率 DataFrame（index=date, columns=factor_name）
        :param days: 向前看的交易日数
        :return: 每日复合收益率
        """
        return df.add(1).rolling(window=days, min_periods=days).apply(np.prod, raw=True).shift(-days) - 1
    
    @staticmethod
    def calc_forward_volatility(df: pd.DataFrame, days: int) -> pd.DataFrame:
        """
        计算未来 days 天的年化波动率（从 t+1 到 t+days）
        """
        return df.rolling(window=days, min_periods=days).std().shift(-days) * np.sqrt(252)

    def build(self, start: str = None, end: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        features = self.feature_assembler.assemble_features(start=start, end=end)
        factor_df = FactorDataReader.read_daily_factors(start=start, end=end)

        y_20d = self.calc_forward_return(factor_df, 20)
        y_60d = self.calc_forward_return(factor_df, 60)

        y_20d.columns = [f"{col}_20d_ret" for col in y_20d.columns]
        y_60d.columns = [f"{col}_60d_ret" for col in y_60d.columns]

        labels = pd.concat([y_20d, y_60d], axis=1)
        df_all = features.join(labels, how="inner").dropna()

        X_raw = df_all.drop(columns=labels.columns)
        Y = df_all[labels.columns]

        X = select_features_vif_pca(X_raw)

        return X, Y

    def train_test_split(self, X: pd.DataFrame, Y: pd.DataFrame, split_date: str) -> tuple:
        """
        将训练集和验证集按时间滚动分割，split_date 为分界日期（训练集 ≤ split_date，测试集 > split_date）
        """
        split_date = pd.to_datetime(split_date)
        X.index = pd.to_datetime(X.index)
        Y.index = pd.to_datetime(Y.index)
        X_train = X[X.index <= split_date]
        Y_train = Y[Y.index <= split_date]
        X_test = X[X.index > split_date]
        Y_test = Y[Y.index > split_date]
        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def train_test_split_ratio(X: pd.DataFrame, Y: pd.DataFrame, test_size: float = 0.2, shuffle: bool = False) -> tuple:
        """
        按照比例划分训练集和测试集。保持索引时间顺序（非随机打乱）。
        :param X: 特征数据
        :param Y: 标签数据
        :param test_size: 测试集占比（如 0.2）
        :param shuffle: 是否打乱（默认 False）
        """
        if shuffle:
            raise NotImplementedError("当前为时间序列任务，不建议打乱顺序。")

        n_samples = len(X)
        split_idx = int(n_samples * (1 - test_size))

        X_train = X.iloc[:split_idx]
        Y_train = Y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        Y_test = Y.iloc[split_idx:]
        return X_train, X_test, Y_train, Y_test