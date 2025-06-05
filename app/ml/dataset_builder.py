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
from app.ml.preprocess import select_features_vif

class DatasetBuilder:
    def __init__(self):
        self.macro_plan = {
            # "社融": [
            #     {"func": MacroFeatureBuilder.yoy_growth, "suffix": "yoy"},
            #     {"func": MacroFeatureBuilder.mom_growth, "suffix": "mom"},
            #     {"func": MacroFeatureBuilder.rolling_mean, "suffix": "avg3", "kwargs": {"window": 3}},
            # ],
            # "PMI": [
            #     {"func": MacroFeatureBuilder.value_minus, "suffix": "diff_50", "kwargs": {"series2": pd.Series(50, index=pd.date_range("2000-01", "2100-01", freq="M"))}},
            #     {"func": MacroFeatureBuilder.rolling_slope, "suffix": "trend12", "kwargs": {"window": 12}},
            # ],
            # "外储": [
            #     {"func": MacroFeatureBuilder.yoy_growth, "suffix": "yoy"},
            #     {"func": MacroFeatureBuilder.rolling_log_slope, "suffix": "trend3", "kwargs": {"window": 3}},
            # ],
            # "黄金": [
            #     {"func": MacroFeatureBuilder.yoy_growth, "suffix": "yoy"},
            #     {"func": MacroFeatureBuilder.rolling_log_slope, "suffix": "trend3", "kwargs": {"window": 3}},
            # ],
            # "CPI": [
            #     {"func": MacroFeatureBuilder.raw, "suffix": "raw"},
            #     {"func": MacroFeatureBuilder.rolling_slope, "suffix": "trend12", "kwargs": {"window": 12}},
            # ],
            # ("BOND_10Y", "CPI"): [
            #     {"func": MacroFeatureBuilder.value_minus, "suffix": "real_rate"},
            #     {"func": MacroFeatureBuilder.diff_yoy, "suffix": "diff_yoy"},
            # ],
            # ("M1YOY", "M2YOY"): [
            #     {"func": MacroFeatureBuilder.value_minus, "suffix": "diff"},
            # ],
            # ("BOND_10Y", "BOND_2Y"): [
            #     {"func": MacroFeatureBuilder.value_minus, "suffix": "diff"},
            #     {"func": MacroFeatureBuilder.diff_yoy, "suffix": "diff_yoy"},
            # ]
        }

        self.mkt_plan = {
            "MKT_NAV": self._default_factor_plans()
        }
        self.smb_hml_plan = {
            "SMB_HML": self._default_factor_plans()
        }
        self.smb_qmj_plan = {
            "SMB_QMJ": self._default_factor_plans()
        }
        self.hml_qmj_plan = {
            "HML_QMJ": self._default_factor_plans()
        }

    @staticmethod
    def _default_factor_plans():
        return [
                # 均线偏离
                {"func": FactorFeatureBuilder.ma_diff, "suffix": "ma_diff_20", "kwargs": {"window": 20}},
                {"func": FactorFeatureBuilder.ma_diff, "suffix": "ma_diff_60", "kwargs": {"window": 60}},

                # 趋势交叉
                {"func": FactorFeatureBuilder.ma_cross_signal, "suffix": "ma10_gt_ma20", "kwargs": {"short_window": 10, "long_window": 20}},

                # RSI
                {"func": FactorFeatureBuilder.rsi, "suffix": "rsi_14", "kwargs": {"window": 14}},

                # z-score
                {"func": FactorFeatureBuilder.rolling_zscore, "suffix": "zscore_20", "kwargs": {"window": 20}},
                {"func": FactorFeatureBuilder.rolling_zscore, "suffix": "zscore_60", "kwargs": {"window": 60}},
                {"func": FactorFeatureBuilder.rolling_zscore, "suffix": "zscore_2000", "kwargs": {"window": 2000}},

                # 动量
                {"func": FactorFeatureBuilder.momentum, "suffix": "momentum_20", "kwargs": {"window": 20}},
                {"func": FactorFeatureBuilder.momentum, "suffix": "momentum_60", "kwargs": {"window": 60}},
                {"func": FactorFeatureBuilder.momentum, "suffix": "momentum_240", "kwargs": {"window": 240}},

                # 波动率
                {"func": FactorFeatureBuilder.volatility, "suffix": "vol_20", "kwargs": {"window": 20}},

                # 趋势斜率
                {"func": FactorFeatureBuilder.slope_of_ma, "suffix": "slope_ma20", "kwargs": {"ma_window": 20, "slope_window": 5}},
                {"func": FactorFeatureBuilder.slope_of_ma, "suffix": "slope_ma60", "kwargs": {"ma_window": 60, "slope_window": 5}},

                # 布林带位置归一化
                {"func": FactorFeatureBuilder.bb_position, "suffix": "bb_norm_20", "kwargs": {"window": 20, "num_std": 2.0}},

                # 布林突破信号
                {"func": FactorFeatureBuilder.bb_break_signal, "suffix": "bb_break_signal", "kwargs": {"window": 20, "num_std": 2.0}},
            ]

    @staticmethod
    def label_three_class(series: pd.Series, lower: float, upper: float) -> pd.Series:
        return series.apply(lambda x: 2 if x > upper else (0 if x < lower else 1))

    def _build_dataset(self, factor_plan: dict, target_series: pd.Series, start: str = None, end: str = None, vif: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        assembler = FeatureAssembler(macro_feature_plan=self.macro_plan, factor_feature_plan=factor_plan)
        df_feature = assembler.assemble_features(start, end)
        
        # 对齐标签
        idx = df_feature.index.intersection(target_series.index)
        target_series = target_series.reindex(idx)
        df_feature = df_feature.reindex(idx)
        df_feature['target'] = target_series.loc[df_feature.index]
        df_feature = df_feature.dropna()
        
        # 拆分特征和标签
        df_X = df_feature.drop(columns=['target'])
        df_Y = df_feature[['target']]
        
        # 特征选择+降维
        if vif:
            df_X = select_features_vif(df_X)
        
        # 再次确保标签和特征对齐（防止降维产生NA）
        df_Y = df_Y.loc[df_X.index]
        
        return df_X, df_Y

    def build_mkt_volatility(self, start: str = None, end: str = None, vif: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = FactorDataReader.read_factor_nav_ratios(start, end)
        target = df['MKT_NAV'].rolling(20).std().shift(-20)
        target = target.dropna()
        return self._build_dataset(self.mkt_plan, target, start, end, vif)

    def build_mkt_tri_class(self, start: str = None, end: str = None, vif: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = FactorDataReader.read_factor_nav_ratios(start, end)
        ma5 = df['MKT_NAV'].rolling(5).mean()
        future_ret = ma5.shift(-10) / ma5 - 1
        future_ret = future_ret.dropna()
        lower = future_ret.quantile(0.25)
        upper = future_ret.quantile(0.75)
        target = self.label_three_class(future_ret, lower=lower, upper=upper)
        return self._build_dataset(self.mkt_plan, target, start, end, vif)

    def build_smb_hml_tri(self, start: str = None, end: str = None, vif: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = FactorDataReader.read_factor_nav_ratios(start, end)
        ma10 = df['SMB_HML'].rolling(10).mean()
        future_ret = ma10.shift(-20) / ma10 - 1
        future_ret = future_ret.dropna()
        lower = future_ret.quantile(0.25)
        upper = future_ret.quantile(0.75)
        target = self.label_three_class(future_ret, lower=lower, upper=upper)
        return self._build_dataset(self.smb_hml_plan, target, start, end, vif)

    def build_smb_qmj_tri(self, start: str = None, end: str = None, vif: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = FactorDataReader.read_factor_nav_ratios(start, end)
        ma10 = df['SMB_QMJ'].rolling(10).mean()
        future_ret = ma10.shift(-20) / ma10 - 1
        future_ret = future_ret.dropna()
        lower = future_ret.quantile(0.25)
        upper = future_ret.quantile(0.75)
        target = self.label_three_class(future_ret, lower=lower, upper=upper)
        return self._build_dataset(self.smb_qmj_plan, target, start, end, vif)

    def build_hml_qmj_tri(self, start: str = None, end: str = None, vif: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = FactorDataReader.read_factor_nav_ratios(start, end)
        ma10 = df['HML_QMJ'].rolling(10).mean()
        future_ret = ma10.shift(-20) / ma10 - 1
        future_ret = future_ret.dropna()
        lower = future_ret.quantile(0.25)
        upper = future_ret.quantile(0.75)
        target = self.label_three_class(future_ret, lower=lower, upper=upper)
        return self._build_dataset(self.hml_qmj_plan, target, start, end, vif)

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