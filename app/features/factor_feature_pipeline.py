import pandas as pd
from typing import Dict, List, Callable
from app.features.factor_feature_builder import FactorFeatureBuilder


class FactorFeaturePipeline:
    """
    因子特征构造 Pipeline：对传入的每日收益率 DataFrame，批量构造累计收益、滚动统计、斜率等特征。
    """

    def __init__(self, feature_plan: Dict[str, List[Dict]]):
        """
        初始化特征构造计划。

        :param feature_plan: 结构如下：
          {
            "MKT": [
              {"func": FactorFeatureBuilder.rolling_mean_change, "suffix": "ma10_chg", "kwargs": {"window": 10}},
              {"func": FactorFeatureBuilder.volatility, "suffix": "vol20", "kwargs": {"window": 20}},
              ...
            ],
            ("SMB", "HML"): [
              {"func": lambda a, b: a - b, "suffix": "spread"}
            ]
          }
        """
        self.feature_plan = feature_plan

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行特征构造。

        :param df: DataFrame，index=date, columns=factor_name（收益率）
        :return: 构造后的特征 DataFrame
        """
        df = df.copy()
        features = []

        for key, plans in self.feature_plan.items():
            if isinstance(key, str):
                cols = [key]
            else:
                cols = list(key)

            if not all(col in df.columns for col in cols):
                continue

            inputs = [df[col] for col in cols]

            for plan in plans:
                func: Callable = plan["func"]
                suffix: str = plan.get("suffix", func.__name__)
                kwargs = plan.get("kwargs", {})

                result = func(*inputs, **kwargs)
                result.name = f"{'-'.join(cols)}_{suffix}"
                features.append(result)

        df_result = pd.concat(features, axis=1).dropna()
        df_result.index.name = "date"
        return df_result
