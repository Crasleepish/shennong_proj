# app/features/macro_feature_pipeline.py

import pandas as pd
from typing import Callable, Dict, List, Union
from .macro_feature_builder import MacroFeatureBuilder


class MacroFeaturePipeline:
    """
    批量特征构造 Pipeline：对传入的 DataFrame 中的多个指标字段，按配置应用不同的构造函数。
    支持派生中间列和多列函数（如 f(x1, x2, ..., xn) → 特征）
    """

    def __init__(self, feature_plan: Dict[str, List[Dict]]):
        """
        初始化特征构造计划

        :param feature_plan: 字典结构如下：
          {
            "社融": [
                {"func": MacroFeatureBuilder.yoy_growth, "suffix": "yoy"},
                {"func": MacroFeatureBuilder.rolling_slope, "suffix": "slope3", "kwargs": {"window": 3}}
            ],
            ("LPR1Y", "CPI"): [
                {"func": lambda a, b: a - b, "suffix": "real_rate"}
            ]
          }
        - key 可以是 str 或 tuple[str, ...]，表示函数输入列名
        """
        self.feature_plan = feature_plan

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据计划对输入数据构造特征

        :param df: 时间序列型 DataFrame（index = 日期, columns = 指标）
        :return: 新构造的特征 DataFrame（多列）
        """
        df = df.copy()
        features = []

        for key, plan_list in self.feature_plan.items():
            if isinstance(key, str):
                col_names = [key]
            else:
                col_names = list(key)

            if not all(col in df.columns for col in col_names):
                continue  # 某些列不存在，跳过

            series_inputs = [df[col] for col in col_names]

            for plan in plan_list:
                func: Callable = plan["func"]
                suffix: str = plan.get("suffix", func.__name__)
                kwargs = plan.get("kwargs", {})

                result_series = func(*series_inputs, **kwargs)
                input_name = "-".join(col_names)
                new_col_name = f"{input_name}_{suffix}"
                result_series.name = new_col_name
                features.append(result_series)

        result = pd.concat(features, axis=1).dropna()
        result.index.name = "date"
        return result
