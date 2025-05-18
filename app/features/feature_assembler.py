import pandas as pd
from app.features.macro_feature_pipeline import MacroFeaturePipeline
from app.features.macro_feature_builder import MacroFeatureBuilder
from app.data_fetcher.macro_data_reader import MacroDataReader


class FeatureAssembler:
    """
    用于组装完整的训练特征集（初始版本仅包含宏观数据处理）。
    后续可扩展添加因子收益、资产价格、情绪等特征模块。
    """

    def __init__(self, macro_feature_plan: dict):
        self.macro_pipeline = MacroFeaturePipeline(
            feature_plan=macro_feature_plan
        )

    def assemble_features(self, start: str = None, end: str = None) -> pd.DataFrame:
        raw_macro = MacroDataReader.read_all_macro_data(start=start, end=end)
        macro_features = self.macro_pipeline.transform(raw_macro)
        return macro_features
