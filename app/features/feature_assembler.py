import pandas as pd
from app.features.macro_feature_pipeline import MacroFeaturePipeline
from app.features.factor_feature_pipeline import FactorFeaturePipeline
from app.features.macro_feature_builder import MacroFeatureBuilder
from app.data_fetcher.macro_data_reader import MacroDataReader
from app.data_fetcher.factor_data_reader import FactorDataReader

class FeatureAssembler:
    """
    用于组装完整的训练特征集（初始版本仅包含宏观数据处理）。
    后续可扩展添加因子收益、资产价格、情绪等特征模块。
    """

    def __init__(self, macro_feature_plan: dict, data_feature_plan: dict):
        self.macro_pipeline = MacroFeaturePipeline(
            feature_plan=macro_feature_plan
        )
        self.factor_pipeline = FactorFeaturePipeline(
            feature_plan=data_feature_plan
        )

    def assemble_features(self, macro_df, data_df, start: str = None, end: str = None) -> pd.DataFrame:
        if self.macro_pipeline:
            macro_features =  pd.DataFrame()
        else:
            # macro_df = MacroDataReader.read_all_macro_data(start, end)
            macro_features = self.macro_pipeline.transform(macro_df)

        # factor_df = FactorDataReader.read_factor_nav(start, end)
        factor_features = self.factor_pipeline.transform(data_df)
        
        if not macro_features.empty:
            # 将月度宏观特征扩展为日度：按 forward fill 补全到每日
            macro_features_daily = macro_features.reindex(factor_features.index).ffill()
        else:
            macro_features_daily = pd.DataFrame()

        # 合并并对齐特征
        combined = pd.concat([macro_features_daily, factor_features], axis=1).dropna()
        combined.index.name = "date"
        return combined
