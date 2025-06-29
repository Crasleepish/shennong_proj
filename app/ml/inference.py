# 文件路径建议：app/ml/inference.py

import joblib
import numpy as np
import pandas as pd
from typing import Literal, Dict
from app.ml.dataset_builder import DatasetBuilder
from datetime import datetime

# 模型路径模板常量
TASK_MODEL_PATHS = {
    "mkt_tri": "./models/mkt_tri/model_{0}.pkl",
    "smb_tri": "./models/smb_tri/model_{0}.pkl",
    "hml_tri": "./models/hml_tri/model_{0}.pkl",
    "qmj_tri": "./models/qmj_tri/model_{0}.pkl",
    "10Ybond_tri": "./models/10Ybond_tri/model_{0}.pkl",
    "gold_tri": "./models/gold_tri/model_{0}.pkl",
}

MODEL_DATE = "2024-06-30"

def load_model_and_features(model_path: str):
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    feature_names = model_bundle["features"]
    return model, feature_names

def predict_softprob(
    task: Literal["mkt_tri", "smb_tri", "hml_tri", "qmj_tri", "10Ybond_tri", "gold_tri"],
    start: str,
    end: str,
    model_path: str,
    data_start_before: int = 3100,
    dataset_builder: DatasetBuilder = None
) -> pd.DataFrame:
    """
    对指定任务进行推理，返回 softprob 预测结果。

    参数：
    task (str): 待预测的因子任务，可选值为 "mkt_tri", "smb_tri", "hml_tri", "qmj_tri", "10Ybond_tri", "gold_tri"。
    start (str): 预测起始日期，格式为 "YYYY-MM-DD"。
    end (str): 预测结束日期，格式为 "YYYY-MM-DD"。
    model_path (str): 模型保存路径。
    data_start_before (int): 由于特征构建需要依赖历史较早的数据，此参数为特征数据起始日期之前的天数，默认为 3100，约为2000个交易日前，如果该值过小，可能导致提取的特征质量较差

    输出格式：
    DataFrame, index 为预测日期，列为 softmax 概率，如：
        MKT_prob0, MKT_prob1, MKT_prob2
        SMB_prob0, SMB_prob1, SMB_prob2
        等按 task 决定。
    每行的概率和为 1，对应三分类预测的置信度分布。
    """
    if dataset_builder is None:
        builder = DatasetBuilder()
    else:
        builder = dataset_builder
    build_fn_map = {
        "mkt_tri": builder.build_mkt_tri_class,
        "smb_tri": builder.build_smb_tri,
        "hml_tri": builder.build_hml_tri,
        "qmj_tri": builder.build_qmj_tri,
        "10Ybond_tri": builder.build_10Ybond_tri,
        "gold_tri": builder.build_gold_tri,
    }

    horizon_start = pd.to_datetime(start) - pd.Timedelta(days=data_start_before)

    build_fn = build_fn_map[task]
    X, _, _ = build_fn(start=datetime.strftime(horizon_start, "%Y-%m-%d"), end=end, vif=False, inference=True)
    X = X.loc[datetime.strptime(start, "%Y-%m-%d").date():datetime.strptime(end, "%Y-%m-%d").date()]

    model, feature_names = load_model_and_features(model_path)
    X = X[feature_names]
    y_proba = model.predict_proba(X)  # shape: [n_samples, 3]

    df_proba = pd.DataFrame(
        y_proba,
        index=X.index,
        columns=[f"{task.split('_')[0].upper()}_prob{i}" for i in range(y_proba.shape[1])]
    )
    return df_proba

def get_softprob_dict(trade_date: str, dataset_builder: DatasetBuilder = None, horizon_days: int = 10) -> Dict[str, np.ndarray]:
    """
    返回指定交易日对应的 softprob_dict 结构，格式如下：
    {
        'MKT': [p0, p1, p2],
        'SMB': [p0, p1, p2],
        ...
    }
    """
    softprob_dict = {}
    horizon_start = pd.to_datetime(trade_date) - pd.Timedelta(days=horizon_days)
    start = horizon_start.strftime("%Y-%m-%d")
    end = trade_date

    for task, path_template in TASK_MODEL_PATHS.items():
        model_path = path_template.format(MODEL_DATE)
        df_proba = predict_softprob(task, start=start, end=end, model_path=model_path, dataset_builder=dataset_builder)
        if df_proba.empty:
            raise ValueError(f"Softprob为空: {task} @ {trade_date}")
        last_row = df_proba.iloc[-1]
        factor = task.split("_")[0].upper()
        softprob_dict[factor] = last_row.values

    return softprob_dict

def get_label_to_ret() -> Dict[str, tuple]:
    """
    加载每个三分类模型的 label_to_ret（用于观点收益映射）
    返回结构：{ 'MKT': (ret0, ret1, ret2), ... }
    """
    label_to_ret = {}
    for task, path_template in TASK_MODEL_PATHS.items():
        model_path = path_template.format(MODEL_DATE)
        bundle = joblib.load(model_path)
        if "label_to_ret" not in bundle:
            raise ValueError(f"模型文件缺失 label_to_ret: {model_path}")
        factor = task.split("_")[0].upper()
        label_to_ret[factor] = bundle["label_to_ret"]

    return label_to_ret
