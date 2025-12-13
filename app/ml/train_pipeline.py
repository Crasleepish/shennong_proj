# app/ml/train_pipeline.py

import os
import joblib
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import optuna
from typing import Literal
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    accuracy_score,
    f1_score,
    classification_report,
    log_loss,  # ✅ 新增：用于计算 val_loss
)
from sklearn.utils.class_weight import compute_sample_weight

from app.ml.dataset_builder import DatasetBuilder  # ✅ 新增：明确导入


logger = logging.getLogger(__name__)

RESULT_DIR = "./ml_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# === 超参数：默认 + 任务级覆盖 ===
DEFAULT_XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_estimators=500,
    learning_rate=0.02,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=1.0,
    reg_lambda=2.0,
    reg_alpha=2.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

# 针对每个任务单独覆盖（未列出的任务走 DEFAULT_XGB_PARAMS）
TASK_HPARAMS = {
    # 四因子（示例：与默认一致或轻微调整）
    "mkt_tri": dict(
        n_estimators=1500,
        learning_rate=0.01,
        max_depth=5,
        min_child_weight=2.0,
        gamma=0.1,
        reg_lambda=4.0,
        reg_alpha=4.0,
        early_stopping_rounds=50,
    ),
    "smb_tri": dict(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=5.0,
        subsample=0.7,
        colsample_bytree=0.5,
        gamma=0.5,
        reg_lambda=5.0,
        reg_alpha=2.0,
        early_stopping_rounds=50,
    ),
    "hml_tri": dict(
        n_estimators=1500,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=2.0,
        gamma=0.1,
        reg_lambda=2.0,
        reg_alpha=2.0,
        early_stopping_rounds=50,
    ),
    "qmj_tri": dict(
        n_estimators=1500,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=2.0,
        gamma=0.1,
        reg_lambda=2.0,
        reg_alpha=2.0,
        early_stopping_rounds=50,
    ),

    "10Ybond_tri": dict(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=2.0,
        gamma=0.1,
        reg_lambda=2.0,
        reg_alpha=2.0,
        early_stopping_rounds=50,
    ),

}

def get_params_for_task(task: str) -> dict:
    p = DEFAULT_XGB_PARAMS.copy()
    p.update(TASK_HPARAMS.get(task, {}))
    return p


def compute_recency_weight(index: pd.Index, decay_half_life: int = 750) -> np.ndarray:
    """
    计算时间衰减权重：
    - index: X_train.index（DatetimeIndex 或 object，可转换为日期）
    - decay_half_life: 半衰期（单位：天），默认750天≈3年
      权重公式： w = 0.5^(age_days / half_life)

    返回：
        ndarray of shape (n_samples,)
    """
    # 转换成 datetime 对象
    dates = pd.to_datetime(index)

    # 最新日期（越接近这个日期 → 权重越大）
    max_date = dates.max()

    # 计算每个样本距离 max_date 的天数
    age_days = (max_date - dates).days.values

    # 半衰期公式（指数衰减）
    w = 0.5 ** (age_days / decay_half_life)

    # 避免极端权重（太小会数值不稳定）
    w = np.clip(w, 1e-6, 1.0)

    return w


def train_one_task(
    task: Literal["mkt_tri", "smb_tri", "hml_tri", "qmj_tri", "10Ybond_tri"],
    start: str,
    end: str,
    split_date: str,
    need_test: bool = True,
) -> dict:
    """
    训练单个任务（如 mkt_tri / 10Ybond_tri）。

    重构要点：
    1. 使用时间窗口：训练 + 验证共 14 年，其中：
       - 最近 6 个月作为验证集（val）
       - 更早的时间段作为训练集（train）
    2. 如果 need_test=True，则在验证集之后的时间段作为测试集（test）
    3. 训练完成后，将 val_loss 一并写入模型文件，供 BL 中 omega 缩放使用。
    4. 训练完成后，将 (year_month, val_loss) 写入 ./models/{task}_val_history.csv，
       以“训练数据结束日期的年-月”为索引（year_month），保证唯一性（同年-月覆盖旧值）。
    """
    # ===== 0. 准备构建器 & 任务映射 =====
    builder = DatasetBuilder()
    build_fn_map = {
        "mkt_tri": builder.build_mkt_tri_class,
        # "smb_tri": builder.build_smb_tri,
        # "hml_tri": builder.build_hml_tri,
        # "qmj_tri": builder.build_qmj_tri,
        "10Ybond_tri": builder.build_10Ybond_tri,
        # "gold_tri": builder.build_gold_tri,  # 如以后启用，可按同样逻辑接入
    }

    result: dict = {}

    if task not in build_fn_map:
        raise ValueError(f"未知任务: {task}")

    build_fn = build_fn_map[task]

    # ===== 1. 构造完整数据集 X, Y, label_to_ret =====
    # 这里的 start/end 是“粗范围”，真正用于训练/验证/测试的时间窗口在下面再切。
    X, Y, label_to_ret = build_fn(start=start, end=end)

    if X.empty or Y.empty:
        logger.warning(
            "train_one_task[%s]: 在区间 [%s, %s] 内构造的数据集为空，跳过训练。",
            task, start, end,
        )
        return {"target": task, "type": "classification", "note": "empty_dataset"}

    # 统一索引为 DatetimeIndex，并按时间排序
    X = X.copy()
    Y = Y.copy()
    X.index = pd.to_datetime(X.index)
    Y.index = pd.to_datetime(Y.index)
    X = X.sort_index()
    Y = Y.sort_index()

    # 标签列名（目前 Y 只有一列）
    target_col = Y.columns[0]

    # ===== 2. 确定“模型日期 as_of”以及训练/验证时间窗口 =====
    # 约定：
    # - as_of = split_date（若为 None 则用 end）
    # - 验证集区间：[val_start, val_end]，长度约 6 个月
    # - 训练集区间：[train_start, val_start)
    # - 训练+验证总窗口长度 ≈ 14 年（不足时用所有历史）
    if split_date is None:
        as_of = pd.to_datetime(end)
    else:
        as_of = pd.to_datetime(split_date)

    val_end = as_of
    # 最近 6 个月作为验证集（滚动窗口）
    val_start = val_end - pd.DateOffset(months=6)
    # 训练 + 验证总窗口 14 年
    train_window_years = 14
    train_start = val_start - pd.DateOffset(years=train_window_years)

    # 实际数据的全局范围
    data_start = X.index.min()
    data_end = X.index.max()

    # 将 train_start 向右截断到数据最早日期
    if train_start < data_start:
        logger.info(
            "train_one_task[%s]: 数据起点 %s 晚于计划训练起点 %s，"
            "实际训练起点调整为数据起点。",
            task, data_start.date(), train_start.date(),
        )
        train_start = data_start

    # 如果 val_start 早于数据起点，也向右截断
    if val_start < data_start:
        val_start = data_start

    # 如果 val_end 超过数据末尾，则向左截断（一般不会发生，但以防万一）
    if val_end > data_end:
        val_end = data_end

    # ===== 3. 按时间切割出 train / val / test =====
    # 先裁掉训练窗口之前和 end 之后的数据（end 通常 >= as_of）
    end_ts = pd.to_datetime(end)
    mask_all = (X.index >= train_start) & (X.index <= end_ts)
    X_all = X.loc[mask_all]
    Y_all = Y.loc[mask_all]

    if X_all.empty:
        logger.warning(
            "train_one_task[%s]: 在 [%s, %s] 内无有效样本，跳过训练。",
            task, train_start.date(), end_ts.date(),
        )
        return {"target": task, "type": "classification", "note": "empty_after_window"}

    # 训练+验证：索引 <= val_end
    mask_trainval = X_all.index <= val_end
    X_trainval = X_all.loc[mask_trainval]
    Y_trainval = Y_all.loc[mask_trainval]

    if len(X_trainval) < 50:
        logger.warning(
            "train_one_task[%s]: 训练+验证样本数仅 %d，可能不足以支撑模型训练。",
            task, len(X_trainval),
        )

    # 验证集：索引 >= val_start
    mask_val = X_trainval.index >= val_start
    X_val = X_trainval.loc[mask_val]
    Y_val = Y_trainval.loc[mask_val]
    # 训练集：其余
    X_train = X_trainval.loc[~mask_val]
    Y_train = Y_trainval.loc[~mask_val]

    # 如果验证集为空，退化为最后 10% 为验证集（兜底）
    if X_val.empty or Y_val.empty:
        logger.warning(
            "train_one_task[%s]: 按 6 个月窗口划分得到的验证集为空，"
            "退化为使用最后 10%% 样本作为验证集。",
            task,
        )
        n_samples = len(X_trainval)
        split_idx = int(n_samples * 0.9)
        X_train = X_trainval.iloc[:split_idx]
        Y_train = Y_trainval.iloc[:split_idx]
        X_val = X_trainval.iloc[split_idx:]
        Y_val = Y_trainval.iloc[split_idx:]

        # 同时把 val_start/val_end 修正为兜底窗口，方便后续写入 meta & val_history
        if len(X_val) > 0:
            val_start = X_val.index.min()
            val_end = X_val.index.max()

    # 测试集：如果 need_test=True 且 end_ts > val_end，则 (val_end, end_ts] 作为测试集
    X_test = None
    Y_test = None
    if need_test:
        mask_test = X_all.index > val_end
        X_test = X_all.loc[mask_test]
        Y_test = Y_all.loc[mask_test]
        if X_test.empty:
            logger.info(
                "train_one_task[%s]: need_test=True，但在 (%s, %s] 内没有样本，测试集为空。",
                task, val_end.date(), end_ts.date(),
            )

    # ===== 4. 拟合 XGBoost 模型 =====
    params = get_params_for_task(task)
    params_with_metric = {
        **params,
        "eval_metric": "mlogloss",
    }
    model = XGBClassifier(**params_with_metric)

    # 类别不平衡权重（保持原来逻辑）
    class_weight = compute_sample_weight("balanced", Y_train[target_col])
    # 如需加入时间衰减，可改为：sample_weight = class_weight * compute_recency_weight(X_train.index)
    sample_weight = class_weight * compute_recency_weight(X_train.index)

    logger.info(
        "train_one_task[%s]: sample_weight stats -> min=%.6f, max=%.6f, mean=%.6f",
        task,
        float(np.min(sample_weight)),
        float(np.max(sample_weight)),
        float(np.mean(sample_weight)),
    )

    model.fit(
        X_train,
        Y_train[target_col],
        sample_weight=sample_weight,
        eval_set=[(X_val, Y_val[target_col])],
        verbose=True,
    )

    # ===== 5. 计算验证集损失（val_loss）并组织结果指标 =====
    try:
        y_val_proba = model.predict_proba(X_val)
        from sklearn.metrics import log_loss
        val_loss = float(
            log_loss(
                Y_val[target_col],
                y_val_proba,
                labels=np.unique(Y_train[target_col]),
            )
        )
    except Exception as e:
        logger.warning(
            "train_one_task[%s]: 计算 val_loss 时出错：%s，val_loss 记为 NaN。",
            task, repr(e),
        )
        val_loss = float("nan")

    if need_test and X_test is not None and len(X_test) > 0:
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        result = {
            "target": task,
            "type": "classification",
            "train_acc": accuracy_score(Y_train[target_col], y_train_pred),
            "train_f1": f1_score(Y_train[target_col], y_train_pred, average="macro"),
            "test_acc": accuracy_score(Y_test[target_col], y_pred),
            "test_f1": f1_score(Y_test[target_col], y_pred, average="macro"),
            "val_loss": val_loss,
        }

        logger.info(
            "\n分类报告 [%s]:\n%s",
            task,
            classification_report(Y_test[target_col], y_pred),
        )
    else:
        # 线上训练（need_test=False）或无测试样本时，只返回训练指标和 val_loss
        y_train_pred = model.predict(X_train)
        result = {
            "target": task,
            "type": "classification",
            "train_acc": accuracy_score(Y_train[target_col], y_train_pred),
            "train_f1": f1_score(Y_train[target_col], y_train_pred, average="macro"),
            "val_loss": val_loss,
        }

    # ===== 6. 保存模型文件：增加 val_loss 和时间窗口 meta 信息 =====
    out_dir = f"./models/{task}"
    os.makedirs(out_dir, exist_ok=True)

    model_meta = {
        "model": model,
        "features": X_train.columns.tolist(),
        "label_to_ret": label_to_ret,
        "val_loss": val_loss,  # ✅ 供后续 BL / omega 使用
        "train_window": {
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": (val_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if len(X_train) > 0
            else None,
            "val_start": val_start.strftime("%Y-%m-%d"),
            "val_end": val_end.strftime("%Y-%m-%d"),
            "test_start": (
                (val_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if need_test and X_test is not None and len(X_test) > 0
                else None
            ),
            "test_end": end_ts.strftime("%Y-%m-%d") if need_test else None,
        },
    }

    model_path = f"{out_dir}/model_{split_date}.pkl"
    joblib.dump(model_meta, model_path)
    logger.info(
        "train_one_task[%s]: 模型已保存至 %s（val_loss=%.6f）。",
        task,
        model_path,
        val_loss,
    )

    # ===== 7. 更新 val_history CSV：models/{task}_val_history.csv =====
    # 以“训练数据结束日期的 年-月”作为索引键，这里采用验证集结束日期 val_end 对应的 year-month。
    year_month = val_end.strftime("%Y-%m")
    history_path = f"./models/{task}_val_history.csv"

    try:
        new_row = pd.DataFrame(
            [{"year_month": year_month, "val_loss": val_loss}]
        )

        if os.path.exists(history_path):
            df_hist = pd.read_csv(history_path)
            # 统一成字符串，保证比较一致
            if "year_month" in df_hist.columns:
                df_hist["year_month"] = df_hist["year_month"].astype(str)
                # 去除同 year_month 的旧记录，保持索引唯一性（覆盖旧值）
                df_hist = df_hist[df_hist["year_month"] != year_month]
            else:
                # 如果旧文件结构异常，直接丢弃旧内容
                df_hist = pd.DataFrame(columns=["year_month", "val_loss"])

            df_hist = pd.concat([df_hist, new_row], ignore_index=True)
        else:
            df_hist = new_row

        # 按 year_month 排序，方便后续计算分位数等
        df_hist = df_hist.sort_values("year_month")
        df_hist.to_csv(history_path, index=False)

        logger.info(
            "train_one_task[%s]: 更新 val_history: %s (val_loss=%.6f)",
            task,
            history_path,
            val_loss,
        )
    except Exception as e:
        logger.warning(
            "train_one_task[%s]: 写入 val_history CSV 失败: %s",
            task,
            repr(e),
        )

    return result

def run_all_models(start, split_date: str = None, end: str = None, need_test: bool = True) -> pd.DataFrame:
    logger.info("开始训练所有模型，split_date=%s", split_date)
    tasks = ["mkt_tri", "10Ybond_tri"]
    if split_date is None:
        split_date = end
    if need_test:
        start = start
        end = pd.to_datetime(split_date) + pd.DateOffset(years=1)

        results = [
            train_one_task(task, start, end.strftime("%Y-%m-%d"), split_date, need_test = need_test)
            for task in tasks
        ]
    else:
        results = [
            train_one_task(task, start, end, split_date, need_test = need_test)
            for task in tasks
        ]

    df_result = pd.DataFrame(results)
    df_result.insert(0, "split_date", split_date)
    df_result.to_csv(f"{RESULT_DIR}/train_summary_{split_date}.csv", index=False)
    return df_result


def rolling_train(start: str, split_dates: list[str]) -> pd.DataFrame:
    all_results = []
    for split_date in split_dates:
        df_result = run_all_models(start, split_date=split_date)
        all_results.append(df_result)

    df_all = pd.concat(all_results, axis=0)
    df_all.to_csv(f"{RESULT_DIR}/rolling_train_metrics.csv", index=False)
    return df_all


def tune_with_optuna(
    task: Literal["mkt_tri", "smb_tri", "hml_tri", "qmj_tri", "10Ybond_tri"],
    n_trials: int = 50
):
    builder = DatasetBuilder()
    build_fn_map = {
        "mkt_tri": builder.build_mkt_tri_class,
        "smb_tri": builder.build_smb_tri,
        "hml_tri": builder.build_hml_tri,
        "qmj_tri": builder.build_qmj_tri,
        # "gold_tri": builder.build_gold_tri,
    }
    build_fn = build_fn_map[task]

    X, Y, _ = build_fn(start="2008-01-01", end="2016-12-31")
    X_train, X_val, Y_train, Y_val = builder.train_test_split_ratio(X, Y, test_size=0.2)
    target = Y.columns[0]
    task_type = "classification"

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 700),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.9, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.9, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 3.0),
            "random_state": 42,
        }

        if task_type == "regression":
            model = XGBRegressor(**params)
            model.fit(X_train, Y_train[target])
            y_pred = model.predict(X_val)
            return r2_score(Y_val[target], y_pred)
        else:
            model = XGBClassifier(**params, objective="multi:softmax", num_class=3, eval_metric="mlogloss")
            sample_weight = compute_sample_weight("balanced", Y_train[target])
            model.fit(X_train, Y_train[target], sample_weight=sample_weight)
            y_pred = model.predict(X_val)
            return f1_score(Y_val[target], y_pred, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info("Best params for %s: %s", task, study.best_params)
    logger.info("Best score: %.4f", study.best_value)
    return study.best_params
