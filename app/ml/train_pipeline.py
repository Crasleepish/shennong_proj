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
    r2_score, root_mean_squared_error, accuracy_score,
    f1_score, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight
from app.ml.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)

RESULT_DIR = "./ml_results"
os.makedirs(RESULT_DIR, exist_ok=True)


def train_one_task(
    task: Literal["mkt_tri", "smb_tri", "hml_tri", "qmj_tri", "10Ybond_tri", "gold_tri"],
    start: str,
    end: str,
    split_date: str,
    need_test: bool = True
) -> dict:
    builder = DatasetBuilder()
    build_fn_map = {
        "mkt_tri": builder.build_mkt_tri_class,
        "smb_tri": builder.build_smb_tri,
        "hml_tri": builder.build_hml_tri,
        "qmj_tri": builder.build_qmj_tri,
        "10Ybond_tri": builder.build_10Ybond_tri,
        "gold_tri": builder.build_gold_tri,
    }

    result = {}

    build_fn = build_fn_map[task]
    X, Y, label_to_ret = build_fn(start=start, end=end)

    if need_test:
        X_train, X_test, Y_train, Y_test = builder.train_test_split(X, Y, split_date)
    else:
        X_train = X
        Y_train = Y
    
    X_train, X_val, Y_train, Y_val = builder.train_test_split_ratio(X, Y, test_size=0.1)

    target_col = Y.columns[0]
    task_type = "classification"

    if task_type == "regression":
        model = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1
        )
        model.fit(X_train, Y_train[target_col])
        if need_test:
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            result = {
                "target": target_col,
                "type": "regression",
                "train_r2": r2_score(Y_train[target_col], y_train_pred),
                "train_rmse": root_mean_squared_error(Y_train[target_col], y_train_pred),
                "test_r2": r2_score(Y_test[target_col], y_pred),
                "test_rmse": root_mean_squared_error(Y_test[target_col], y_pred),
            }

    else:
        model = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.02, subsample=0.9, colsample_bytree=0.9, 
            random_state=42, reg_alpha=2.0, reg_lambda=2.0, n_jobs=-1,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss", early_stopping_rounds=50
        )
        sample_weight = compute_sample_weight("balanced", Y_train[target_col])
        model.fit(X_train, Y_train[target_col], sample_weight=sample_weight, 
                  eval_set=[(X_val, Y_val[target_col])], 
                  verbose=True)
        if need_test:
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            result = {
                "target": task,
                "type": "classification",
                "train_acc": accuracy_score(Y_train[target_col], y_train_pred),
                "train_f1": f1_score(Y_train[target_col], y_train_pred, average="macro"),
                "test_acc": accuracy_score(Y_test[target_col], y_pred),
                "test_f1": f1_score(Y_test[target_col], y_pred, average="macro"),
            }

            logger.info("\n分类报告 [%s]:\n%s", task, classification_report(Y_test[target_col], y_pred))

    out_dir = f"./models/{task}"
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": X_train.columns.tolist(),
        "label_to_ret": label_to_ret
    }, f"{out_dir}/model_{split_date}.pkl")

    return result

def run_all_models(start, split_date: str = None, end: str = None, need_test: bool = True) -> pd.DataFrame:
    logger.info("开始训练所有模型，split_date=%s", split_date)
    tasks = ["mkt_tri", "smb_tri", "hml_tri", "qmj_tri", "10Ybond_tri", "gold_tri"]
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
    task: Literal["mkt_tri", "smb_tri", "hml_tri", "qmj_tri", "10Ybond_tri", "gold_tri"],
    n_trials: int = 50
):
    builder = DatasetBuilder()
    build_fn_map = {
        "mkt_tri": builder.build_mkt_tri_class,
        "smb_tri": builder.build_smb_tri,
        "hml_tri": builder.build_hml_tri,
        "qmj_tri": builder.build_qmj_tri,
        "gold_tri": builder.build_gold_tri,
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
