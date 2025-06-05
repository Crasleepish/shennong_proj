# app/ml/train.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error, accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRegressor, XGBClassifier
from app.ml.dataset_builder import DatasetBuilder
import joblib
import os
import logging
import optuna

logger = logging.getLogger(__name__)

REGRESSION_COLS = ["MKT_20d_vol"]
CLASSIFICATION_COLS = [
    "MKT_20d_tri", "MKT_60d_tri",
    "SMB_20d_tri", "SMB_60d_tri",
    "HML_20d_tri", "HML_60d_tri"
]

RESULT_DIR = "./ml_results"
os.makedirs(RESULT_DIR, exist_ok=True)

def train_one_split(start, end, split_date):
    logger.info("训练窗口：%s ~ %s，测试分割点：%s", start, end, split_date)
    builder = DatasetBuilder()
    X, Y = builder.build(start=start, end=end)
    X_train, X_test, Y_train, Y_test = builder.train_test_split(X, Y, split_date)

    results = []

    for col in Y.columns:
        if col in REGRESSION_COLS:
            model = XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1
            )
            model.fit(X_train, Y_train[col])
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            test_r2 = r2_score(Y_test[col], y_pred)
            test_rmse = root_mean_squared_error(Y_test[col], y_pred)
            train_r2 = r2_score(Y_train[col], y_train_pred)
            train_rmse = root_mean_squared_error(Y_train[col], y_train_pred)

            results.append({
                "target": col,
                "type": "regression",
                "train_r2": train_r2,
                "train_rmse": train_rmse,
                "test_r2": test_r2,
                "test_rmse": test_rmse
            })

            df_pred = pd.DataFrame({"actual": Y_test[col], "predicted": y_pred}, index=Y_test.index)
            df_pred.to_csv(f"{RESULT_DIR}/{split_date.replace('-', '')}_{col}_pred.csv")

            plt.figure(figsize=(10, 4))
            plt.plot(df_pred.index, df_pred["actual"], label="Actual", linewidth=1.5)
            plt.plot(df_pred.index, df_pred["predicted"], label="Predicted", linestyle="--")
            plt.title(f"{col} Prediction vs Actual ({split_date})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{RESULT_DIR}/{split_date.replace('-', '')}_{col}_plot.png")
            plt.close()

        elif col in CLASSIFICATION_COLS:
            model = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1,
                eval_metric="mlogloss", objective="multi:softmax", num_class=3
            )
            sample_weight = compute_sample_weight("balanced", Y_train[col])
            model.fit(X_train, Y_train[col], sample_weight=sample_weight)
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            test_acc = accuracy_score(Y_test[col], y_pred)
            test_f1 = f1_score(Y_test[col], y_pred, average="macro")
            train_acc = accuracy_score(Y_train[col], y_train_pred)
            train_f1 = f1_score(Y_train[col], y_train_pred, average="macro")

            logger.info("\n分类报告 [%s]:\n%s", col, classification_report(Y_test[col], y_pred))

            results.append({
                "target": col,
                "type": "classification",
                "train_acc": train_acc,
                "train_f1": train_f1,
                "test_acc": test_acc,
                "test_f1": test_f1
            })

        else:
            logger.warning("跳过未知目标变量: %s", col)
            continue

        out_dir = f"./models/{col}"
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(model, f"{out_dir}/model_{split_date}.pkl")

    return pd.DataFrame(results)

def rolling_train(start: str, split_dates: list[str]) -> pd.DataFrame:
    all_results = []
    for split_date in split_dates:
        end = pd.to_datetime(split_date) + pd.DateOffset(years=1)
        df_result = train_one_split(start=start, end=end.strftime("%Y-%m-%d"), split_date=split_date)
        df_result.insert(0, "split_date", split_date)
        all_results.append(df_result)

    df_all = pd.concat(all_results, axis=0).reset_index(drop=True)
    df_all.to_csv(f"{RESULT_DIR}/rolling_train_metrics.csv", index=False)

    for metric in ["test_r2", "test_rmse", "test_f1"]:
        metric_df = df_all[df_all.columns.intersection(["split_date", "target", metric])]
        if not metric_df.empty:
            pivoted = metric_df.pivot(index="split_date", columns="target", values=metric)
            pivoted.plot(marker="o", figsize=(10, 5))
            plt.title(f"{metric} Over Time")
            plt.ylabel(metric)
            plt.xlabel("Split Date")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{RESULT_DIR}/metric_trend_{metric}.png")
            plt.close()

    return df_all

def tune_model_with_optuna(target: str, task: str = "regression", n_trials: int = 50):
    builder = DatasetBuilder()
    X, Y = builder.build(start="2008-01-01", end="2013-12-31")
    X_train, X_val, Y_train, Y_val = builder.train_test_split_ratio(X, Y, test_size=0.2, shuffle=False)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": 42,
        }

        if task == "regression":
            model = XGBRegressor(**params)
            model.fit(X_train, Y_train[target])
            y_pred = model.predict(X_val)
            return r2_score(Y_val[target], y_pred)

        elif task == "classification":
            model = XGBClassifier(**params, eval_metric="mlogloss", objective="multi:softmax", num_class=3)
            sample_weight = compute_sample_weight("balanced", Y_train[target])
            model.fit(X_train, Y_train[target], sample_weight=sample_weight)
            y_pred = model.predict(X_val)
            return f1_score(Y_val[target], y_pred, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info("Best params for %s (%s): %s", target, task, study.best_params)
    logger.info("Best score: %.4f", study.best_value)
