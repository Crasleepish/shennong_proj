# app/ml/train.py

import logging
import os
from app.ml.dataset_builder import DatasetBuilder
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from typing import List
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from app.ml.metrics_summary_util import summarize_all_metrics
import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULT_DIR = "ml_results"
os.makedirs(RESULT_DIR, exist_ok=True)

def train_one_split(start: str, end: str, split_date: str):
    logger.info("构建数据集: %s ~ %s", start, end)
    builder = DatasetBuilder()
    X, Y = builder.build(start=start, end=end)

    logger.info("Train/Test 分割日期: %s", split_date)
    X_train, X_test, Y_train, Y_test = builder.train_test_split(X, Y, split_date=split_date)

    logger.info("X_train: %s, X_test: %s", X_train.shape, X_test.shape)

    model = MultiOutputRegressor(XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.43,
        subsample=0.6,
        colsample_bytree=0.98,
        reg_alpha=2.0,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0,
    ))
    model.fit(X_train, Y_train)

    Y_pred = pd.DataFrame(model.predict(X_test), index=Y_test.index, columns=Y_test.columns)
    Y_pred_train = pd.DataFrame(model.predict(X_train), index=Y_train.index, columns=Y_train.columns)

    metrics = []
    for col in Y_test.columns:
        r2_test = r2_score(Y_test[col], Y_pred[col])
        rmse_test = root_mean_squared_error(Y_test[col], Y_pred[col])

        r2_train = r2_score(Y_train[col], Y_pred_train[col])
        rmse_train = root_mean_squared_error(Y_train[col], Y_pred_train[col])

        logger.info("[目标: %s] Train R^2: %.4f, RMSE: %.4f | Test R^2: %.4f, RMSE: %.4f", col, r2_train, rmse_train, r2_test, rmse_test)

        metrics.append({
            "target": col,
            "train_R2": r2_train,
            "train_RMSE": rmse_train,
            "test_R2": r2_test,
            "test_RMSE": rmse_test
        })

        # 图表对比
        plt.figure(figsize=(10, 4))
        plt.plot(Y_test[col], label="Actual", linewidth=1.5)
        plt.plot(Y_pred[col], label="Predicted", linestyle="--")
        plt.title(f"{col} Prediction vs Actual ({split_date})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/{split_date.replace('-', '')}_{col}_plot.png")
        plt.close()

    # 保存 CSV
    Y_test.to_csv(f"{RESULT_DIR}/{split_date.replace('-', '')}_Y_test.csv")
    Y_pred.to_csv(f"{RESULT_DIR}/{split_date.replace('-', '')}_Y_pred.csv")
    pd.DataFrame(metrics).to_csv(f"{RESULT_DIR}/{split_date.replace('-', '')}_metrics.csv", index=False)

    # 保存模型
    joblib.dump(model, f"{RESULT_DIR}/{split_date.replace('-', '')}_model.pkl")

    return model, X_test, Y_test, Y_pred

def rolling_train(start: str, split_dates: List[str]):
    """
    结束时间会随着轮次推移，如：
    start:2010-01-01 end:2015-12-31 split: 2014-12-31
    start:2010-01-01 end:2016-12-31 split: 2015-12-31
    start:2010-01-01 end:2017-12-31 split: 2016-12-31
    """
    results = []
    for split_date in split_dates:
        # end = split_date + 1 year
        split_dt = datetime.strptime(split_date, "%Y-%m-%d")
        end_dt = split_dt + relativedelta(years=1)
        end = end_dt.strftime("%Y-%m-%d")

        logger.info("\n===== Rolling Split @ %s =====", split_date)
        model, X_test, Y_test, Y_pred = train_one_split(start, end, split_date)
        results.append({
            "split_date": split_date,
            "model": model,
            "X_test": X_test,
            "Y_test": Y_test,
            "Y_pred": Y_pred,
        })

    summarize_all_metrics(result_dir=RESULT_DIR)
    return results

def tune_xgb_with_optuna():
    def objective(trial):
        builder = DatasetBuilder()
        X, Y = builder.build(start="2008-01-01", end="2013-12-31")
        X_train, X_val, Y_train, Y_val = builder.train_test_split_ratio(X, Y, test_size=0.2, shuffle=False)

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": 42,
            "verbosity": 0,
        }

        model = MultiOutputRegressor(XGBRegressor(**params))
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_val)

        score = 0
        for i, col in enumerate(Y_val.columns):
            score += r2_score(Y_val.iloc[:, i], Y_pred[:, i])
        return score / Y_val.shape[1]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best params:", study.best_params)
    print("Best mean R²:", study.best_value)

if __name__ == "__main__":
    date_list = [
        "2012-12-31", "2013-12-31", "2014-12-31",
        "2015-12-31", "2016-12-31", "2017-12-31",
        "2018-12-31", "2019-12-31", "2020-12-31"
    ]
    rolling_train(start="2008-01-01", split_dates=date_list)
