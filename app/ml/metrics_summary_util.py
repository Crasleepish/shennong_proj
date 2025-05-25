# app/ml/metrics_summary_util.py

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def summarize_all_metrics(result_dir: str = "results"):
    all_metrics = []
    for file in glob.glob(os.path.join(result_dir, "*_metrics.csv")):
        split_date = os.path.basename(file).split("_")[0]
        df = pd.read_csv(file)
        df.insert(0, "split_date", split_date)
        all_metrics.append(df)

    if all_metrics:
        summary_df = pd.concat(all_metrics, axis=0).sort_values(["target", "split_date"])
        summary_path = os.path.join(result_dir, "all_metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info("All metrics summary saved to %s", summary_path)

        # 演示 R2 随时间的趋势图
        for target in summary_df["target"].unique():
            target_df = summary_df[summary_df["target"] == target]
            plt.figure(figsize=(8, 4))
            plt.plot(pd.to_datetime(target_df["split_date"]), target_df["test_R2"], marker="o")
            plt.title(f"R^2 over time - {target}")
            plt.xlabel("Split Date")
            plt.ylabel("R^2")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f"{target}_r2_trend.png"))
            plt.close()
    else:
        logger.warning("No metric CSV files found in %s", result_dir)
