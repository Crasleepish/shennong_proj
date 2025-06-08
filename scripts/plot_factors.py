import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import create_app
from app.ml.train import *
import pandas as pd
import matplotlib.pyplot as plt
from app.data_fetcher.factor_data_reader import FactorDataReader
import joblib


app = create_app()

logger = logging.getLogger(__name__)



def plot_cumulative_nav(
    start: str,
    end: str,
    factors: list[str],
    mean: int = 0,
    title: str = "因子累计净值曲线",
    figsize: tuple = (10, 6),
    save_path: str = None
):
    """
    绘制指定因子的累计净值曲线（可选平滑）

    :param start: 起始日期 (YYYY-MM-DD)
    :param end: 结束日期 (YYYY-MM-DD)
    :param factors: 要绘制的因子名称列表
    :param mean: 平滑窗口大小（单位：天），0 表示不平滑
    :param title: 图表标题
    :param figsize: 图表尺寸
    :param save_path: 如果指定路径，将图像保存
    """
    df_ret = FactorDataReader.read_daily_factors(start, end)[factors].dropna()
    df_nav = (df_ret + 1).cumprod()

    if mean > 0:
        df_nav = df_nav.rolling(window=mean, min_periods=1).mean()

    plt.figure(figsize=figsize)
    for col in df_nav.columns:
        plt.plot(df_nav.index, df_nav[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_nav_ratio(
    start: str,
    end: str,
    ratio_pairs: list[tuple[str, str]],
    mean: int = 0,
    title: str = "因子净值比值曲线",
    figsize: tuple = (10, 6),
    save_path: str = None
):
    """
    绘制多个因子净值比值曲线（如 HML / MKT），支持滚动均值平滑

    :param start: 起始日期 (YYYY-MM-DD)
    :param end: 结束日期 (YYYY-MM-DD)
    :param ratio_pairs: 比值对列表 [(numerator, denominator), ...]
    :param mean: 平滑窗口大小（单位：天），0 表示不平滑
    :param title: 图表标题
    :param figsize: 图表尺寸
    :param save_path: 图像保存路径
    """
    all_factors = list(set([f for pair in ratio_pairs for f in pair]))
    df_ret = FactorDataReader.read_daily_factors(start, end)[all_factors].dropna()
    df_nav = (df_ret + 1).cumprod()

    df_ratio = pd.DataFrame(index=df_nav.index)
    for num, denom in ratio_pairs:
        colname = f"{num}/{denom}"
        df_ratio[colname] = df_nav[num] / df_nav[denom]

    if mean > 0:
        df_ratio = df_ratio.rolling(window=mean, min_periods=1).mean()

    plt.figure(figsize=figsize)
    for col in df_ratio.columns:
        plt.plot(df_ratio.index, df_ratio[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("净值比值")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def predict_with_model(task: str, start: str, end: str, model_path: str):
    builder = DatasetBuilder()
    build_fn_map = {
        "mkt_tri": builder.build_mkt_tri_class,
        "smb_tri": builder.build_smb_tri,
        "hml_tri": builder.build_hml_tri,
        "qmj_tri": builder.build_qmj_tri,
    }

    build_fn = build_fn_map[task]
    X, _ = build_fn(start=start, end=end, vif=False, inference=True)

    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    feature_names = model_bundle["features"]
    
    X = X[feature_names]
    y_pred = model.predict(X)

    # 获取 softprob 模式下的概率输出（每行是 [P0, P1, P2]）
    y_proba = model.predict_proba(X)

    # 获取最大概率作为置信度
    confidence = y_proba.max(axis=1)

    df_out = pd.DataFrame({
        "date": X.index,
        "pred": y_pred,
        "confidence": confidence
    }).set_index("date")

    return df_out

def plot_cumulative_nav_with_predictions(
    start: str,
    end: str,
    factor: str,
    pred_df: pd.DataFrame,
    mean: int = 0,
    title: str = "预测点标注的因子净值曲线",
    figsize: tuple = (12, 6),
    save_path: str = None
):
    """
    绘制带预测标签标注的单因子累计净值曲线。
    
    :param start: 起始日期
    :param end: 结束日期
    :param factor: 因子名称，如 "SMB"
    :param pred_df: 包含预测值的 DataFrame，需包含 "pred" 和索引为日期
    :param mean: 滚动平滑窗口
    :param title: 图标题
    :param figsize: 图尺寸
    :param save_path: 保存路径
    """
    df_ret = FactorDataReader.read_daily_factors(start, end)[[factor]].dropna()
    df_nav = (df_ret + 1).cumprod()

    if mean > 0:
        df_nav = df_nav.rolling(window=mean, min_periods=1).mean()

    plt.figure(figsize=figsize)
    plt.plot(df_nav.index, df_nav[factor], label=factor, linewidth=1.5)

    for label, color in [(0, "red"), (2, "blue")]:
        idx = pred_df[pred_df["pred"] == label].index
        idx = idx.intersection(df_nav.index)
        plt.scatter(idx, df_nav.loc[idx, factor], color=color, label=f"Predicted {label}", s=30)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_nav_ratio_with_predictions(
    start: str,
    end: str,
    ratio_pair: tuple[str, str],
    pred_df: pd.DataFrame,
    mean: int = 0,
    title: str = "预测点标注的净值比值曲线",
    figsize: tuple = (12, 6),
    save_path: str = None
):
    num, denom = ratio_pair
    df_ret = FactorDataReader.read_daily_factors(start, end)[[num, denom]].dropna()
    df_nav = (df_ret + 1).cumprod()
    df_ratio = df_nav[num] / df_nav[denom]

    if mean > 0:
        df_ratio = df_ratio.rolling(window=mean, min_periods=1).mean()

    plt.figure(figsize=figsize)
    plt.plot(df_ratio.index, df_ratio.values, label=f"{num}/{denom}", linewidth=1.5)

    # 标注预测点
    for label, color in [(0, "red"), (2, "blue")]:
        idx = pred_df[pred_df["pred"] == label].index
        idx = idx.intersection(df_ratio.index)  # 确保存在于ratio中
        plt.scatter(idx, df_ratio.loc[idx], color=color, label=f"Predicted {label}", s=30)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("NAV Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def run_predict_and_export():
    df_out = predict_with_model(
        task="smb_tri",
        start="2023-01-01",
        end="2025-06-06",
        model_path="./models/smb_tri/model_2024-06-30.pkl"
    )

    df_out.to_csv("./ml_results/smb_tri_pred_vs_true.csv")

    # plot_nav_ratio_with_predictions(
    #     start="2023-01-01",
    #     end="2025-05-13",
    #     ratio_pair=("SMB", "QMJ"),
    #     pred_df=df_out,
    #     mean=10,
    #     title="SMB/QMJ 净值比值曲线（预测类别点标注）",
    #     save_path="./ml_results/smb_qmj_tri_plot_with_preds.png"
    # )

    plot_cumulative_nav_with_predictions(
        start="2023-01-01",
        end="2025-06-06",
        factor="SMB",
        pred_df=df_out,
        mean=10,
        title="SMB 累计净值曲线（预测类别点标注）",
        save_path="./ml_results/smb_nav_plot_with_preds.png"
    )


if __name__ == '__main__':
    with app.app_context():
        # plot_cumulative_nav(start='2019-01-04', end='2025-05-13', factors=['SMB', 'HML', 'QMJ'], title='因子净值曲线', mean=60)
        # plot_nav_ratio(start='2019-01-04', end='2025-05-13', ratio_pairs=[('HML', 'QMJ'), ('SMB', 'QMJ')], title='因子净值比值曲线', mean=60)
        run_predict_and_export()
