import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import create_app
from app.ml.train import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from app.data_fetcher.factor_data_reader import FactorDataReader
from app.data_fetcher.csi_index_data_fetcher import CSIIndexDataFetcher
from app.ml.inference import predict_softprob
import joblib


app = create_app()

logger = logging.getLogger(__name__)

factor_data_reader = FactorDataReader()
csi_index_data_fetcher = CSIIndexDataFetcher()

# 设置字体路径
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
chinese_font = FontProperties(fname=font_path)

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
    df_ret = factor_data_reader.read_daily_factors(start, end)[factors].dropna()
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
    df_ret = factor_data_reader.read_daily_factors(start, end)[all_factors].dropna()
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

def plot_cumulative_nav_with_predictions(
    df_nav: pd.DataFrame,
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
    
    :param df_nav: 累计净值数据
    :param start: 起始日期
    :param end: 结束日期
    :param factor: 因子名称，如 "SMB"
    :param pred_df: 包含预测值的 DataFrame，需包含 "pred" 和索引为日期
    :param mean: 滚动平滑窗口
    :param title: 图标题
    :param figsize: 图尺寸
    :param save_path: 保存路径
    """
    if mean > 0:
        df_nav = df_nav.rolling(window=mean, min_periods=1).mean()

    plt.figure(figsize=figsize)
    plt.plot(df_nav.index, df_nav[factor], label=factor, linewidth=1.5)

    for label, color in [(0, "red"), (2, "blue")]:
        idx = pred_df[pred_df["pred"] == label].index
        idx = idx.intersection(df_nav.index)
        plt.scatter(idx, df_nav.loc[idx, factor], color=color, label=f"Predicted {label}", s=30)

    plt.rcParams['font.family'] = chinese_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示为方块
    plt.title(title, fontproperties=chinese_font)
    plt.xlabel("Date", fontproperties=chinese_font)
    plt.ylabel("Cumulative NAV", fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
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
    df_ret = factor_data_reader.read_daily_factors(start, end)[[num, denom]].dropna()
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
    start="2017-11-01"
    end="2018-12-31"
    df_prob = predict_softprob(
        task="smb_tri",
        start=start,
        end=end,
        model_path="./models/smb_tri/model_2017-10-31.pkl"
    )

    df_prob.index.name = "date"
    max_labels = np.argmax(df_prob.values, axis=1)
    df_prob["pred"] = max_labels
    df_prob["confidence"] = df_prob.iloc[:, :3].max(axis=1)
    df_out = df_prob[["pred", "confidence"]]

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
    df_ret = factor_data_reader.read_daily_factors(start, end)[["SMB"]].dropna()
    df_nav = (df_ret + 1).cumprod()
    # df_nav = csi_index_data_fetcher.get_data_by_code_and_date("H11004.CSI", start, end)
    # df_nav = df_nav.set_index("date").sort_index()[["close"]].dropna()
    plot_cumulative_nav_with_predictions(
        df_nav,
        start=start,
        end=end,
        factor="SMB",
        pred_df=df_out,
        mean=5,
        title="smb 累计净值曲线(预测类别点标注)",
        save_path="./ml_results/smb_nav_plot_with_preds.png"
    )


if __name__ == '__main__':
    with app.app_context():
        # plot_cumulative_nav(start='2019-01-04', end='2025-05-13', factors=['SMB', 'HML', 'QMJ'], title='因子净值曲线', mean=60)
        # plot_nav_ratio(start='2019-01-04', end='2025-05-13', ratio_pairs=[('HML', 'QMJ'), ('SMB', 'QMJ')], title='因子净值比值曲线', mean=60)
        run_predict_and_export()
