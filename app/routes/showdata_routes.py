# app/routes/showdata_routes.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from flask import Blueprint, request, jsonify
import logging
import threading
import time
from datetime import datetime
import numpy as np
import pandas as pd
from app.ml.inference import get_last_day_of_prev_month
from app.ml.train_pipeline import run_all_models
import os

logger = logging.getLogger(__name__)

showdata_bp = Blueprint("showdata_routes", __name__, url_prefix="/showdata")

# -----------------------------
# 依赖：使用你项目中的模块
# -----------------------------
from app.ml.inference import (
    TASK_MODEL_PATHS,
    load_model_and_features,
    predict_softprob,
)
from app.ml.dataset_builder import DatasetBuilder
from app.data_fetcher.factor_data_reader import FactorDataReader
from app.data_fetcher.csi_index_data_fetcher import CSIIndexDataFetcher
from app.data_fetcher.gold_data_fetcher import GoldDataFetcher

# 这几个函数按你的实时主流程存在于 app.service.portfolio_opt 或相邻模块
from app.service.portfolio_opt import (
    fetch_realtime_market_data,
    compute_portfolio_returns,
    calculate_intraday_factors,
    estimate_intraday_index_value,
)


# =============================
# 内存中的盘中快照（进程级）
# =============================
_INTRADAY_STATE = {
    "additional_factor_df": None,  # pd.DataFrame: index=date, columns=因子列（MKT、SMB、HML、QMJ等）仅当前交易日一行
    "additional_map": None,        # dict[str, pd.DataFrame]: {"H11004.CSI": df, "Au99.99.SGE": df}
    "updated_at": None,            # float: time.time()
}
_INTRADAY_LOCK = threading.Lock()


# =============================
# 工具函数
# =============================

def _parse_mode() -> str:
    """
    mode: 'daily' | 'with_intraday' ；默认 daily
    """
    mode = (request.args.get("mode") or request.json.get("mode") if request.is_json else None) or "daily"
    mode = str(mode).strip().lower()
    if mode not in ("daily", "with_intraday"):
        mode = "daily"
    return mode


def _get_builder_for_mode(mode: str) -> DatasetBuilder:
    """
    根据模式构造 DatasetBuilder，确保在 with_intraday 时注入内存盘中快照。
    """
    if mode == "with_intraday":
        with _INTRADAY_LOCK:
            add_df = _INTRADAY_STATE["additional_factor_df"]
            add_map = _INTRADAY_STATE["additional_map"]

        if add_df is None or add_map is None:
            # 未点击“更新盘中数据”就请求 with_intraday → 打日志并降级为 daily
            logger.warning("with_intraday 模式请求，但盘中快照为空，自动降级为 daily。")
            return DatasetBuilder()  # 无注入
        return DatasetBuilder(additional_factor_df=add_df, additional_map=add_map)

    # daily
    return DatasetBuilder()


def _normalize_nav_from_close(close: pd.Series) -> pd.Series:
    """
    将价格序列标准化为累计净值（首日=1.0）。
    """
    close = close.dropna()
    if close.empty:
        return close
    base = float(close.iloc[0])
    if base == 0 or not np.isfinite(base):
        return pd.Series(index=close.index, dtype=float)
    return close / base


def _ensure_date_range(start: str | None, end: str | None) -> tuple[str, str]:
    """
    允许前端不传时间范围，给一个默认窗（例如最近两年）；你也可以替换为项目中现有的统一工具。
    """
    if start and end:
        return start, end
    # 可根据你项目习惯调整默认
    today = datetime.today().strftime("%Y-%m-%d")
    two_years_ago = (pd.Timestamp(today) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    return (start or two_years_ago), (end or today)


def _df_to_kv_list(series: pd.Series, value_key: str = "nav") -> list[dict]:
    """
    将时间序列转为 [{date:'YYYY-MM-DD', <value_key>:float}, ...] 便于前端绘图。
    """
    out = []
    for dt, val in series.items():
        # 兼容 date / Timestamp
        if isinstance(dt, (pd.Timestamp, datetime)):
            dstr = dt.strftime("%Y-%m-%d")
        else:
            dstr = pd.to_datetime(dt).strftime("%Y-%m-%d")
        out.append({"date": dstr, value_key: None if pd.isna(val) else float(val)})
    return out


def _pack_intraday_meta() -> dict:
    with _INTRADAY_LOCK:
        ts = _INTRADAY_STATE["updated_at"]
    return {
        "has_intraday": bool(ts),
        "intraday_updated_at": None if not ts else datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
    }


# =============================
# 接口 1：刷新盘中快照
# =============================
@showdata_bp.route("/intraday/update", methods=["POST"])
def update_intraday_snapshot():
    """
    点击“更新盘中数据”按钮时调用：
    - 拉取当前盘中行情 → 计算盘中因子与估算指数 → 生成 additional_factor_df / additional_map
    - 写入进程内存，供 with_intraday 模式的查询使用
    """
    try:
        # Step 1: 拉取盘中行情与组合收益
        stock_rt, index_rt = fetch_realtime_market_data()
        portfolio_returns = compute_portfolio_returns(stock_rt)

        # Step 2: 计算盘中因子
        intraday_factors = calculate_intraday_factors(portfolio_returns, index_rt)

        # Step 3: 指数盘中估值映射（10年国债 & 黄金）
        index_to_etf = {
            "H11001.CSI": "511010.SH",  # 中证综合债 → 国债ETF（如有需要）
            "H11004.CSI": "511260.SH",  # 中证10债 → 十年国债ETF
            "Au99.99.SGE": "518880.SH", # 黄金 → 黄金ETF
        }
        est_index_values = estimate_intraday_index_value(index_to_etf)

        # Step 4: 组装 additional_*
        from app.service.portfolio_opt import build_real_time_date
        additional_factor_df, additional_map = build_real_time_date(intraday_factors, est_index_values)

        with _INTRADAY_LOCK:
            _INTRADAY_STATE["additional_factor_df"] = additional_factor_df
            _INTRADAY_STATE["additional_map"] = additional_map
            _INTRADAY_STATE["updated_at"] = time.time()

        meta = _pack_intraday_meta()
        return jsonify({"ok": True, "message": "盘中快照已更新", "meta": meta})

    except Exception as e:
        logger.exception("更新盘中数据失败")
        return jsonify({"ok": False, "error": str(e)}), 500


# =============================
# 接口 2：盘中快照状态查询（可选）
# =============================
@showdata_bp.route("/intraday/status", methods=["GET"])
def intraday_status():
    return jsonify({"ok": True, "meta": _pack_intraday_meta()})


# =============================
# 接口 3：历史累计净值数据
# =============================
@showdata_bp.route("/nav", methods=["GET"])
def get_nav_series():
    """
    返回：四因子 + H11004.CSI + Au99.99.SGE 的累计净值曲线
    Query:
      - start: YYYY-MM-DD
      - end:   YYYY-MM-DD
      - mode:  'daily' | 'with_intraday' （默认 daily）
    响应：
      {
        ok: true,
        series: {
          MKT: [{date, nav}, ...],
          SMB: [...],
          HML: [...],
          QMJ: [...],
          BOND10Y: [...],
          GOLD: [...]
        },
        meta: { mode, has_intraday, intraday_updated_at }
      }
    """
    try:
        start = request.args.get("start")
        end = request.args.get("end")
        mode = _parse_mode()
        start, end = _ensure_date_range(start, end)

        builder = _get_builder_for_mode(mode)

        # ---- 四因子 NAV（通常 read_factor_nav 会返回 *_NAV 或收益可累乘）----
        factor_reader = FactorDataReader(additional_df=getattr(builder, "factor_data_reader", None) and builder.factor_data_reader.additional_df)
        df_factor = factor_reader.read_factor_nav(start, end)  # 预期含: MKT_NAV, SMB_NAV, HML_NAV, QMJ_NAV
        series = {}

        # 如果已经是净值就直接标准化首日为1（防止不同起点不一致）
        for col, key in [("MKT_NAV", "MKT"), ("SMB_NAV", "SMB"), ("HML_NAV", "HML"), ("QMJ_NAV", "QMJ")]:
            if col in df_factor.columns:
                nav = _normalize_nav_from_close(df_factor[col].sort_index())
                series[key] = _df_to_kv_list(nav)

        # ---- 10Y 国债（H11004.CSI）----
        csi_fetcher = CSIIndexDataFetcher(additional_map=getattr(builder, "csi_index_data_fetcher", None) and builder.csi_index_data_fetcher.additional_map)
        df_bond = csi_fetcher.get_data_by_code_and_date("H11004.CSI", start, end)
        if df_bond is not None and not df_bond.empty:
            df_bond = df_bond.set_index("date").sort_index()
            nav_bond = _normalize_nav_from_close(df_bond["close"])
            series["BOND10Y"] = _df_to_kv_list(nav_bond)

        # ---- 黄金（Au99.99.SGE）----
        gold_fetcher = GoldDataFetcher(additional_map=getattr(builder, "gold_data_fetcher", None) and builder.gold_data_fetcher.additional_map)
        df_gold = gold_fetcher.get_data_by_code_and_date("Au99.99.SGE", start, end)
        if df_gold is not None and not df_gold.empty:
            df_gold = df_gold.set_index("date").sort_index()
            nav_gold = _normalize_nav_from_close(df_gold["close"])
            # 为了 key 不含点号，统一下划线
            series["GOLD"] = _df_to_kv_list(nav_gold)

        resp = {
            "ok": True,
            "series": series,
            "meta": {"mode": mode, **_pack_intraday_meta(), "start": start, "end": end},
        }
        return jsonify(resp)

    except Exception as e:
        logger.exception("拉取净值数据失败")
        return jsonify({"ok": False, "error": str(e)}), 500


# =============================
# 接口 4：预测 softprob 序列
# =============================
@showdata_bp.route("/predict", methods=["GET"])
def get_predict_series():
    """
    返回：四因子 + 10Y 国债 + 黄金的三分类 softprob 概率曲线（按日）
    Query:
      - start: YYYY-MM-DD
      - end:   YYYY-MM-DD
      - mode:  'daily' | 'with_intraday' （默认 daily）
    响应：
      {
        ok: true,
        probs: {
          MKT:  [{date, prob0, prob1, prob2}, ...],
          SMB:  [...],
          HML:  [...],
          QMJ:  [...],
          BOND10Y or H11004_CSI: [...],
          GOLD: [...]
        },
        meta: { mode, has_intraday, intraday_updated_at }
      }
    """
    try:
        start = request.args.get("start")
        end = request.args.get("end")
        mode = _parse_mode()
        start, end = _ensure_date_range(start, end)

        builder = _get_builder_for_mode(mode)

        # 任务映射（任务名 -> 展示 key）
        # 你可按项目规范替换为 inference.py 中已有的映射
        TASK_TO_KEY = {
            "mkt_tri": "MKT",
            "smb_tri": "SMB",
            "hml_tri": "HML",
            "qmj_tri": "QMJ",
            "10Ybond_tri": "BOND10Y",
            "gold_tri": "GOLD",
        }

        out = {}
        for task, path_tpl in TASK_MODEL_PATHS.items():
            if task not in TASK_TO_KEY:
                continue
            # === 用“上个月最后一天”的日期作为 model_date ===
            model_date = get_last_day_of_prev_month(
                datetime.strptime(end, "%Y-%m-%d")
            ).strftime("%Y-%m-%d")

            model_path = path_tpl.format(model_date)

            # 若模型不存在，则触发一次训练（与你在 get_softprob_dict 中的回退一致）
            if not os.path.exists(model_path):
                logger.warning(f"[predict] 模型不存在: {model_path}，开始触发训练...")
                # 全量跑一次，覆盖 mkt/smb/hml/qmj/10Ybond/gold
                run_all_models(start="2007-12-01", split_date=None, end=model_date, need_test=False)
                # 训练完再定位一次模型路径
                if not os.path.exists(model_path):
                    # 仍不存在就报错，便于定位模型命名/路径模板问题
                    raise FileNotFoundError(
                        f"训练后仍找不到模型文件: {model_path}（task={task}, model_date={model_date}）"
                    )

            # 直接使用 predict_softprob（你贴出的版本支持 dataset_builder 注入 & 返回 DataFrame）
            df_proba = predict_softprob(
                task=task,
                start=start,
                end=end,
                model_path=model_path,
                dataset_builder=builder,
            )
            if df_proba is None or df_proba.empty:
                continue

            # 软概率合法性护栏（冗余但安全）
            proba = df_proba.values
            proba = np.clip(proba, 0, None)
            row_sums = proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            proba = proba / row_sums

            df_norm = pd.DataFrame(proba, index=df_proba.index, columns=df_proba.columns)
            # 组装 [{date, prob0, prob1, prob2}, ...]
            key = TASK_TO_KEY[task]
            rows = []
            for dt, row in df_norm.iterrows():
                if isinstance(dt, (pd.Timestamp, datetime)):
                    dstr = dt.strftime("%Y-%m-%d")
                else:
                    dstr = pd.to_datetime(dt).strftime("%Y-%m-%d")
                row_dict = {"date": dstr}
                # 统一字段名：prob0/1/2（不带任务前缀，便于前端复用组件）
                # 如果你更喜欢保留原列名，也可直接透传
                for i in range(row.shape[0]):
                    row_dict[f"prob{i}"] = float(row.iloc[i]) if np.isfinite(row.iloc[i]) else None
                rows.append(row_dict)
            out[key] = rows

        resp = {
            "ok": True,
            "probs": out,
            "meta": {"mode": mode, **_pack_intraday_meta(), "start": start, "end": end},
        }
        return jsonify(resp)

    except Exception as e:
        logger.exception("获取预测 softprob 失败")
        return jsonify({"ok": False, "error": str(e)}), 500


def get_default_date_range_if_missing(start: str | None, end: str | None) -> tuple[str, str]:
        return _ensure_date_range(start, end)