from flask import Blueprint, request, jsonify
import logging

logger = logging.getLogger(__name__)

service_bp = Blueprint("service_routes", __name__, url_prefix="/service")

from app.service.portfolio_opt import optimize_portfolio_realtime, optimize_portfolio_history

#
@service_bp.route("/portfolio_opt_hist", methods=["POST"])
def portfolio_opt_hist():
    """
    Optimize portfolio ;history
    """
    data = request.get_json()
    start_date = data.get("start_date", None)
    end_date = data.get("end_date", None)
    portfolio_id = int(data.get("portfolio_id"))
    try:
        optimize_portfolio_history(portfolio_id, start_date, end_date)
        return jsonify({"message": "success"})
    except Exception as e:
        return jsonify({"message": "error", "data": str(e)})

@service_bp.route("/portfolio_opt_rt", methods=["POST"])
def portfolio_opt_rt():
    """
    Optimize portfolio realtime
    """
    data = request.get_json()
    portfolio_id = int(data.get("portfolio_id"))
    try:
        portfolio_plan = optimize_portfolio_realtime(portfolio_id)
        return jsonify({"message": "success", "data": portfolio_plan})
    except Exception as e:
        logger.error(e)
        return jsonify({"message": "error", "data": str(e)})
    
from app.service.portfolio_crud import query_weights_by_date

@service_bp.route("/weights/<int:portfolio_id>", methods=["GET"])
def get_weights_by_date(portfolio_id: int):
    """
    查询指定交易日的组合权重（ewma平滑后）
    """
    try:
        date = request.args.get("date")
        if not date:
            return jsonify({"message": "缺少参数: date"}), 400

        result = query_weights_by_date(date, portfolio_id)
        return jsonify(result)

    except Exception as e:
        logger.error(f"查询组合权重失败: {e}")
        return jsonify({"message": "error", "data": str(e)}), 500

from app.service.portfolio_crud import query_current_position

@service_bp.route("/current_position/<int:portfolio_id>", methods=["GET"])
def get_current_position(portfolio_id: int):
    """
    查询当前持仓（包括 asset, code, name, amount, price）
    """
    try:
        result = query_current_position(portfolio_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"查询当前持仓失败: {e}")
        return jsonify({"message": "error", "data": str(e)}), 500

from app.service.portfolio_crud import generate_reallocation_ops
from flask import request


@service_bp.route("/calculate_rebalance", methods=["POST"])
def calculate_rebalance():
    """
    生成调仓建议（含数据预处理与校验）
    """
    try:
        data = request.get_json()
        merged = data.get("merged", [])

        cleaned = []
        for row in merged:
            row["amount"] = float(row.get("amount") or 0)
            row["current_percent"] = float(row.get("current_percent") or 0)
            row["target_amount"] = float(row.get("target_amount") or 0)
            row["target_percent"] = float(row.get("target_percent") or 0)
            code = row.get("code")
            price = float(row.get("price") or 0)
            row["price"] = price

            # 跳过 amount 和 target_amount 同时为 0 的行
            if row["amount"] == 0 and row["target_amount"] == 0:
                continue

            # code 不能为空，price 必须为正数
            if not code or price <= 0:
                return jsonify({"message": f"字段缺失或价格无效: {code}" }), 400

            cleaned.append(row)

        if not cleaned:
            return jsonify({"message": "无有效资产用于调仓"}), 400

        # 调用已有函数计算转换建议
        result = generate_reallocation_ops(
            codes=[r["code"] for r in cleaned],
            from_value=[r["amount"] * r["price"] for r in cleaned],
            to_value=[r["target_amount"] * r["price"] for r in cleaned],
            price=[r["price"] for r in cleaned]
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"调仓建议生成失败: {e}")
        return jsonify({"message": "error", "data": str(e)}), 500
    

from app.service.portfolio_crud import save_transfer_logs, save_current_holdings

@service_bp.route("/submit_rebalance", methods=["POST"])
def submit_rebalance():
    """
    确认调仓操作，保存调仓日志与目标持仓
    """
    try:
        payload = request.get_json()
        transfers = payload.get("rebalance", [])
        holdings = payload.get("new_holdings", [])
        portfolio_id = int(payload.get("portfolio_id"))

        if not transfers:
            return jsonify({"message": "缺少调仓内容"}), 400

        if not holdings:
            return jsonify({"message": "缺少目标持仓内容"}), 400

        save_transfer_logs(transfers)
        save_current_holdings(portfolio_id, holdings)

        return jsonify({"message": "调仓记录与持仓已保存", "status": "ok"})
    except Exception as e:
        logger.error(f"提交调仓失败: {e}")
        return jsonify({"message": "error", "data": str(e)}), 500


from app.service.portfolio_crud import query_latest_transfer

@service_bp.route("/latest_transfer", methods=["GET"])
def get_latest_transfer():
    """
    获取最近一次调仓操作
    """
    try:
        transfers = query_latest_transfer()
        return jsonify(transfers)
    except Exception as e:
        logger.error(f"获取调仓日志失败: {e}")
        return jsonify({"message": "error", "data": str(e)}), 500


from app.service.portfolio_crud import query_fund_names_by_codes

@service_bp.route("/fund_names", methods=["POST"])
def get_fund_names():
    """
    根据基金代码查询名称，POST body 为: {"codes": ["001234.OF", "518880.SH"]}
    """
    try:
        data = request.get_json()
        codes = data.get("codes", [])
        result = query_fund_names_by_codes(codes)
        return jsonify(result)
    except Exception as e:
        logger.error(f"获取基金名称失败: {e}")
        return jsonify({"message": "error", "data": str(e)}), 500
    
from flask import request, jsonify
from app.service.portfolio_crud import query_price_by_codes

@service_bp.route("/fund_prices", methods=["POST"])
def get_fund_prices():
    """
    根据基金代码列表，查询最近交易日价格
    请求格式: { "codes": ["017837.OF", "004253.OF"] }
    响应格式: { "017837.OF": 1.073, "004253.OF": 2.544 }
    """
    try:
        data = request.get_json()
        codes = data.get("codes", [])
        result = query_price_by_codes(codes)
        return jsonify(result)
    except Exception as e:
        logger.error(f"获取价格失败: {e}")
        return jsonify({"message": "error", "data": str(e)}), 500
    

from app.service.portfolio_opt import compute_diverge

@service_bp.route("/compute_diverge", methods=["POST"])
def update_dynamic_beta():
    try:
        data = request.get_json()
        portfolio_id = data.get("portfolio_id")
        trade_date = data.get("trade_date")
        current_w = data.get("current_w")
        target_w = data.get("target_w")
        diverge = compute_diverge(portfolio_id, trade_date, current_w, target_w)
        return jsonify({"message": "ok", "data": diverge})
    except Exception as e:
        logger.error(f"计算跟踪误差失败: {e}")
        return jsonify({"message": "error", "data": str(e)}), 500
    
from werkzeug.exceptions import BadRequest
from app.service.portfolio_assets_service import (
    upsert_portfolio_assets,
    get_portfolio_assets,
)

@service_bp.route("/portfolio_assets_upsert/<int:portfolio_id>", methods=["POST"])
def route_portfolio_assets_upsert(portfolio_id: int):
    """不存在则插入；存在则更新。请求体 JSON：
    {
      "asset_source_map": {"008114.OF": "factor", ...},
      "code_factors_map": {"008114.OF": ["MKT","SMB","HML","QMJ"], ...},
      "view_codes": ["008114.OF", ...],
      "params": {"post_view_tau": "0.07", "alpha": "0.1", "variance": "0.01"}
    }
    post_view_tau: black-litterman后验观点融合系数，越大代表预测观点对模型结果影响越强，通常取0.05-0.3
    alpha: 每日权重的移动平滑系数，通常取0.01-0.2
    variance: 组合优化的方差约束，表示最大能容忍的方差（注意不是标准差，是标准差的平方），通常取0.0006-0.1
    """
    body = request.get_json(silent=True) or {}
    try:
        result = upsert_portfolio_assets(
            portfolio_id=portfolio_id,
            asset_source_map=body.get("asset_source_map") or {},
            code_factors_map=body.get("code_factors_map") or {},
            view_codes=body.get("view_codes") or [],
            params=body.get("params"),
        )
    except (ValueError, BadRequest) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result), 200


@service_bp.route("/portfolio_assets_query/<int:portfolio_id>", methods=["GET"])
def route_portfolio_assets_query(portfolio_id: int):
    data = get_portfolio_assets(portfolio_id)
    if not data:
        return jsonify({"error": "portfolio_id 不存在"}), 404
    return jsonify(data), 200


from app.ml.support_asset import find_support_assets
from app.data_fetcher import CalendarFetcher
import pandas as pd

@service_bp.route("/portfolio_assets/find_support_assets", methods=["POST"])
def route_find_support_assets():
    data = request.get_json()
    asof_date = data.get("asof_date", None)
    if asof_date:
        asof_date_str = pd.to_datetime(asof_date).date().strftime("%Y%m%d")
    else:
        asof_date_str = pd.to_datetime("today").date().strftime("%Y%m%d")
    asof_trade_date_str = CalendarFetcher().get_trade_date(start="19900101", end=asof_date_str, format="%Y-%m-%d", limit=1, ascending=False)[0]
    asset_list = find_support_assets(asof_trade_date_str, epsilon=0.03, M=4096, topk_per_iter=32, debug=True)
    if not asset_list:
        return jsonify({"message": "发现支撑资产失败"}), 200
    logger.info(f"发现支持资产: {asset_list}")
    return jsonify({"message": "succuess", "data": asset_list}), 200