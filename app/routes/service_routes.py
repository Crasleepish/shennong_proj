from flask import Blueprint, request, jsonify
import logging

logger = logging.getLogger(__name__)

service_bp = Blueprint("service_routes", __name__, url_prefix="/service")

from app.service.portfolio_opt import optimize_portfolio_realtime

@service_bp.route("/portfolio_opt_rt", methods=["POST"])
def portfolio_opt_rt():
    """
    Optimize portfolio realtime
    """
    try:
        portfolio_plan = optimize_portfolio_realtime()
        return jsonify({"message": "success", "data": portfolio_plan})
    except Exception as e:
        logger.error(e)
        return jsonify({"message": "error", "data": str(e)})
    
from app.service.portfolio_crud import query_weights_by_date

@service_bp.route("/weights", methods=["GET"])
def get_weights_by_date():
    """
    查询指定交易日的组合权重（ewma平滑后）
    """
    try:
        date = request.args.get("date")
        if not date:
            return jsonify({"message": "缺少参数: date"}), 400

        result = query_weights_by_date(date)
        return jsonify(result)

    except Exception as e:
        logger.error(f"查询组合权重失败: {e}")
        return jsonify({"message": "error", "data": str(e)}), 500

from app.service.portfolio_crud import query_current_position

@service_bp.route("/current_position", methods=["GET"])
def get_current_position():
    """
    查询当前持仓（包括 asset, code, name, amount, price）
    """
    try:
        result = query_current_position()
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

        if not transfers:
            return jsonify({"message": "缺少调仓内容"}), 400

        if not holdings:
            return jsonify({"message": "缺少目标持仓内容"}), 400

        save_transfer_logs(transfers)
        save_current_holdings(holdings)

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
