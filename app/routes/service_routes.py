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