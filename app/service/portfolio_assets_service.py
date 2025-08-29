# ==============================================
# app/service/portfolio_assets_service.py
# ==============================================
from __future__ import annotations
from typing import Optional, Dict, List, Any
from app.database import get_db
from app.models.portfolio_assets_model import PortfolioAssets
import logging

log = logging.getLogger(__name__)


def upsert_portfolio_assets(
    portfolio_id: int,
    *,
    asset_source_map: Dict[str, str],
    code_factors_map: Dict[str, List[str]],
    view_codes: List[str],
    params: Optional[Dict[str, Any]] = None,   # ← 新增
) -> dict:
    """不存在则插入，存在则整体更新。"""

    # 轻量类型校验（保持原有风格）
    if not isinstance(asset_source_map, dict):
        raise ValueError("asset_source_map 必须是 dict")
    if not isinstance(code_factors_map, dict):
        raise ValueError("code_factors_map 必须是 dict")
    if not isinstance(view_codes, list):
        raise ValueError("view_codes 必须是 list")
    if params is not None and not isinstance(params, dict):
        raise ValueError("params 必须是 dict 或 None")

    # 统一 key/值
    asset_source_map = {str(k): str(v) for k, v in asset_source_map.items()}
    code_factors_map = {str(k): [str(x) for x in v] for k, v in code_factors_map.items()}
    view_codes = [str(x) for x in view_codes]
    # params 不强制转换 value，保留原 JSON 结构，仅规范 key
    if params is not None:
        params = {str(k): str(v) for k, v in params.items()}

    with get_db() as db:
        obj = db.query(PortfolioAssets).get(portfolio_id)
        if obj is None:
            obj = PortfolioAssets(
                portfolio_id=portfolio_id,
                asset_source_map=asset_source_map,
                code_factors_map=code_factors_map,
                view_codes=view_codes,
                params=params,
            )
            db.add(obj)
            action = "insert"
        else:
            obj.asset_source_map = asset_source_map
            obj.code_factors_map = code_factors_map
            obj.view_codes = view_codes
            obj.params = params
            action = "update"

        db.commit()
        db.refresh(obj)
        log.info("portfolio_assets %s ok: portfolio_id=%s", action, portfolio_id)
        return obj.to_dict()

def get_portfolio_assets(portfolio_id: int) -> Optional[dict]:
    """按 id 查询，返回字典（包含 params）。"""
    with get_db() as db:
        obj = db.query(PortfolioAssets).get(portfolio_id)
        return obj.to_dict() if obj else None