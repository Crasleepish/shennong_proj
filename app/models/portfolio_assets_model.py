# ==============================================
# app/models/portfolio_assets_model.py
# ==============================================
from __future__ import annotations
from sqlalchemy import Column, Integer
try:
    from sqlalchemy.dialects.postgresql import JSONB as JSONType
except Exception:  # fallback（如本地用 SQLite 进行轻量跑通）
    from sqlalchemy.types import JSON as JSONType

from app.database import Base


class PortfolioAssets(Base):
    __tablename__ = "portfolio_assets"

    portfolio_id = Column(Integer, primary_key=True, autoincrement=False)
    asset_source_map = Column(JSONType, nullable=False)
    code_factors_map = Column(JSONType, nullable=False)
    view_codes = Column(JSONType, nullable=False)

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "asset_source_map": self.asset_source_map or {},
            "code_factors_map": self.code_factors_map or {},
            "view_codes": self.view_codes or [],
        }
