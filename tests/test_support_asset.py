import pytest
import pandas as pd
from datetime import datetime
from app.dao.betas_dao import FundBetaDao
from app.database import get_db
from app.models.fund_models import FundBeta, FundInfo
from app.models.etf_model import EtfInfo
from app import create_app
from app.config import TestConfig, Config
import json
import numpy as np
from app.ml.support_asset import find_support_assets
import logging


@pytest.fixture
def app():
    app = create_app(config_class=Config)
    # Note: since our app's init_db() was already called, we assume the tables are created.
    # If necessary, you could call init_db() again here.
    yield app

def test_find_support_assets(app):
    logger = logging.getLogger(__name__)
    logger.info("Starting test_find_support_assets")
    find_support_assets('2025-08-21', epsilon=0.03, M=4096, topk_per_iter=32, debug=True)

