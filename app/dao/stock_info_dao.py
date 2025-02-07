from typing import List
from sqlalchemy.orm import Session
from app.models.stock_models import StockInfo
from app.database import get_db
import logging

logger = logging.getLogger(__name__)

class StockInfoDao:
    def __init__(self):
        pass

    def load_stock_info(self) -> List[StockInfo]:
        try:
            with get_db() as db:
                stock_info_lst = db.query(StockInfo).all()
                return stock_info_lst
        except Exception as e:
            logger.error(e)
        
    def batch_insert(self, stock_info_lst: List[StockInfo]):
        try:
            with get_db() as db:
                db.add_all(stock_info_lst)
                db.commit()
                return stock_info_lst
        except Exception as e:
            logger.error("Error during batch insert: %s", e)
            db.rollback()
            raise e
        