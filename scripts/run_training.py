import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import create_app
from app.ml.train_pipeline import *



app = create_app()

logger = logging.getLogger(__name__)

def main():
    date_list = [
        "2016-12-31", "2024-06-30"
    ]
    rolling_train(start="2007-12-01", split_dates=date_list)
    # tune_with_optuna("qmj_tri")

def train_all_models_by_month():
    dates = pd.date_range(start="2011-12-01", end="2025-09-01", freq='M').to_list()
    for d in dates:
        run_all_models(start="2007-12-01", split_date=None, end=d.strftime("%Y-%m-%d"), need_test=False)
    # run_all_models(start="2007-12-01", split_date=None, end="2020-06-30", need_test=False)


if __name__ == '__main__':
    with app.app_context():
        train_all_models_by_month()
