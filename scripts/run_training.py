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


if __name__ == '__main__':
    with app.app_context():
        main()
