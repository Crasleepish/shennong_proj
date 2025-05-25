import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import create_app
from app.ml.train import *



app = create_app()

logger = logging.getLogger(__name__)

def main():
    date_list = [
        "2012-12-31", "2013-12-31", "2014-12-31",
        "2015-12-31", "2016-12-31", "2017-12-31",
        "2018-12-31", "2019-12-31", "2020-12-31"
    ]
    rolling_train(start="2008-01-01", split_dates=date_list)
    # tune_xgb_with_optuna()


if __name__ == '__main__':
    with app.app_context():
        main()
