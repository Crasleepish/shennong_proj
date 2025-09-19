# run.py
from api_key import TUSHARE_API_KEY
import tushare as ts
ts.set_token(TUSHARE_API_KEY)

import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
