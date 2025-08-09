# run.py
from api_key import TUSHARE_API_KEY
import tushare as ts
ts.set_token(TUSHARE_API_KEY)

import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def init_if_empty(target_dir, src_dir):
    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        print(f"Initializing {target_dir} from {src_dir}")
        shutil.copytree(src_dir, target_dir, dirs_exist_ok=True)

init_if_empty(
    os.path.join(BASE_DIR, "bt_result"),
    os.path.join(BASE_DIR, "bt_result_init")
)
init_if_empty(
    os.path.join(BASE_DIR, "ml_results"),
    os.path.join(BASE_DIR, "ml_results_init")
)
init_if_empty(
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "models_init")
)
init_if_empty(
    os.path.join(BASE_DIR, "output"),
    os.path.join(BASE_DIR, "output_init")
)

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
