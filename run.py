# run.py
from app import create_app
import tushare as ts
from api_key import TUSHARE_API_KEY

ts.set_token(TUSHARE_API_KEY)

app = create_app()

if __name__ == '__main__':
    app.run(debug=False)
