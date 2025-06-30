# run.py
from app import create_app
import tushare as ts

ts.set_token('')

app = create_app()

if __name__ == '__main__':
    app.run(debug=False)
