# run.py
from app import create_app
import tushare as ts

ts.set_token('2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211')

app = create_app()

if __name__ == '__main__':
    app.run(debug=False)
