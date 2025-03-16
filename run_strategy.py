from app import create_app
from app.backtest.value_strategy import backtest_strategy

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        backtest_strategy("2004-12-31", "2024-12-31")