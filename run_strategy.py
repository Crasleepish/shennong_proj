from app import create_app
from app.backtest.value_strategy import backtest_strategy as strategy_1
from app.backtest.portofolio_BM_S_L import backtest_strategy as portofolio_BM_S_L
from app.backtest.portofolio_BM_B_L import backtest_strategy as portofolio_BM_B_L
from app.backtest.portofolio_BM_S_M import backtest_strategy as portofolio_BM_S_M
from app.backtest.portofolio_BM_B_M import backtest_strategy as portofolio_BM_B_M
from app.backtest.portofolio_BM_S_H import backtest_strategy as portofolio_BM_S_H
from app.backtest.portofolio_BM_B_H import backtest_strategy as portofolio_BM_B_H
from app.backtest.portofolio_ALL import backtest_strategy as portofolio_ALL

app = create_app()

functions = {
    0: strategy_1,
    1: portofolio_BM_S_L,
    2: portofolio_BM_B_L,
    3: portofolio_BM_S_M,
    4: portofolio_BM_B_M,
    5: portofolio_BM_S_H,
    6: portofolio_BM_B_H,
    7: portofolio_ALL
}

start_date = "2004-12-31"
end_date = "2025-03-18"

def list_functions():
    """列出所有可用的函数及其序号"""
    print("可用的函数列表：")
    for idx, func in functions.items():
        print(f"{idx}: {func.__module__}.{func.__name__}")

def execute_function(choice):
    """根据用户输入的序号执行对应的函数"""
    if choice in functions:
        print(f"正在执行函数: {functions[choice].__module__}.{functions[choice].__name__}")
        functions[choice](start_date=start_date, end_date=end_date)  # 执行对应的函数
    else:
        print("无效的序号，请重新选择。")

if __name__ == '__main__':
    # 列出所有函数
    list_functions()
    with app.app_context():
        # 获取用户输入
        try:
            choice = int(input("请输入要执行的函数序号: "))
            execute_function(choice)
        except ValueError:
            print("输入无效，请输入一个数字。")