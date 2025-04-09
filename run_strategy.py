from app import create_app
from app.backtest.value_strategy import backtest_strategy as strategy_1
from app.backtest.portfolio_BM_S_L import backtest_strategy as portfolio_BM_S_L
from app.backtest.portfolio_BM_B_L import backtest_strategy as portfolio_BM_B_L
from app.backtest.portfolio_BM_S_M import backtest_strategy as portfolio_BM_S_M
from app.backtest.portfolio_BM_B_M import backtest_strategy as portfolio_BM_B_M
from app.backtest.portfolio_BM_S_H import backtest_strategy as portfolio_BM_S_H
from app.backtest.portfolio_BM_B_H import backtest_strategy as portfolio_BM_B_H
from app.backtest.portfolio_ALL import backtest_strategy as portfolio_ALL
from app.backtest.portfolio_OP_S_L import backtest_strategy as portfolio_OP_S_L
from app.backtest.portfolio_OP_B_L import backtest_strategy as portfolio_OP_B_L
from app.backtest.portfolio_OP_S_M import backtest_strategy as portfolio_OP_S_M
from app.backtest.portfolio_OP_B_M import backtest_strategy as portfolio_OP_B_M
from app.backtest.portfolio_OP_S_H import backtest_strategy as portfolio_OP_S_H
from app.backtest.portfolio_OP_B_H import backtest_strategy as portfolio_OP_B_H
from app.backtest.portfolio_INV_S_L import backtest_strategy as portfolio_INV_S_L
from app.backtest.portfolio_INV_B_L import backtest_strategy as portfolio_INV_B_L
from app.backtest.portfolio_INV_S_M import backtest_strategy as portfolio_INV_S_M
from app.backtest.portfolio_INV_B_M import backtest_strategy as portfolio_INV_B_M
from app.backtest.portfolio_INV_S_H import backtest_strategy as portfolio_INV_S_H
from app.backtest.portfolio_INV_B_H import backtest_strategy as portfolio_INV_B_H
from app.backtest.portfolio_MOM_S_H import backtest_strategy as portfolio_MOM_S_H
from app.backtest.portfolio_MOM_B_H import backtest_strategy as portfolio_MOM_B_H
from app.backtest.portfolio_MOM_S_L import backtest_strategy as portfolio_MOM_S_L
from app.backtest.portfolio_MOM_B_L import backtest_strategy as portfolio_MOM_B_L
from app.backtest.portfolio_MOM6_S_H import backtest_strategy as portfolio_MOM6_S_H
from app.backtest.portfolio_MOM6_S_L import backtest_strategy as portfolio_MOM6_S_L
from app.backtest.portfolio_VLT_S_L import backtest_strategy as portfolio_VLT_S_L
from app.backtest.portfolio_VLT_B_L import backtest_strategy as portfolio_VLT_B_L
from app.backtest.portfolio_VLT_S_H import backtest_strategy as portfolio_VLT_S_H
from app.backtest.portfolio_VLT_B_H import backtest_strategy as portfolio_VLT_B_H
from app.data.helper import get_index_daily_return
from app.backtest.portfolio_fund import backtest_strategy as portfolio_fund

app = create_app()

def csi_index_zzqz(start_date: str, end_date: str):
    df = get_index_daily_return("000985")
    df = df.sort_index()
    df = df.loc[start_date:end_date]
    df.to_csv(r"./result/csi_index_zzqz.csv", index=True)
    return df

functions = {
    0: portfolio_ALL,
    1: portfolio_BM_S_L,
    2: portfolio_BM_B_L,
    3: portfolio_BM_S_M,
    4: portfolio_BM_B_M,
    5: portfolio_BM_S_H,
    6: portfolio_BM_B_H,
    7: portfolio_OP_S_L, 
    8: portfolio_OP_B_L,
    9: portfolio_OP_S_M,
    10: portfolio_OP_B_M, 
    11: portfolio_OP_S_H,
    12: portfolio_OP_B_H,
    13: portfolio_INV_S_L,
    14: portfolio_INV_B_L,
    15: portfolio_INV_S_M,
    16: portfolio_INV_B_M,
    17: portfolio_INV_S_H,
    18: portfolio_INV_B_H,
    19: portfolio_MOM_S_H,
    20: portfolio_MOM_B_H,
    21: portfolio_MOM_S_L,
    22: portfolio_MOM_B_L,
    23: portfolio_MOM6_S_H,
    24: portfolio_MOM6_S_L,
    25: portfolio_VLT_S_L,
    26: portfolio_VLT_B_L,
    27: portfolio_VLT_S_H,
    28: portfolio_VLT_B_H,
    29: csi_index_zzqz,
    30: portfolio_fund
}

start_date = "2017-06-29"
end_date = "2025-03-31"

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
        except ValueError as e:
            print("输入无效，请输入一个数字。")
            print(e)