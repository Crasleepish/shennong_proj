from numba import njit
from vectorbt.portfolio.enums import Direction, OrderStatus, NoOrder, CallSeqType 
from vectorbt.portfolio import nb
import vectorbt as vbt
import pandas as pd



def test_vectorbt():

    @njit
    def order_func_nb(c, size):
        print("==========c.i===========:")
        print(c.i)
        print("==========c.col===========:")
        print(c.col)
        return nb.order_nb(size=size)

    close = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
    pf = vbt.Portfolio.from_order_func(close, order_func_nb, 10)
    print(pf.orders.records_readable)

    pf.assets()
    # 0    10.0
    # 1    20.0
    # 2    30.0
    # 3    40.0
    # 4    40.0

    pf.cash()
    # 0    90.0
    # 1    70.0
    # 2    40.0
    # 3     0.0
    # 4     0.0