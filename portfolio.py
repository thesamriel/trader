import enum
import pandas as pd
# from strategy import SimpleStrategy

class Portfolio():

    def __init__(self, start_cash=1000, broker=None):
        self.init_cash = start_cash
        self.value = start_cash
        self.cash = start_cash
        self.history = []
    
    def reset(self):
        self.cash = self.init_cash
        self.history = []

    def setcash(self, value):
        self.cash = value
    
    def getcash(self):
        return self.cash
    
    def addcash(self, value):
        self.cash += value


class Order():
    def __init__(self, symbol="", size=0, ordertype=None, delay=0, lossprice=None, target=None, comment=""):
        self.size = abs(size)
        self.symbol = symbol
        self.status = OrderStatus(0)
        self.ordertype = ordertype or ("buy","sell")[size<0]
        self.limit = None
        self.stoploss = None
        self.delay = delay
        self.lossprice = lossprice
        self.target = target
        self.comment = comment


class OrderStatus(enum.Enum):
    Created = 0
    Submitted = 1
    Accepted = 2    
    Completed = 3
    Rejected = 4


########################################################
if __name__ == "__main__":
    order = Order(size=-10)
    print(order.ordertype)
    print(OrderStatus(0))