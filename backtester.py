# from __future__ import (absolute_import, division, print_function,
#                         unicode_literals)
import sys
import os
import datetime
from hist_data import AVClient, AV_API_KEY

import backtrader as bt

class TestStrategy(bt.Strategy):

    params = ( ('maperiod', 15), ('printlog', False))
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.totalgross = 0
        self.totalnet  = 0
        self.totaltrades = 0
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod)

        # Indicators for the plotting show
        """
        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25,
                                            subplot=True)
        bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])
        rsi = bt.indicators.RSI(self.datas[0])
        bt.indicators.SmoothedMovingAverage(rsi, period=10)
        bt.indicators.ATR(self.datas[0], plot=False)
        """

    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log("BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm: %.2f" % (order.executed.price, order.executed.value, order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.totaltrades += 1
            elif order.issell():
                self.log("SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm: %.2f" % (order.executed.price, order.executed.value, order.executed.comm))
                self.totaltrades += 1

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")
        
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm))
        self.totalgross += trade.pnl
        self.totalnet  += trade.pnlcomm
    
    def stop(self):
        # self.log("TOTAL GROSS: %.2f, NET: %.2f, Number of trades: %d" %(self.totalgross, self.totalnet, self.totaltrades), doprint=True)
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.maperiod, self.broker.getvalue()), doprint=True)


    def next(self):
        self.log("Close %.2f" % self.dataclose[0])

        if self.order:
            return

        if not self.position:

            if self.dataclose[0] > self.sma[0]:
                self.log("BUY CREATE, %.2f" % self.dataclose[0])
                self.order = self.buy()

        else:
            if self.dataclose[0] < self.sma[0]:
                self.log("SELL CREATE %.2f" % self.dataclose[0])

                self.order = self.sell()


cerebro = bt.Cerebro()
cerebro.addstrategy(TestStrategy, printlog=True)

avclient = AVClient(AV_API_KEY, csv=True)
ibm_data = avclient.time_series_intraday('IBM', interval=5)

print(ibm_data.info())
data = bt.feeds.PandasData(dataname=ibm_data,
                                    datetime=0,
                                    openinterest=None,
                                    # fromdate=datetime.datetime(2010,1,1), 
                                    # todate=datetime.datetime(2010,12,31),
                                    )




cerebro.adddata(data)

cerebro.broker.set_cash(10000.0)
cerebro.broker.setcommission(commission=0.1, automargin=True, commtype=bt.CommInfoBase.COMM_FIXED)
cerebro.addsizer(bt.sizers.FixedSize, stake=10)

print("Starting portfolio value: %.2f" % cerebro.broker.getvalue())

cerebro.run(maxcpus=1)

print ("Final portfolio value: %.2f" % cerebro.broker.getvalue())

