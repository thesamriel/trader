from pathlib import Path
import pandas as pd
import numpy as np
from portfolio import Portfolio, Order, OrderStatus
from broker import Broker, Commission
from datetime import date, datetime, time
import bs4
import requests
from bs4 import BeautifulSoup


class BaseStrategy():
    def __init__(self, broker=None, portfolio=None, default_size=10, default_stop=0.96, name=""):
        self.watchlist = []
        self.watchdata = {}
        self.prev_close = {}
        self.positions = {}
        self.default_size = default_size
        self.step_count = 0
        self.portfolio = portfolio
        self.broker = broker
        self.open_buys = []
        self.open_sells = []
        self.default_stop = default_stop
        self.name = name

    def append_minute(self, symbol, data):
        """ Append the Price data gathered for the last full minute """
        self.watchdata[symbol] = self.watchdata[symbol].append(data)
        if pd.isna(self.watchdata[symbol].iat[-1, self.watchdata[symbol].columns.get_loc('Volume')]):
            self.watchdata[symbol].iat[-1, self.watchdata[symbol].columns.get_loc('Price')] = self.watchdata[symbol]['Price'].iloc[-2]
            self.watchdata[symbol].iat[-1, self.watchdata[symbol].columns.get_loc('Volume')] = 0        

        self.calculate_indicators(symbol)

    def append_yesterday_minutes(self, symbol, data, prev_c=None):
        """ Start the tick data with the intraday data from yesterday """
        self.watchdata[symbol] = data
        # Save the close price of previous day
        if prev_c:
            self.prev_close[symbol] = prev_c

    def update_watchlist(self, screener_url, day=None, sim=False):
        """ Set the watchlist for the day from a yahoo screener"""
        page = 0
        stock_count = 1
        if len(self.watchlist) < 1:
            # Decide watchlist from yesterdays data
            paths = Path('./hist_data').glob('**/*.csv')
            for path in paths:
                data = pd.read_csv(str(path), index_col="Time", parse_dates=True)
                if sim:
                    try:
                        date_index = np.where(data.index.date == day.date())[0][0]
                    except IndexError:
                        continue
                else:
                    date_index = -1

                if (data['Close'].iloc[date_index] > 1.5 and data['Close'].iloc[date_index] < 25.0 
                        and data['Volume'].iloc[date_index] > 1000000 and data['Volume'].iloc[date_index-1] > 1000000 and data['Volume'].iloc[date_index-2] > 1000000
                        and data['Volume'].iloc[date_index] > data['Volume'].iloc[date_index-1]):
                    symbol = path.name.split('_')[0]
                    self.watchlist.append(symbol)
            print(len(self.watchlist), " possible stocks")

            # Check which symbols are also in selected yahoo screener
            if not sim:
                yahoo_symbols = []
                while stock_count > 0:
                    # URL for stock screener: US, price [1,30], volume > 1,000,000
                    url = screener_url + "?count=100&offset=" + str(page*100)
                    r = requests.get(url)
                    soup = bs4.BeautifulSoup(r.text, "lxml")
                    table = soup.select('tr[class*=simpTblRow]')
                    stock_count = len(table)
                    for row in table:
                        row_soup = bs4.BeautifulSoup(str(row), "lxml")
                        symbol = row_soup.find('a', {'class':'Fw(600) C($linkColor)'}).text
                        yahoo_symbols.append(symbol)
                    page += 1

                # combine watchlists      
                yahoo_symbols = set(yahoo_symbols)
                watchlist = yahoo_symbols.intersection(self.watchlist)
                print(len(self.watchlist), " stocks being watched.")

        return self.watchlist

    def set_portfolio(self, portfolio):
        self.portfolio = portfolio

    def steps_until_close(self):
        market_close = time(22,0,0)
        now = datetime.now().time
        delta = market_close - now
        return delta.minute - 1


    def buy(self, symbol):
        cur_price = self.watchdata[symbol]["Price"].iloc[-1]
        lossprice = self.calculate_max_loss(symbol, cur_price)        
        size = self.size_condition(cur_price, lossprice)
        targetprice = self.calculate_target(symbol, cur_price, lossprice) 
        new_order = Order(symbol=symbol, size=size, ordertype='buy', delay=self.broker.delay, lossprice=lossprice, target=targetprice)
        self.open_buys.append(new_order)

    def sell(self, symbol, all=False):
        # For now always sell the full position
        size = self.positions[symbol]['amount']
        new_order = Order(symbol=symbol, size=size, ordertype='sell', delay=self.broker.delay)
        self.open_sells.append(new_order)

    def step(self):
        """ Execute every minute after the new quotes have been appended """
        # Increment the step counter a.k.a. minutes after starting the trader
        self.step_count += 1
        self.begin_step()
        # Check whether to buy or sell a certain symbol
        for symbol in self.watchlist:
            if self.buy_condition(symbol):
                self.buy(symbol)
            if self.sell_condition(symbol):
                self.sell(symbol)

        # Execute open sells
        for order in self.open_sells:
            cur_price = self.watchdata[order.symbol]["Price"].iloc[-1]
            order = self.broker.execute_sell(self.portfolio, order, cur_price)
            if order.status.value == 3:
                self.positions[order.symbol]['amount'] -= order.size
                if self.positions[order.symbol]['amount'] <= 0:
                    del self.positions[order.symbol]
                self.portfolio.history.append({ "Time": self.watchdata[order.symbol].index.tolist()[-1],
                                                "Symbol": order.symbol,
                                                "Amount": order.size,
                                                "Price": self.watchdata[order.symbol]["Price"].iloc[-1],
                                                "Type": order.ordertype.upper(),
                                                "Status": OrderStatus(order.status).name}) 

        # Execute open buys
        for order in self.open_buys:
            cur_price = self.watchdata[order.symbol]["Price"].iloc[-1]
            order = self.broker.execute_buy(self.portfolio, order, cur_price)
            if order.status.value == 3:
                self.positions[order.symbol] = {'buy_price': cur_price,
                                                        'buy_time': self.watchdata[order.symbol].index.tolist()[-1],
                                                        'amount': order.size,
                                                        'stop_loss': order.lossprice,
                                                        'target': order.target}

                self.portfolio.history.append({ "Time": self.watchdata[order.symbol].index.tolist()[-1],
                                            "Symbol": order.symbol,
                                            "Amount": order.size,
                                            "Price": cur_price,
                                            "Type": order.ordertype.upper(),
                                            "Status": OrderStatus(order.status).name})
            elif order.status.value == 4:
                 self.portfolio.history.append({ "Time": self.watchdata[order.symbol].index.tolist()[-1],
                                            "Symbol": order.symbol,
                                            "Amount": order.size,
                                            "Price": cur_price,
                                            "Type": order.ordertype.upper(),
                                            "Status": OrderStatus(order.status).name})               

        # Remove completed or rejected orders from the open transactions
        self.open_sells = [order for order in self.open_sells if order.status.value < 3]
        self.open_buys = [order for order in self.open_buys if order.status.value < 3]
        self.end_step()


    def begin_step(self):
        """ (Changable) Additional Things to do at the beginning of a step """
        return
    
    def end_step(self):
        """ (Changable) Additional Things to do at the end of a step """
        return

    def calculate_indicators(self, symbol):
        """ (Changable) Add technical indicator columns needed for the strategy """
        return

    def calculate_max_loss(self, symbol, cur_price):
        """ (Changable) Determine the minimum price to sell the stock """
        return cur_price * self.default_stop
    
    def calculate_target(self, symbol, cur_price, lossprice):
        """ (Changable) Determine the target price for selling the stock """
        target = cur_price
        return target

    def buy_condition(self, symbol):
        """ (Changable) Determine when to buy stock """
        return False

    def sell_condition(self, symbol):
        """ (Changable) Determine when to sell stock """
        return False

    def size_condition(self, cur_price, loss_price):
        """ (Changable) Determine and return the amount of shares to be bought in one order """
        return self.default_size

    def end(self, day=None):
        """ Execute at the end of the day or end of trading time """
        day = day.date() if day else date.today()
        pd.DataFrame(self.portfolio.history).to_csv("portfolio_{}_{}.csv".format(day.isoformat(), self.name))
        print("Final cash in portfolio: ", self.portfolio.cash, " €")
        for symbol in set(pd.DataFrame(self.portfolio.history)['Symbol']):
            self.watchdata[symbol].to_csv("traded_symbols/{}_{}.csv".format(day, symbol))
        
        
    def append_quote(self, symbol, quote):
        """ (Deprecated) Used when course is determined by quote polling """
        if symbol in self.watchdata:
            self.watchdata[symbol] = self.watchdata[symbol].append(quote)
        else:
            self.watchdata[symbol] = quote



class MACDStrategy(BaseStrategy):
    def __init__(self, spans=(26, 12, 9), risk=0.01, **kwargs):
        super(MACDStrategy, self).__init__(**kwargs)
        self.risk = risk
        self.spans = spans


    def calculate_indicators(self, symbol):
        """ (Changable) Add technical indicator columns needed for the strategy """
        ema_long = self.watchdata[symbol].Price.ewm(span=self.spans[0], adjust=False).mean()
        ema_short = self.watchdata[symbol].Price.ewm(span=self.spans[1], adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=self.spans[2], adjust=False).mean()
        sma50 = self.watchdata[symbol].Price.rolling(50).mean()

        self.watchdata[symbol]['MACD'] = macd
        self.watchdata[symbol]['Signal'] = signal
        self.watchdata[symbol]['MACD-Dif'] = macd - signal
        self.watchdata[symbol]['SMA50'] = sma50

    def calculate_max_loss(self, symbol, cur_price):
        """ (Changable) Determine the minimum price to sell the stock """
        return cur_price * self.default_stop
    
    def calculate_target(self, symbol, cur_price, lossprice):
        """ (Changable) Determine the target price for selling the stock """
        target = cur_price + (cur_price - lossprice) * 2
        return target

    def buy_condition(self, symbol):
        """ (Changable) Determine when to buy stock """
        if self.step_count > 15 and self.step_count < 80 and symbol not in self.positions:
            if self.watchdata[symbol]['SMA50'].iloc[-1] > self.watchdata[symbol]['SMA50'].iloc[-2]:
                if self.watchdata[symbol]['MACD-Dif'].iloc[-1] > 0 and self.watchdata[symbol]['MACD-Dif'].iloc[-2] < 0:
                    return True
        return False

    def sell_condition(self, symbol):
        """ (Changable) Determine when to sell stock """
        cur_price = self.watchdata[symbol]['Price'].iloc[-1]
        if symbol in self.positions:
            if (self.positions[symbol]['target'] <= cur_price or 
                self.positions[symbol]['stop_loss'] >= cur_price or
                self.watchdata[symbol]['SMA50'].iloc[-1] < self.watchdata[symbol]['SMA50'].iloc[-2] or
                (self.watchdata[symbol]['MACD-Dif'].iloc[-1] < 0 and self.watchdata[symbol]['MACD-Dif'].iloc[-2] > 0)):
                return True

        return False

    def size_condition(self, cur_price, loss_price):
        """ (Changable) Determine and return the amount of shares to be bought in one order """
        if self.risk:
            size  = (self.portfolio.cash * self.risk) // (cur_price - loss_price)
            if size == 0: size += 1

            return min(size, 400//cur_price)
        return self.default_size

    def end_step(self):
        if self.steps_until_close in [150, 149, 148]:
            for symbol in self.positions:
                self.positions[symbol]['target'] = self.positions[symbol]['buy_price'] * 1.01
        if self.steps_until_close in [80, 79, 78]:
            for symbol in self.positions:
                self.positions[symbol]['target'] = self.positions[symbol]['buy_price']
        if self.steps_until_close in [5, 4, 3]:
            for symbol in self.positions:
                self.positions[symbol]['target'] = self.positions[symbol]['stop_loss']           



class MomentumStrategy(BaseStrategy):
    def __init__(self, risk=0.01, **kwargs):
        super(MomentumStrategy, self).__init__(**kwargs)
        self.risk = risk
        self.highs15 = {}
        self.volume_today = {}


    def calculate_indicators(self, symbol):
        """ (Changable) Add technical indicator columns needed for the strategy """
        ema_long = self.watchdata[symbol].Price.ewm(span=26, adjust=False).mean()
        ema_short = self.watchdata[symbol].Price.ewm(span=12, adjust=False).mean()
        macd1 = ema_short - ema_long
        ema_long = self.watchdata[symbol].Price.ewm(span=60, adjust=False).mean()
        ema_short = self.watchdata[symbol].Price.ewm(span=40, adjust=False).mean()
        macd2 = ema_short - ema_long

        self.watchdata[symbol]['MACDshort'] = macd1
        self.watchdata[symbol]['MACDlong'] = macd2


    def calculate_max_loss(self, symbol, cur_price):
        """ (Changable) Determine the minimum price to sell the stock """
        minute_hist = self.watchdata[symbol]['Price'].to_numpy()
        minute_hist = minute_hist[-(min(self.step_count, len(minute_hist))):]
        diffs = np.diff(minute_hist)
        low_index = np.where((diffs[:-1]<=0) & (diffs[1:]>0))[0] + 1
        if len(low_index) > 0:
            return minute_hist[low_index[-1]] - 0.01
        return cur_price * self.default_stop
    
    def calculate_target(self, symbol, cur_price, lossprice):
        """ (Changable) Determine the target price for selling the stock """
        target = cur_price + (cur_price - lossprice) * 2.5
        return target

    def buy_condition(self, symbol):
        """ (Changable) Determine when to buy stock """
        if self.step_count > 15 and self.step_count < 80 and symbol not in self.positions:
            if (self.watchdata[symbol]['Price'].iloc[-1] > self.prev_close[symbol] * 1.04 and 
                self.watchdata[symbol]['Price'].iloc[-1] > self.highs15[symbol] and
                self.volume_today[symbol] > 20000):

                if (self.watchdata[symbol]['MACDshort'].iloc[-1] < 0 or not 
                    (self.watchdata[symbol]['MACDshort'].iloc[-3] < self.watchdata[symbol]['MACDshort'].iloc[-2] < self.watchdata[symbol]['MACDshort'].iloc[-1])):
                    return False
                if (self.watchdata[symbol]['MACDlong'].iloc[-1] < 0 or 
                    (self.watchdata[symbol]['MACDlong'].iloc[-1] < self.watchdata[symbol]['MACDlong'].iloc[-2])):
                    return False

                return True
        return False

    def sell_condition(self, symbol):
        """ (Changable) Determine when to sell stock """
        cur_price = self.watchdata[symbol]['Price'].iloc[-1]
        if symbol in self.positions:
            if (self.positions[symbol]['target'] <= cur_price or 
                self.positions[symbol]['stop_loss'] >= cur_price ):
                return True
        return False

    def size_condition(self, cur_price, loss_price):
        """ (Changable) Determine and return the amount of shares to be bought in one order """
        if self.risk:
            size  = (self.portfolio.cash * self.risk) // (cur_price - loss_price)
            if size == 0: size += 1

            return min(size, 400//cur_price)
        return self.default_size

    def end_step(self):
        if self.step_count <= 15:
            for symbol in self.watchlist:
                self.highs15[symbol] = max(self.highs15.get(symbol, 0), self.watchdata[symbol]['Price'].iloc[-1])
        if self.steps_until_close in [150, 149, 148]:
            for symbol in self.positions:
                self.positions[symbol]['target'] = self.positions[symbol]['buy_price'] * 1.01
        if self.steps_until_close in [80, 79, 78]:
            for symbol in self.positions:
                self.positions[symbol]['target'] = self.positions[symbol]['buy_price']
        if self.steps_until_close in [5, 4, 3]:
            for symbol in self.positions:
                self.positions[symbol]['target'] = self.positions[symbol]['stop_loss']
        for symbol in self.watchlist:
            self.volume_today[symbol] = self.volume_today.get(symbol, 0) + self.watchdata[symbol]['Volume'].iloc[-1]