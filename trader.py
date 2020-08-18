from datetime import datetime, time, date, timedelta
import time as tm
import data_scraper
import pandas as pd
from portfolio import Portfolio
from strategy import BaseStrategy, MACDStrategy, MomentumStrategy
from broker import Broker, Commission
import threading
from pathlib import Path
import numpy as np

class Trader():
    def __init__(self, update_dailies=False, ignore_market=False, wait_open=True, max_minutes=None):
        self.trading_hours = [time(15,30,0), time(22,0,0)]
        self.uptime = timedelta()
        self.strategies = []
        self.global_watchlist = ()
        self.data_scraper = data_scraper.DataScraper()
        self.temp_global_watchdata = {}
        self.global_prev_close = {}
        self.update_dailies = update_dailies
        self.ignore_market = ignore_market
        self.wait_open = wait_open
        self.max_minutes = max_minutes
        return



    def trading_time_left(self):
        trading_end = datetime.combine(date.today(), self.trading_hours[1])
        return trading_end - datetime.now()

    def in_trading_hours(self):
        return datetime.now().time() >= self.trading_hours[0] and datetime.now().time() <= self.trading_hours[1]

    def add_strategy(self, strategy):
        self.strategies.append(strategy)

    def update_global_watchlist(self):
        full_watchlist = []
        for strategy in self.strategies:
                full_watchlist += strategy.update_watchlist()
                self.global_watchlist = set(full_watchlist)
        print(self.global_watchlist)
        if len(self.global_watchlist) > 50:
            raise ValueError("Too many symbols on the watchlist for the free Finnhub API Plan")
    
    def update_global_watchlist_from_data(self, del_symbols):
        for strategy in self.strategies:
            for symbol in del_symbols:
                if symbol in strategy.watchlist:
                    strategy.watchlist.remove(symbol)
        for symbol in del_symbols:
            self.global_watchlist.remove(symbol)

    def get_quotes(self):
        self.global_quotes = {}
        for symbol in self.global_watchlist:
            quote = self.data_scraper.quote_as_pd(symbol)
            self.global_quotes[symbol] = quote
            tm.sleep(0.02)

    def send_quotes(self):
        for symbol, quote in self.global_quotes.items():
            for strategy in self.strategies:
                if symbol in strategy.watchlist:
                    strategy.append_quote(symbol, quote)
                
    def get_minutes(self):
        minute_data = self.data_scraper.get_today_minutes()
        now = datetime.now()
        get_minute = (now - timedelta(hours=2, minutes=1, seconds=now.second, microseconds=now.microsecond))
        for symbol in self.global_watchlist:
            if minute_data[symbol].size > 0 and get_minute in minute_data[symbol].index.to_pydatetime():
                self.global_watchdata[symbol] = minute_data[symbol].loc[get_minute, :]
            else:
                self.global_watchdata[symbol] = pd.Series(name=get_minute, dtype='float64')

    def send_minutes(self):
        for symbol, data in self.global_watchdata.items():
            for strategy in self.strategies:
                if symbol in strategy.watchlist:
                    strategy.append_minute(symbol, data)

    def send_yesterday_minutes(self, prev_closes=False):
        for symbol, data in self.global_watchdata.items():
            for strategy in self.strategies:
                if symbol in strategy.watchlist:
                    if prev_closes:
                        strategy.append_yesterday_minutes(symbol, data, prev_c=self.global_prev_close[symbol])
                    else:
                        strategy.append_yesterday_minutes(symbol, data)

    def calculations(self):
        for strategy in self.strategies:
            strategy.step()

    def end(self):
        for strategy in self.strategies:
            strategy.end()


    def run(self):
        self.starttime = datetime.now()

        if self.update_dailies:
            self.data_scraper.updateTrading212Dailies()

        self.update_global_watchlist()
        self.global_watchdata, no_data_symbols, self.global_prev_close = self.data_scraper.get_yesterday_minutes(self.global_watchlist)
        self.update_global_watchlist_from_data(no_data_symbols)
        self.data_scraper.watchlist = self.global_watchlist
        self.send_yesterday_minutes(prev_closes=True)

        if self.wait_open and not self.in_trading_hours():
            while not self.in_trading_hours():
                tm.sleep(10)

        self.data_scraper.start_ws_thread()

        conn_timeout = 20
        while not self.data_scraper.ws.sock.connected and conn_timeout:
            print("Connecting Websocket ...")
            tm.sleep(1)
            conn_timeout -= 1

        if not self.data_scraper.ws.sock.connected:
            print("Connection to Finnhub Socket failed!!!")

        self.starttime = datetime.now()
        while (self.in_trading_hours() or self.ignore_market):
            tm.sleep((65.0 - (datetime.now().second+1)) % 60) # poll data 5 secs after full minute
            print("tick, thread: ", threading.current_thread().ident)
            self.get_minutes()
            self.send_minutes()

            self.calculations()

            self.uptime =(datetime.now() - self.starttime)
            if self.max_minutes and (self.uptime > timedelta(minutes=self.max_minutes)):
                break
            # self.data_scraper.start_ws_thread()
            print("Threads in main: ", threading.active_count())
            self.data_scraper.keep_websocket_running()
            tm.sleep(1)

        self.end()
        self.data_scraper.ws.close()


class SimTrader(Trader):
    def __init__(self, sim_date, **kwargs):
        super(SimTrader, self).__init__(**kwargs)
        self.day = sim_date
    

    def test(self):
        path = Path('./hist_data/ACTG_Daily.csv')
        data = pd.read_csv(str(path), index_col="Time", parse_dates=True)
        print(np.where(data.index.date == self.day))
        index = np.where(data.index.date == self.day)[0][0]
        print(index)
        print(data.Close.iloc[index])




commission_tr = Commission(fixed=1.17)
broker1 = Broker(commission=commission_tr, send_note=False)
broker2 = Broker(commission=commission_tr)

strats = []
strats.append(MACDStrategy(broker=broker1, portfolio=Portfolio(), name="macd-26-12-9"))
# strats.append(MomentumStrategy(broker=broker2, portfolio=Portfolio(), name="mom"))


# trader = Trader(update_dailies=True, ignore_market=False, wait_open=True, max_minutes=None)
trader = SimTrader(sim_date=date(2020,8, 8))

for strat in strats:
    trader.add_strategy(strat)

trader.test()