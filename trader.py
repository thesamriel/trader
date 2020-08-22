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
from keys import SCREENER_URL
from utils import progressBar

class Trader():
    def __init__(self, strategy, update_dailies=False, ignore_market=False, wait_open=True, max_minutes=None, day=None):
        self.trading_hours = [time(15,30,0), time(22,0,0)]
        self.uptime = timedelta()
        self.strategy = strategy
        self.global_watchlist = []
        self.data_scraper = data_scraper.DataScraper()
        self.temp_global_watchdata = {}
        self.global_prev_close = {}
        self.update_dailies = update_dailies
        self.ignore_market = ignore_market
        self.wait_open = wait_open
        self.max_minutes = max_minutes
        self.day = day or datetime.today() 
        return



    def trading_time_left(self):
        trading_end = datetime.combine(date.today(), self.trading_hours[1])
        return trading_end - datetime.now()

    def in_trading_hours(self):
        return datetime.now().time() >= self.trading_hours[0] and datetime.now().time() <= self.trading_hours[1]

    def set_global_watchlist(self, sim=False):
        full_watchlist = self.strategy.set_watchlist(screener_url=SCREENER_URL, sim=sim)
        self.global_watchlist = full_watchlist
    
    def update_global_watchlist_from_data(self, del_symbols):
        for symbol in del_symbols:
            if symbol in self.strategy.watchlist:
                self.strategy.watchlist.remove(symbol)
            if symbol in self.global_watchlist:
                self.global_watchlist.remove(symbol)
        print("Final watchlist: \n", self.global_watchlist)

    def get_quotes(self):
        self.global_quotes = {}
        for symbol in self.global_watchlist:
            quote = self.data_scraper.quote_as_pd(symbol)
            self.global_quotes[symbol] = quote
            tm.sleep(0.02)

    def send_quotes(self):
        for symbol, quote in self.global_quotes.items():
            if symbol in self.strategy.watchlist:
                self.strategy.append_quote(symbol, quote)
                
    def get_minutes(self):
        self.global_watchdata = self.data_scraper.get_today_minutes()

    def send_minutes(self):
        for symbol, data in self.global_watchdata.items():
            if symbol in self.strategy.watchlist:
                self.strategy.append_minute(symbol, data)

    def send_yesterday_minutes(self, prev_closes=False):
        for symbol, data in self.global_watchdata.items():
            if symbol in self.strategy.watchlist:
                if prev_closes:
                    self.strategy.append_yesterday_minutes(symbol, data, prev_c=self.global_prev_close[symbol])
                else:
                    self.strategy.append_yesterday_minutes(symbol, data)

    def calculations(self):
        self.strategy.step()

    def end(self):
        self.strategy.end()


    def run(self):
        self.starttime = datetime.now()

        if self.update_dailies:
            self.data_scraper.updateTrading212Dailies()

        self.set_global_watchlist()
        self.global_watchdata, no_data_symbols, self.global_prev_close = self.data_scraper.get_intraday_minutes(self.global_watchlist, flag='yesterday')
        self.update_global_watchlist_from_data(no_data_symbols)
        self.data_scraper.watchlist = self.global_watchlist
        self.send_yesterday_minutes(prev_closes=True)

        if self.wait_open and not self.in_trading_hours():
            while not self.in_trading_hours():
                tm.sleep(10)

        self.starttime = datetime.now()
        step = 0
        while (self.in_trading_hours() or self.ignore_market):
            tm.sleep((65.0 - (datetime.now().second+1)) % 60) # poll data 5 secs after full minute
            print("tick ", step)
            self.get_minutes()
            self.send_minutes()

            self.calculations()

            self.uptime =(datetime.now() - self.starttime)
            if self.max_minutes and (self.uptime > timedelta(minutes=self.max_minutes)):
                break
            tm.sleep(1)
            step += 1

        self.end()


class SimTrader(Trader):
    def __init__(self, market_days=1, **kwargs):
        super(SimTrader, self).__init__(**kwargs)
        self.watchdata_today = {}
        self.sim_dates = self.list_market_days(self.day, market_days)

    def list_market_days(self, last_day, market_days):
        cur_date = last_day
        dates = []
        for i in range(market_days):
            while cur_date.weekday() > 4:
                cur_date = cur_date - timedelta(1)
            dates.append(cur_date)
            cur_date = cur_date - timedelta(1)
        dates.reverse()
        return dates

    def get_minutes(self, step):
        open_time = self.day.replace(hour=13, minute=30,second=0)
        get_minute = (open_time + timedelta(minutes=step))

        for symbol in self.global_watchlist:
            if get_minute in self.watchdata_today[symbol].index.to_pydatetime():
                self.global_watchdata[symbol] = self.watchdata_today[symbol].loc[get_minute, :]
            else:
                self.global_watchdata[symbol] = pd.Series(name=get_minute, dtype='float64')

    def early_stop(self):
        positioned = self.strategy.in_position()
        return not positioned

    def run(self):

        if self.day.weekday() > 4:
            print("Requested day was weekend!")
            return

        if self.update_dailies:
            self.data_scraper.updateTrading212Dailies()

        for date in self.sim_dates:
            self.day = date
            self.strategy.initialize(self.day)

            self.set_global_watchlist(sim=True)
            self.global_watchdata, no_data_symbols, self.global_prev_close = self.data_scraper.get_intraday_minutes(self.global_watchlist, self.day, flag='yesterday')
            self.update_global_watchlist_from_data(no_data_symbols)
            self.data_scraper.watchlist = self.global_watchlist
            self.send_yesterday_minutes(prev_closes=True)

            self.watchdata_today, no_data_symbols, __ = self.data_scraper.get_intraday_minutes(self.global_watchlist, self.day)
            self.update_global_watchlist_from_data(no_data_symbols)

            for i in range(400):
                self.get_minutes(step=i)
                self.send_minutes()

                self.calculations()

                if self.early_stop():
                    break

            self.end()



commission_tr = Commission(fixed=1.17)
broker1 = Broker(commission=commission_tr, send_note=False)
broker2 = Broker(commission=commission_tr)

strat1 = MACDStrategy(broker=broker1, portfolio=Portfolio(), sma=50, name="macd-rsi-smatest")
# strat2 = (MomentumStrategy(broker=broker2, portfolio=Portfolio(), name="mom"))


# trader = Trader(update_dailies=False, ignore_market=True, wait_open=False, max_minutes=0)
trader = SimTrader(strategy=strat1, update_dailies=False, market_days=1, day=datetime(2020,8, 18))

trader.run()