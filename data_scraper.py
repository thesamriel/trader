import pathlib
import json
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, date
import finnhub
from slack import WebClient
from slack.errors import SlackApiError
from hist_data import AVClient, AV_API_KEY
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
import websocket
import threading
from keys import API_KEY_FINNHUB, SLACK_TOKEN, SCREENER_URL
import bs4
from utils import progressBar

cwd = pathlib.Path(__file__).parent.absolute()
json_path = cwd.joinpath('results.json')

def time_to_unix(t):
    return int(time.mktime(t.timetuple()))

def unix_to_time(t):
    return datetime.fromtimestamp(t)




class DataScraper(finnhub.Client):
    """ Combined Class that retrieves historical stock data via finnhub.io api and realtime quotes off the yahoo finance website """
    def __init__(self):
        super().__init__(API_KEY_FINNHUB)
        self.watchlist = ()
        self.minutes_today = {} 

    def candles_as_pd(self, *args):
        json_data = self.stock_candles(*args)
        pd_data = pd.DataFrame.from_dict(json_data)
        pd_data.rename(columns={'t': 'Date',
                                'c':'Close',
                                'o':'Open',
                                'h':'High',
                                'l': 'Low',
                                'v': 'Volume'}, inplace=True)

        pd_data.drop('s', axis=1, inplace=True)
        pd_data['Date'] = pd.to_datetime(pd_data['Date'], unit='s')
        pd_data.set_index('Date', inplace=True)
        return pd_data

    def candle_prep(self, json_data):
        pd_data = pd.DataFrame.from_dict(json_data)
        pd_data.rename(columns={'t': 'Time',
                                'c':'Close',
                                'o':'Open',
                                'h':'High',
                                'l': 'Low',
                                'v': 'Volume'}, inplace=True)

        pd_data.drop('s', axis=1, inplace=True)
        pd_data['Time'] = pd.to_datetime(pd_data['Time'], unit='s')
        pd_data.set_index('Time', inplace=True)
        return pd_data

    def quote_as_pd(self, symbol):
        json_data = self.quote(symbol)
        pd_data = pd.DataFrame(json_data, index=[0])
        pd_data.rename(columns={'t': 'Time',
                                'c':'Close',
                                'o':'Open',
                                'h':'High',
                                'l': 'Low',
                                'pc': 'Previous Close'}, inplace=True)
        pd_data['Time'] = pd.to_datetime(pd_data['Time'], unit='s')
        pd_data.set_index('Time', inplace=True)                                    
        return pd_data

    def updateTrading212Dailies(self):
        stock_list = pd.read_csv("Trading212US.csv")
        for symbol in progressBar(stock_list['Symbol'], "Updating Dailies: ", "Complete", length=50, decimals=1):
            while True:
                try:
                    data = self.stock_candles(symbol, 'D', time_to_unix(datetime(2019, 8,1,5,0,0)), time_to_unix((datetime.today()-timedelta(1)).replace(hour=23)))
                    break
                except finnhub.exceptions.FinnhubAPIException:
                    time.sleep(10)

            if data['s'] == 'no_data':
                continue
            
            data = self.candle_prep(data)
            data.to_csv("hist_data/{}_Daily.csv".format(symbol))

    def get_intraday_minutes(self, watchlist, day=None, flag='today'):
        no_data = []
        watch_dict = {}
        end_close = {}

        day = day if day else datetime.today()
        if flag == 'yesterday':
            delta = 3 if day.weekday() == 0 else 1
            day = day - timedelta(delta)

        save_path = Path("minute_data/" + day.date().isoformat())
        save_path.mkdir(parents=True, exist_ok=True)        
        time_start = time_to_unix(day.replace(hour=15,minute=30, second=0))
        time_end = time_to_unix(day.replace(hour=22,minute=0, second=30))

        for symbol in progressBar(watchlist, "Fetching intraday data of {}: ".format(flag), length=50, decimals=1):
            symbol_csv = save_path.joinpath("{}.csv".format(symbol))
            if symbol_csv.is_file():
                watch_dict[symbol] = pd.read_csv(symbol_csv, index_col='Time', parse_dates=True)
                end_close[symbol] = watch_dict[symbol]['Price'].iloc[-1]
                continue

            while True:
                try:
                    data = self.stock_candles(symbol, 1, time_start, time_end)
                    break
                except (finnhub.exceptions.FinnhubAPIException, requests.exceptions.RequestException) as e:
                    print(e)
                    time.sleep(5)

            if data['s'] == 'no_data':
                no_data.append(symbol)
                continue
            
            data = self.minutes_prep(data)
            watch_dict[symbol] = data
            end_close[symbol] = data['Price'].iloc[-1]
            data.to_csv(symbol_csv)

        if len(no_data) > 0:
            print("no intraday data found for these symbols: \n", no_data)   
        return watch_dict, no_data, end_close


    def minutes_prep(self, json_data):
        pd_data = pd.DataFrame.from_dict(json_data)
        pd_data.rename(columns={'t': 'Time',
                                'c':'Price',
                                'v': 'Volume'}, inplace=True)

        pd_data.drop(['s', 'o', 'h', 'l'], axis=1, inplace=True)
        pd_data['Time'] = pd.to_datetime(pd_data['Time'], unit='s')
        pd_data.set_index('Time', inplace=True)
        return pd_data

    def get_today_minutes(self):
        data = {}
        page = 0
        stock_count = 1
        cur_time = datetime.now().replace(second=0, microsecond=0)
        while stock_count > 0:
            # URL for stock screener: US, price [1,30], volume > 1,000,000
            url = url = SCREENER_URL + "?count=100&offset=" + str(page*100)
            try:
                r = requests.get(url)
                soup = bs4.BeautifulSoup(r.text, "lxml")
                table = soup.select('tr[class*=simpTblRow]')
                stock_count = len(table)
                for row in table:
                    row_soup = bs4.BeautifulSoup(str(row), "lxml")
                    symbol = row_soup.find('a', {'class':'Fw(600) C($linkColor)'}).text
                    if symbol in self.watchlist:
                        price = row_soup.find('span').text
                        volume = row_soup.find('td', {'aria-label':'Volume'}).text
                        data[symbol] = pd.DataFrame({'Price':[price], 'Volume':[volume]}, index=[cur_time])
                page += 1                
            except requests.exceptions.RequestException:
                time.sleep(10)

        return data


if __name__ == "__main__":


    # ws.send('{"type":"subscribe","symbol":"APA"}')

    # slackClient = WebClient(token=SLACK_TOKEN)
    # send_slack_dm(slackClient, "helo helo my friendos, look at those gainers: \n https://finance.yahoo.com/gainers")

    finnclient = DataScraper()
    finnclient.updateTrading212Dailies()
    # print(finnclient.quote('AAPL'))
    # apple_minutes = finnclient.candles_as_pd("AAPL", 1, time_to_unix(datetime(2020, 7,31,5,0,0)), time_to_unix(datetime(2020, 8,1,12,0,0)))
    # print(apple_minutes)

    # data not available for current day, use quote for realtime updates
    # apple_candle = finn_client.stock_candles('AAPL', 1, time_to_unix(datetime(2020, 7,27,20,30,0)), time_to_unix(datetime(2020, 7,28,16,0,0))) # 

    # print(apple_candle)

    # apple_candle = candle_prep(apple_candle)
    # print(apple_candle.head())


    # mpf.plot(apple_candle, type='candle', style='charles', volume=True)
    #fig, ax = plt.subplots()
    #candlestick_ohlc(ax, apple_candle[])


    # pushbullet_message("Test", "Jo, trade mal die Siemens Aktie weg ganz fix!")

    
""" Get Daily data for all stocks supported by Trading212 
    finn_client = DataScraper()
    stock_list = pd.read_csv("Trading212US.csv")
    for symbol in stock_list['Symbol'][60:]:
        print(symbol)
        while True:
            try:
                data = finn_client.stock_candles(symbol, 'D', time_to_unix(datetime(2019, 7,1,5,0,0)), time_to_unix(datetime(2020, 7,29,23,0,0)))
                break
            except finnhub.exceptions.FinnhubAPIException:
                time.sleep(10)

        if data['s'] == 'no_data':
            continue
        
        data = candle_prep(data)
        data.to_csv("hist_data/{}_Daily.csv".format(symbol))

"""