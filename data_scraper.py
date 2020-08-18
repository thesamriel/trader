import pathlib
import json
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import finnhub
from slack import WebClient
from slack.errors import SlackApiError
from hist_data import AVClient, AV_API_KEY
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
import websocket
import threading
from keys import API_KEY_FINNHUB, SLACK_TOKEN

cwd = pathlib.Path(__file__).parent.absolute()
json_path = cwd.joinpath('results.json')

def time_to_unix(t):
    return int(time.mktime(t.timetuple()))

def unix_to_time(t):
    return datetime.fromtimestamp(t)

def candle_prep(json_data):
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


def send_slack_dm(client, msg):
    try:
        response = client.chat_postMessage(
            channel='U0179PW8C4X',              # my User-ID
            text=msg)
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")



class DataScraper(finnhub.Client):
    def __init__(self):
        super().__init__(API_KEY_FINNHUB)
        self.watchlist = ()
        self.minutes_today = {}
        websocket.enableTrace("True")
        self.ws = websocket.WebSocketApp("wss://ws.finnhub.io?token=bsd8487rh5r8dht947q0",
                        on_message = lambda ws,msg: self.on_message(ws, msg),
                        on_error   = lambda ws,msg: self.on_error(ws, msg),
                        on_close   = lambda ws:     self.on_close(ws),
                        on_open    = lambda ws:     self.on_open(ws))
        
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        
        return
    def start_ws_thread(self):
        self.ws_thread.start()

    def keep_websocket_running(self):
        print("Threads in Datascraper: ", threading.active_count())
        if not self.ws.sock:
            if self.ws:
                self.ws.close()
            print("Trying to create a new socket")
            self.ws = websocket.WebSocketApp("wss://ws.finnhub.io?token=bsd8487rh5r8dht947q0",
                        on_message = lambda ws,msg: self.on_message(ws, msg),
                        on_error   = lambda ws,msg: self.on_error(ws, msg),
                        on_close   = lambda ws:     self.on_close(ws),
                        on_open    = lambda ws:     self.on_open(ws))
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.start()
            conn_timeout = 10
            while self.ws.sock and (not self.ws.sock.connected and conn_timeout):
                time.sleep(1)
                conn_timeout -= 1

    def on_message(self, ws, message):
        # print(message)
        json_message = json.loads(message)

        if json_message['type'] == 'trade':
            symbol = json_message['data'][0]['s']
            tick_frame = pd.DataFrame(json_message['data'])
            tick_frame['t'] = pd.to_datetime(tick_frame['t'], unit='ms').dt.floor('Min')
            tick_frame.rename(columns={'t':'Time',
                                        'v': 'Volume',
                                        'p': 'Price'}, inplace=True)
            tick_frame.drop('s', axis=1, inplace=True)
            tick_frame.set_index('Time', inplace=True)

            if len(self.minutes_today[symbol].index) >= 1 and self.minutes_today[symbol].index[-1] == tick_frame.index[-1]:
                new_vol = tick_frame['Volume'].iloc[-1]
                old_vol = self.minutes_today[symbol]['Volume'].iloc[-1]
                self.minutes_today[symbol].iat[-1, self.minutes_today[symbol].columns.get_loc('Volume')] = old_vol + new_vol
                self.minutes_today[symbol].iat[-1, self.minutes_today[symbol].columns.get_loc('Price')] = (self.minutes_today[symbol]['Price'].iloc[-1] * old_vol + tick_frame['Price'].iloc[-1] * new_vol) / (old_vol + new_vol)
            else:
                self.minutes_today[symbol] = self.minutes_today[symbol].append(tick_frame)
   

    def on_error(self, ws, error):
        print("ERRROR in websocket!")
        print(error)

    def on_close(self, ws):
        print("### closed ###")

    def on_open(self, ws):
        for symbol in self.watchlist:
            ws.send('{"type":"subscribe","symbol": "%s"}' % symbol)
            self.minutes_today[symbol] = pd.DataFrame()   

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
        print("Fetching Daily Data, this may take a while ...")
        for symbol in stock_list['Symbol']:
            while True:
                try:
                    data = self.stock_candles(symbol, 'D', time_to_unix(datetime(2019, 8,1,5,0,0)), time_to_unix((datetime.today()-timedelta(1)).replace(hour=23)))
                    break
                except finnhub.exceptions.FinnhubAPIException:
                    time.sleep(10)

            if data['s'] == 'no_data':
                continue
            
            data = candle_prep(data)
            data.to_csv("hist_data/{}_Daily.csv".format(symbol))

    def get_intraday_minutes(self, watchlist, day=None, flag='today'):
        no_data = []
        watch_dict = {}
        prev_closes = {}

        print("Fetching intraday data...")
        day = day if day else datetime.today()
        if flag == 'yesterday':
            delta = 3 if day.weekday() == 0 else 1
            day = day - timedelta(delta)
        time_start = time_to_unix(day.replace(hour=15,minute=30, second=0))
        time_end = time_to_unix(day.replace(hour=22,minute=0, second=30))
        for symbol in watchlist:
            while True:
                try:
                    data = self.stock_candles(symbol, 1, time_start, time_end)
                    break
                except finnhub.exceptions.FinnhubAPIException or urllib3.exceptions.ReadTimeoutError:
                    time.sleep(10)

            if data['s'] == 'no_data':
                no_data.append(symbol)
                continue
            
            data = self.minutes_prep(data)
            watch_dict[symbol] = data
            prev_closes[symbol] = data['Price'].iloc[-1]
            data.to_csv("minute_data/{}_Intraday_{}.csv".format(symbol, day.date().isoformat()))

        if len(no_data) > 0:
            print("no intraday data found for these symbols: \n", no_data)   
        return watch_dict, no_data, prev_closes


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
        data = self.minutes_today.copy()
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