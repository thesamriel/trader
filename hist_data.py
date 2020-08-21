import requests
import os
import json
import io
import pandas as pd
from keys import AV_API_KEY


class AVClient(object):
    API_URL = "https://www.alphavantage.co/query?"

    def __init__(self, api_key, csv=False, requests_params=None):
        self.api_key = api_key
        self.csv = csv
        self.session = self._init__session()
        self._requests_params = requests_params

    def _init__session(self):
        session = requests.session()
        return session

    def _create_api_uri(self, base_url, **kwargs):
        param_url = base_url
        for i in kwargs:
            param_url = "{}&{}={}".format(param_url, i, kwargs[i])
        return param_url

    def _get(self, function, **kwargs):
        expl_csv = kwargs.pop("csv", False)
        url = self.API_URL + "function={}".format(function)
        url = self._create_api_uri(url, **kwargs)
        url += "&apikey=" + self.api_key

        if expl_csv is not None:
            return_csv=expl_csv 
        else:
            return_csv=self.csv

        if return_csv : url += "&datatype=csv" 
        response = self.session.get(url)
        return self._handle_response(response, return_csv=return_csv)

    def _handle_response(self, response, return_csv=False):
        if not str(response.status_code).startswith('2'):
            raise ValueError("Bad Response from Alpha Vantage. Status Code: {}".format(response.status_code))
        
        # raise an error if api call limit is reached (max. 5 calls per minute)
        if response.text.find("Thank you for using", 0, 50) != -1:
            raise ValueError("API calls limit reached, please wait a minute!")

        if return_csv:
            try:
                return pd.read_csv(io.StringIO(response.text), parse_dates=[0]).iloc[::-1]
            except ValueError:
                raise ValueError("Invalid Response: {}".format(response.text))
        else:
            try:
                return response.json()
            except ValueError:
                raise ValueError("Invalid Response: {}".format(response.text))
        return 


    def quote(self, symbol, csv=None):
        function = "GLOBAL_QUOTE"
        return self._get(function, symbol=symbol, csv=csv)

    def time_series_intraday(self, symbol, interval=5, outputsize='full', csv=None):
        function = "TIME_SERIES_INTRADAY"
        assert interval in (1,5,15,30,60), "interval value must be one of (1, 5, 15, 30, 60)"
        interval = str(interval) + "min"
        return self._get(function, symbol=symbol, interval=interval, outputsize=outputsize, csv=csv)

    def time_series_daily(self, symbol, outputsize='full', csv=None):
        function = "TIME_SERIES_DAILY"
        return self._get(function, symbol=symbol, outputsize=outputsize, csv=csv)

    def time_series_daily_adjusted(self, symbol, outputsize='full', csv=None):
        function = "TIME_SERIES_DAILY_ADJUSTED"
        return self._get(function, symbol=symbol, outputsize=outputsize, csv=csv)

    def time_series_weekly(self, symbol, outputsize='full', csv=None):
        function = "TIME_SERIES_WEEKLY"
        return self._get(function, symbol=symbol, outputsize=outputsize, csv=csv)

    def time_series_weekly_adjusted(self, symbol, outputsize='full', csv=None):
        function = "TIME_SERIES_WEEKLY_ADJUSTED"
        return self._get(function, symbol=symbol, outputsize=outputsize, csv=csv)




if __name__ == '__main__':

    df = pd.DataFrame({'nums': [1,2,3,4,5,6]})
    print(df, '\n')
    print(len(df['nums']))
    print(df['nums'].iloc[1:len(df['nums'])])