import bs4
import requests
from bs4 import BeautifulSoup
from time import sleep
from datetime import datetime

test_symbols = ["AA", "AAL", "AAN", "AAOI", "AAP", "AAPL", "AAXN", "ABBV", "ABCB", "ABC", "AB", "ABMD"]

def parsePrice(symbol):
    url = "https://de.finance.yahoo.com/quote/" + symbol + "?p=" + symbol + "&.tsrc=fin-srch"
    r = requests.get(url)
    soup = bs4.BeautifulSoup(r.text, "lxml")
    price = soup.find('div', {'class': 'D(ib) Mend(20px)'}).find('span').text
    return price

def parseWatchlist():
    page = 0
    stock_count = 1
    while stock_count > 0:
        # URL for stock screener: US, price [1,30], volume > 1,000,000
        url = "https://finance.yahoo.com/screener/unsaved/2144d085-b107-4462-8d5f-483a4a31513a?count=100&offset=" + str(page*100)
        r = requests.get(url)
        soup = bs4.BeautifulSoup(r.text, "lxml")
        table = soup.select('tr[class*=simpTblRow]')
        stock_count = len(table)
        for row in table:
            row_soup = bs4.BeautifulSoup(str(row), "lxml")
            symbol = row_soup.find('a', {'class':'Fw(600) C($linkColor)'}).text
            price = row_soup.find('span').text
            print("Current stock price of {} is: {}".format(symbol, price))
        page += 1

if __name__ == "__main__":
    
    for i in range(10):
        parseWatchlist()

        sleep((65.0 - (datetime.now().second+1)) % 60)