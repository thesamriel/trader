import bs4
import requests
from bs4 import BeautifulSoup

def parsePrice(symbol):
    url = "https://de.finance.yahoo.com/quote/" + symbol + "?p=" + symbol + "&.tsrc=fin-srch"
    r = requests.get(url)
    soup = bs4.BeautifulSoup(r.text, "lxml")
    price = soup.find('div', {'class': 'D(ib) Mend(20px)'}).find('span').text
    return price

if __name__ == "__main__":
    for i in range(10):
        print("Current stock price of Healthineers AG: ", parsePrice('SHL.DE'))