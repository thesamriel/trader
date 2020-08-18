# from portfolio import Portfolio
from portfolio import OrderStatus
from slack import WebClient
from slack.errors import SlackApiError
from keys import SLACK_TOKEN

def send_slack_dm(client, msg):
    try:
        client.chat_postMessage(
            channel='U0179PW8C4X',              # my User-ID
            text=msg)
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")
    except:
        return

class Broker():
    def __init__(self, commission=None, portfolio=None, send_note=False):
        self.commission = commission
        self.portfolio = portfolio
        self.delay = 0
        self.slackClient = WebClient(token=SLACK_TOKEN)
        self.send_note = send_note
        return

    def execute_buy(self, portfolio, order, current_price):
        if order.delay > 0:
            order.delay -= 1
            return order
        order.status = OrderStatus(order.status.value + 1)
        if not self.check_funds(portfolio, order, current_price):
            order.status = OrderStatus(4)
            return order
        else:
            order.status = OrderStatus(order.status.value + 1)

        portfolio.cash -= order.size * current_price
        order.status = OrderStatus(order.status.value + 1)
        if self.send_note:
            send_slack_dm(self.slackClient, "{} {} shares of {}!".format(order.ordertype, order.size, order.symbol))        
        print("Cash left to Trade after Buy: ", portfolio.cash)
        return order

    def execute_sell(self, portfolio, order, current_price):
        if order.delay > 0:
            order.delay -= 1
            return order
        order.status = OrderStatus(order.status.value + 1)
        order.status = OrderStatus(order.status.value + 1)
        portfolio.cash += order.size * current_price
        order.status = OrderStatus(order.status.value + 1)
        if self.send_note:
            send_slack_dm(self.slackClient, "{} {} shares of {}!".format(order.ordertype, order.size, order.symbol))        
        print("Cash left to Trade after Sell: ", portfolio.cash)        
        return order

    def check_funds(self, portfolio, order, current_price):
        return portfolio.cash >= order.size * current_price



class Commission():
    def __init__(self, fixed=0.0, percentage=0.0, fixed_per_share=0.0):
        self.fixed = fixed
        self.percentage = percentage
        self.fixed_per_share = fixed_per_share

        return