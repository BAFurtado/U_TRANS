import pandas as pd


class HousingMarket:
    def __init__(self, model=None):
        # List of houses for sale
        self.house_for_sale_list = list()
        self.trade_record_list = list()
        self.model = model
        self.price_index = [1]  # price index starts from 1

    def trade_house(self, buyer=None, seller=None, house_for_sale=None, final_price=-1.0):
        house = house_for_sale.house
        seller.sell_house()
        buyer.buy_house(house)
        self.house_for_sale_list.remove(house_for_sale)
        trade_record = TradeRecord(house, final_price, self.model.current_step)
        self.trade_record_list.append(trade_record)

    def match_process(self):
        index_all, n_transactions = list(), 0
        n_bids = sum([len(house_for_sale.bid_list) for house_for_sale in self.house_for_sale_list])
        self.model.reporter.loc[self.model.current_step, 'n_houses_for_sale'] = len(self.house_for_sale_list)
        self.model.reporter.loc[self.model.current_step, 'n_bids'] = n_bids
        for house_for_sale in self.house_for_sale_list:
            if house_for_sale.bid_list:
                highest_bid = max(house_for_sale.bid_list, key=sort_bid)
                if house_for_sale.ask_price <= highest_bid.bid_price:
                    final_price = (house_for_sale.ask_price + highest_bid.bid_price) / 2
                    index_all.append(final_price / house_for_sale.house.initial_price)
                    n_transactions += 1
                    buyer = highest_bid.buyer
                    seller = house_for_sale.house.occupier
                    self.trade_house(buyer=buyer, seller=seller, house_for_sale=house_for_sale, final_price=final_price)
        if index_all:
            avg_index = self.model.np.nanmean(index_all)
            self.price_index.append(avg_index)
        else:
            self.price_index.append(self.price_index[-1])  # if no transaction, index is flat
        self.model.reporter.loc[self.model.current_step, 'n_matching_housing'] = n_transactions
        self.model.reporter.loc[self.model.current_step, 'housing_price_indexes'] = self.price_index[-1]
        self.model.logger.info(
            f'n houses for sale: {len(self.house_for_sale_list)}, n bids: {n_bids}, n transactions: {n_transactions}, housing price index: {self.price_index[-1]}')

    def update_house_for_sale_list(self):
        random_list = self.model.seed.uniform(0, 1, len(self.house_for_sale_list))
        for house_for_sale in self.house_for_sale_list:
            if not house_for_sale.seller.to_leave and house_for_sale.tom > self.model.params['MAX_TOM_HOUSING']:
                self.take_down_house_for_sale(house_for_sale)
        for house_for_sale, random_number in zip(self.house_for_sale_list, random_list):
            house_for_sale.tom += 1
            price_adjustment = self.model.params['ASK_PREMIUM'] - house_for_sale.tom * self.model.params[
                'ASK_DECREASE_PER_WEEK']
            reference_price = self.reference_price(house_for_sale.house.neighbourhood)
            new_ask_price = (1 + price_adjustment) * reference_price + random_number
            house_for_sale.set_ask_price(new_ask_price)

    def take_down_house_for_sale(self, house_for_sale):
        print(f'house for sale taken down, TOM= {house_for_sale.tom}')
        self.house_for_sale_list.remove(house_for_sale)
        house_for_sale.seller.to_sell_house = False
        house_for_sale.seller.house_for_sale = None

    def reference_price(self, neighbourhood):
        avg_index = self.model.np.nanmean(neighbourhood.data['housing_price_indexes']
                                          [-min(int(self.model.params['REFERENCE_PRICE_PERIOD']),
                                                len(neighbourhood.data['housing_price_indexes'])):])
        ref_price = neighbourhood.init_avg_price * avg_index
        # print(f'ref_price for {neighbourhood.name} = {ref_price}, initial price = {neighbourhood.init_avg_price}')
        return ref_price

    def update_nbhd_housing_price(self):
        month_trades = [t for t in self.trade_record_list if t.step == self.model.current_step]
        for neighbourhood in self.model.neighbourhoods.values():
            n_trades = [t.final_price / neighbourhood.init_avg_price for t in month_trades if
                        t.house.neighbourhood.name == neighbourhood.name]
            neighbourhood.data['housing_price_indexes'].append(
                self.model.np.nanmean(n_trades) if n_trades else neighbourhood.data['housing_price_indexes'][-1])

    def process_data(self, last_step):
        # Averaging final prices by neighbourhood.
        results = pd.DataFrame(columns=['avghprcs'])
        nbhds = set(map(lambda t: t.house.neighbourhood.name, self.trade_record_list))
        for nbhd in nbhds:
            prices = [t.final_price for t in self.trade_record_list
                      if t.house.neighbourhood.name == nbhd and t.step > last_step]
            if prices:
                nbhd_prices = self.model.np.nanmean(prices)
                results.loc[nbhd, 'avghprcs'] = nbhd_prices
        return results.reset_index().rename(columns={'index': 'Name'})


class HouseForSale:
    def __init__(self, seller, house=None, ask_price=-1.0):
        self.seller = seller
        self.house = house
        self.ask_price = ask_price
        self.bid_list = list()
        self.tom = 0  # time on market

    def set_ask_price(self, _ask_price):
        self.ask_price = _ask_price

    def add_bid(self, bid):
        self.bid_list.append(bid)


def sort_bid(bid):
    return bid.bid_price


class Bid:
    def __init__(self, buyer=None, house_for_sale=None, bid_price=0.0):
        self.buyer = buyer
        self.house_for_sale = house_for_sale
        self.bid_price = bid_price


class TradeRecord:
    def __init__(self, house=None, final_price=-1.0, step=0):
        self.house = house
        self.final_price = final_price
        self.step = step
