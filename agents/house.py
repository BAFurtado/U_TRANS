"""
(flat, detached, non-detached),
the tenure type (rented, owned with mortgage, owned outright, social housing)
characteristics of the accommodation (number of bedrooms, bathrooms, public rooms, whether it has a garage, and
whether it has a garden etc.).

"""
from markets.housingMarket import TradeRecord


class House:
    def __init__(self, _id=0, model=None, neighbourhood=None, initial_price=-1, _type='detached', location=None,
                 tenure='owned',
                 age=0, bedrooms=2, bathrooms=2,
                 rooms=2, heating=True, glazing=True, parking=True, garden=True):
        self.id = _id
        self.model = model
        self.neighbourhood = neighbourhood
        self.initial_price = initial_price
        self.type = _type
        self.location = location # a shapely point
        self.tenure = tenure
        self.occupier = None
        self.attributes = {'age': age, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
                           'rooms': rooms, 'heating': heating, 'glazing': glazing,
                           'parking': parking, 'garden': garden}
        # This may need to change depending on info to record. Tuple: price and date
        trade_record = TradeRecord(self, initial_price, 0)
        self.trade_records = [trade_record]
        model.housing_market.trade_record_list.append(trade_record)

    def set_household(self, occupier, tenure_type):
        self.tenure = tenure_type
        self.occupier = occupier
