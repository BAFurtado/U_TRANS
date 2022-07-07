import copy
import math

from markets.housingMarket import HouseForSale, Bid

""" Split and merge households may also be located somewhere """

""" Potential considerations when choosing residences:
    1) house price, 
    2) energy efficiency of the house, 
    3) safety of neighbourhood, 
    4) quality of parks/greenspace, 
    5) quality of schools, 
    6) distance to work, 
    7) good public transport, 
    8) availability of shops/cafes/supermarkets/restaurants, 
    9) access to healthcare services and 
    10) dwelling size.
"""


# It returns distance according to the original data unit of the input shapefile
def calculate_distance(place1, place2):
    return ((place1.x - place2.x) ** 2 + (place1.y - place2.y) ** 2) ** .5


class Household:
    def __init__(self, model, *members):
        self.model = model
        self.members = dict()
        self.skilled = False
        self.previous_income = 0.0  # annual income
        self.current_income = 0.0  # annual income
        if members:
            self.add_member(members[0])
            self.current_income = sum(m.income for m in self.members.values())
            self.previous_income = self.current_income
            self.update_skilled()
        self.house = None
        self.to_sell_house = False
        self.to_buy_house = False
        self.to_leave = False
        self.house_for_sale = None
        self.house_gains = 100.0  # index value
        self.house_acquisition_step = None
        self.w_distance_to_work = max(0, model.seed.normal(model.params['MEAN_WEIGHT_DISTANCE'], model.params['MEAN_WEIGHT_DISTANCE']/3))
        self.w_income = max(0, model.seed.normal(model.params['MEAN_WEIGHT_INCOME'], model.params['MEAN_WEIGHT_INCOME']/3))
        self.w_price = max(0, model.seed.normal(model.params['MEAN_WEIGHT_PRICE'], model.params['MEAN_WEIGHT_PRICE']/3))
        self.w_service = max(0, model.seed.normal(model.params['MEAN_WEIGHT_SERVICE'], model.params['MEAN_WEIGHT_SERVICE']/3))
        self.w_unemployment = max(0, model.seed.normal(model.params['MEAN_WEIGHT_UNEMPLOYMENT'], model.params['MEAN_WEIGHT_UNEMPLOYMENT']/3))
        if self.skilled:
            self.w_p_skilled = max(0, model.seed.normal(model.params['MEAN_WEIGHT_SKILL'], model.params['MEAN_WEIGHT_SKILL']/3))
        else:
            self.w_p_skilled = 0  # non-skilled hh do not care
        # make sure all weights add to 1
        total = self.w_distance_to_work + self.w_income + self.w_price + self.w_service + self.w_unemployment + self.w_p_skilled
        self.w_distance_to_work /= total
        self.w_income /= total
        self.w_price /= total
        self.w_service /= total
        self.w_unemployment /= total
        self.w_p_skilled /= total
        self.buyer_tom = 0
        self.random_week = self.model.seed.choice(
            range(1, (1 + model.params['WEEKS_PER_YEAR'])))  # change income once a year
        self.neighbourhood = None
        self.min_house_score = self.model.seed.normal(self.model.params['MIN_HOUSE_SCORE'], self.model.params['MIN_HOUSE_SCORE_SD'])

    def add_member(self, *members):
        for m in members[0]:
            self.members[m.id] = m
            self.members[m.id].household = self

    def update_skilled(self):
        self.skilled = any([self.members[k].skill > self.model.params['SKILLED_THRESHOLD'] for k in self.members])

    def set_house(self, house, tenure):
        self.house = house
        self.house_for_sale = False
        house.set_household(self, tenure)

    def update_income(self):
        self.previous_income = self.current_income
        self.current_income = sum(m.income for m in self.members.values())
        # return self.current_income

    def buy_house(self, house):
        self.house = house
        self.house_acquisition_step = copy.copy(self.model.current_step)
        house.occupier = self
        self.move_to(house.neighbourhood)
        self.to_buy_house = False
        self.buyer_tom = 0

    def get_house_gains(self):
        self.calculate_house_gains()
        return self.house_gains

    def calculate_house_gains(self):
        if self.house:
            if self.house.occupier:
                if self.house.occupier == self:
                    if self.house_acquisition_step:
                        p0 = self.model.reporter.loc[self.house_acquisition_step, 'housing_price_indexes']
                        p1 = self.model.reporter.loc[self.model.current_step, 'housing_price_indexes']
                        if self.house_acquisition_step != self.model.current_step:
                            self.house_gains *= 1 + (p1 - p0) if not math.isnan(p1) else 1
                            # The difference for current period has been calculated. Reinitiate period for next count.
                            self.house_acquisition_step = copy.copy(self.model.current_step)

    def sell_house(self):
        self.calculate_house_gains()
        self.house_acquisition_step = None
        self.house = None
        self.to_sell_house = False
        self.house_for_sale = None

    def update_to_leave(self):
        if self.model.current_step % self.random_week == 0:
            if all([m.industry.name == 'unemployed' for m in self.members.values()]):
                if any([m.skill > self.model.params['SKILLED_THRESHOLD'] for m in self.members.values()]):
                    if self.model.seed.uniform(0, 1) < self.model.params['PERC_LEAVING_JOB_OUTSIDE']:
                        self.to_leave = True
                else:
                    if self.model.seed.uniform(0, 1) < self.model.params['PERC_LEAVING_UNEMPLOYED']:
                        self.to_leave = True
            if self.model.seed.uniform(0, 1) < self.model.params['PERC_LEAVING_RANDOM']:
                # person in hh quit job and leave
                self.to_leave = True
                [person.employer.fire(person) for person in self.members.values() if person.employer]

    def update_to_sell_house(self):
        if not self.to_sell_house and self.house:
            # If current income is a percentage higher than previously, sell house
            if self.previous_income > 0 and \
                    (self.current_income - self.previous_income) / self.previous_income \
                    <= self.model.params['DELTA_INCOME_SELL_HOUSE']:
                self.to_sell_house = True
                self.list_house_on_market()
            elif self.to_leave:
                self.to_sell_house = True
                self.list_house_on_market()
            elif self.model.current_step % self.random_week == 0 \
                    and self.model.seed.uniform(0, 1) < self.model.params['P_RANDOM_SELL']:
                self.to_sell_house = True
                self.list_house_on_market()

    def update_to_buy_house(self):
        if self.house is None and not self.to_buy_house \
                and self.model.current_step % self.random_week == 0 \
                and self.model.seed.uniform(0, 1) < self.model.params['P_RENTER_BUY']:
            self.to_buy_house = True
        elif self.to_buy_house and self.buyer_tom > self.model.params['MAX_TOM_HOUSING']:
            self.to_buy_house = False

    def list_house_on_market(self):
        ref_price = self.model.housing_market.reference_price(self.house.neighbourhood)
        ask_price = ref_price * (1 + self.model.params['ASK_PREMIUM'])
        house_for_sale = HouseForSale(seller=self, house=self.house, ask_price=ask_price)
        self.model.housing_market.house_for_sale_list.append(house_for_sale)
        self.house_for_sale = house_for_sale

    # Houses are heterogeneous in the sense of the neighbourhood they are located in.
    def choose_house_on_market(self):
        house_selected = None
        affordable_houses_for_sale = [house_for_sale for house_for_sale in self.model.housing_market.house_for_sale_list
                             if self.can_afford(house_for_sale.ask_price)]
        if affordable_houses_for_sale:
            if len(affordable_houses_for_sale) > self.model.params['N_HOUSES_TO_EVALUATE']:
                affordable_houses_for_sale = list(self.model.seed.choice(affordable_houses_for_sale,
                                                                size=int(self.model.params['N_HOUSES_TO_EVALUATE']),
                                                                replace=False))
            affordable_houses_for_sale.sort(key=lambda h: self.evaluate_house(h))
            best_house = affordable_houses_for_sale[-1]
            best_score = self.evaluate_house(best_house)
            house_selected = best_house if best_score > self.min_house_score else None
            #if house_selected:
            #    print(f'house in {best_house.house.neighbourhood.name} selected, score = {best_score}')
            #else:
            #    print(f'house in {best_house.house.neighbourhood.name} NOT selected, score = {best_score}')
            if house_selected:
                bid_price = self.bid_price(house_selected)
                bid = Bid(buyer=self, house_for_sale=house_selected, bid_price=bid_price)
                house_selected.bid_list.append(bid)
        return house_selected

    def bid_price(self, house_selected):
        p_adjust = self.buyer_tom * self.model.params['BID_INCREASE_PER_WEEK'] - self.model.params['BID_DISCOUNT']
        max_budget = self.current_income * \
                     self.model.params['MONTHS_PER_YEAR'] * \
                     self.model.params['MAX_HOUSE_PRICE_TO_ANNUAL_HH_INCOME']
        return min(house_selected.ask_price * (1 + p_adjust) + self.model.seed.uniform(0, 1), max_budget)

    def can_afford(self, price):
        max_income_ratio = self.model.params['MONTHS_PER_YEAR'] * self.model.params[
            'MAX_HOUSE_PRICE_TO_ANNUAL_HH_INCOME']
        if self.current_income * max_income_ratio >= price:
            return True
        return False

    def evaluate_house(self, house_for_sale):
        neighbourhood = house_for_sale.house.neighbourhood
        distance_to_work = [calculate_distance(m.employer.location, house_for_sale.house.location)
                            if m.employer else 0
                            for m in self.members.values()]
        avg_distance = sum(distance_to_work) / len(distance_to_work)
        distance_index = 1 - avg_distance / self.model.params['MAX_DISTANCE_TO_JOB']
        max_budget = self.current_income * self.model.params['MONTHS_PER_YEAR'] * \
                     self.model.params['MAX_HOUSE_PRICE_TO_ANNUAL_HH_INCOME']
        price_score = 1 - house_for_sale.ask_price / max_budget
        # the individual scores are normalised to [0, 1]
        nbhd_score = self.w_distance_to_work * distance_index + \
                     self.w_income * neighbourhood.data['income_score'][-1] + \
                     self.w_p_skilled * neighbourhood.data['p_skilled'][-1] + \
                     self.w_price * price_score + \
                     self.w_service * neighbourhood.data['service_score'][-1] + \
                     self.w_unemployment * (1 - neighbourhood.data['unemployment rate'][-1]) + \
                     self.model.seed.uniform(-self.model.params['NBHD_RANDOM_FACTOR_RANGE'], self.model.params['NBHD_RANDOM_FACTOR_RANGE'])  # unobservable factor
        return nbhd_score

    def update_buyer_tom(self):
        if self.to_buy_house:
            self.buyer_tom += 1

    def remove_members(self):
        for m in self.members.values():
            self.model.persons.remove(m)
        self.members = dict()

    def step(self):
        # First, person update income, job seeking
        self.update_to_leave()
        # Only when both to_leave and no house will a hh and its members be removed
        if self.to_leave and not self.house:
            [person.employer.fire(person) for person in self.members.values() if person.employer]
            self.remove_members()
            return True
        self.update_income()
        self.update_buyer_tom()
        self.update_to_sell_house()
        self.update_to_buy_house()
        # Moved process of choosing house for simulation.
        # We have to do it after the listing of all households is complete

    def move_to(self, neighbourhood):
        # if self.neighbourhood and neighbourhood:
        #    print(f'renter hh move from nbhd {self.neighbourhood.name} to {neighbourhood.name}')
        self.neighbourhood = neighbourhood

    def renter_move(self):
        if not self.house:
            if any([m.employer for m in self.members.values()]):
                nbhds = []
                for m in self.members.values():
                    if m.employer:
                        nbhds += [neighbourhood for neighbourhood in self.model.neighbourhoods.values() if
                                  neighbourhood.name[1] == m.employer.neighbourhood.name[1]]  # a,b...f
                if self.neighbourhood not in nbhds:
                    # Only assign a random neighbourhood, if current not in members' employers' location.
                    nbhd = self.model.seed.choice(nbhds)
                    self.move_to(nbhd)
            else:
                if not self.neighbourhood:
                    nbhd = self.model.seed.choice(list(self.model.neighbourhoods.values()),
                                                  p=[neighbourhood.init_unemployment_index
                                                     for neighbourhood in self.model.neighbourhoods.values()])
                    self.move_to(nbhd)
