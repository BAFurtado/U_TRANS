"""
1. Create agents
2. Run sequence
3. Produce outputs

"""
import copy
import datetime
import json
import logging
import os

import numpy as np
import pandas as pd
import shapely

from shapely.geometry import Point

import read_data
from agents.business import Business
from agents.house import House
from agents.household import Household
from agents.industry import Industry
from agents.neighbourhood import Neighbourhood
from agents.person import Person
from analysis import calculate_statistics
from analysis import plotting
from markets.housingMarket import HousingMarket
from markets.labourMarket import LabourMarket


class Model:
    def __init__(self, params, param=None, value=None, seed=None, scene_params=None, verbose=False):
        self.id = datetime.datetime.utcnow().isoformat().replace(':', '_')
        logging.basicConfig()
        self.logger = logging.getLogger(f'SIM:{self.id[-6:]}')
        self.logger.setLevel(level=logging.INFO)
        self.logger.info('Simulation starting...')
        if not verbose:
            self.logger.setLevel(level=logging.DEBUG)
        self.np = np
        self.seed = self.np.random.RandomState(seed=seed)
        # All necessary data now comes from parameters at params.py and are kept within the simulation.
        self.params = params.copy()
        if param:
            self.params[param] = value
            new_path = os.path.join(self.params['SAVING_DIRECTORY'], f'{param}={value}')
            self.params['SAVING_DIRECTORY'] = new_path
            if not os.path.exists(new_path):
                os.mkdir(new_path)
        # always start from baseline and run for grace period before scenarios
        if not scene_params:
            self.scene_params = self.params
        else:
            self.scene_params = scene_params
        self.industries = dict()
        self.neighbourhoods = dict()
        self.persons = list()
        self.persons_id = 0
        self.houses = list()
        self.households = list()
        self.businesses = list()
        self.friend_links = list()
        self.current_step = 0
        self.housing_market = HousingMarket(self)
        self.labour_market = LabourMarket(self)
        self.space = None
        self.households_graveyard = list()
        # initialise reporter
        cols = ['current_step', 'population', 'hh_income', 'gini', 'unemployment',
                'housing_price_indexes', 'n_matching_housing', 'tom_buy', 'tom_sell',
                'total_posted_jobs', 'skill', 'n_matching_labour', 'recruit_outside', 'tom_labour']
        for i in list((self.params['INDUSTRIES']['name'])):
            cols += [f'person_income_{i}']
            cols += [f'skill_{i}']
            cols += [f'n_posted_jobs_{i}']
            cols += [f'n_applicants_{i}']
            cols += [f'n_workers_{i}']
        self.reporter = pd.DataFrame(columns=cols)

    # in the simulation class because the need to compare with max and min income
    def update_neighbourhoods(self):
        for neighbourhood in self.neighbourhoods.values():
            neighbourhood.update_all()
        all_incomes = [neighbourhood.data['avg_income'][-1] for neighbourhood in self.neighbourhoods.values()]
        max_income, min_income = np.max(all_incomes), np.min(all_incomes)
        all_services = [neighbourhood.data['n_service'][-1] for neighbourhood in self.neighbourhoods.values()]
        max_service, min_service = np.max(all_services), np.min(all_services)
        for neighbourhood in self.neighbourhoods.values():
            neighbourhood.update_income_score((neighbourhood.data['avg_income'][-1] - min_income) /
                                              (max_income - min_income) if max_income - min_income > 0 else 0)
            neighbourhood.update_service_score((neighbourhood.data['n_service'][-1] - min_service) /
                                               (max_service - min_service) if max_service - min_service > 0 else 0)

    def get_random_point_in_polygon(self, n, polygon):
        points = list()
        minx, miny, maxx, maxy = polygon.bounds.iloc[0]
        while len(points) < n:
            pnt = shapely.geometry.Point(self.seed.uniform(minx, maxx),
                                         self.seed.uniform(miny, maxy))
            if polygon.contains(pnt).iloc[0]:
                points.append(pnt)
        return points

    def set_max_distance(self):
        minx, miny, maxx, maxy = self.space.total_bounds
        self.params['MAX_DISTANCE_TO_JOB'] = \
            ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** .5
        self.logger.info(f"Max. simulation distance is = {self.params['MAX_DISTANCE_TO_JOB']:.4f}")

    def run(self):
        self.initialize()
        for i in range(self.params['TIME_TO_BE_ELIMINATED'] + self.params['GRACE_PERIOD'] + self.params['WEEKS']):
            self.step()
        for neighbourhood in self.neighbourhoods.values():
            dta = neighbourhood.data['housing_price_indexes']
            print(f'{neighbourhood.name} = {dta}')
        # Saving all data and params for each run!
        # Also saving data and used parameters for given simulation. Both have the name of the time of the simulation
        p = f"{self.scene_params['SAVING_DIRECTORY']}"
        if not os.path.exists(p):
            os.mkdir(p)
        p_csv = os.path.join(p, f'{self.id}.csv')
        self.reporter.to_csv(p_csv, index=False)
        with open(f"{self.scene_params['SAVING_DIRECTORY']}/{self.id}.json", 'w') as f:
            json.dump(self.scene_params, f, default=str)
        ######################
        # Neighbourhood data #
        ######################
        if self.params['REPORT_NEIGHBOURHOOD_DATA']:
            path = f"{self.scene_params['SAVING_DIRECTORY']}/neighbourhoods"
            if not os.path.exists(path):
                os.mkdir(path)
            for neighbourhood in self.neighbourhoods.values():
                out = pd.DataFrame()
                out = out.from_dict(neighbourhood.data)
                out.to_csv(f"{path}/{self.id}_{neighbourhood.name}.csv", sep=',', index=False)
        if self.scene_params['PLOT_NEIGHBOURHOOD_DATA']:
            neighbourhoods_data = dict()
            for neighbourhood in self.neighbourhoods.values():
                out = pd.DataFrame()
                out = out.from_dict(neighbourhood.data)
                neighbourhoods_data[neighbourhood.name] = out
            plotting.plot_neighbourhoods(neighbourhoods_data,
                                         f"{self.scene_params['SAVING_DIRECTORY']}/neighbourhoods",
                                         self.id,
                                         self.params['TIME_TO_BE_ELIMINATED'])

    def step(self):
        self.current_step += 1  # step increase at the beginning to avoid '%week = 0' issue
        # Checking for scenarios cases. Changing TYPICAL model parameters

        if self.current_step == self.params['GRACE_PERIOD']:
            self.params = copy.deepcopy(self.scene_params)

        # Change of growth rate parameter
        if self.current_step in self.params['INDUSTRIES']['growth_rate_mean']['old']['step-rate'] or \
                self.current_step in self.params['INDUSTRIES']['growth_rate_mean']['new']['step-rate']:
            [industry.update_growth_rate() for industry in self.industries.values()]

        ##################################
        # incoming persons and household #
        ##################################
        if self.current_step > 0:
            # Returns new people without a household
            new_comers = self.add_new_persons(round(self.params['PERC_ENTERING_MODEL'] * len(self.persons)))
            self.add_new_households(new_comers)
        ###################
        # Business Update #
        ###################
        [business.update_target_size() for business in self.businesses]
        [business.make_recruit_decision() for business in self.businesses]
        #################
        # Labour Market #
        #################
        # Include all candidates who are unemployed
        [p.job_search() for p in self.persons if p.industry.name == "unemployed"]
        # Match candidates and posts
        self.labour_market.match_process()
        # If business cannot find enough workers, recruit outside with PERC_RECRUIT_OUTSIDE
        n_total_recruit_outside = 0
        for business in self.businesses:
            if len(business.workers) < business.target_size:
                n_recruit = round(business.target_size - len(business.workers))
                if n_recruit > 0:  # service does not recruit from outside
                    business.tom += 1
                    if business.tom > self.params['MAX_BUSINESS_TOM']:
                        n_total_recruit_outside += n_recruit
                        business.recruit_outside(n_recruit)
                        business.tom = 0
                        self.logger.debug(f'business {business.id} recruit {n_recruit} from outside')
        self.reporter.loc[self.current_step, 'recruit_outside'] = n_total_recruit_outside
        ##########
        # Person #
        ##########
        # [person.update_income() for person in self.persons]
        [person.update_tom() for person in self.persons]
        [person.update_skill() for person in self.persons]
        # quit_chance = self.seed.uniform(0, 1, size=len(self.persons))
        # [person.update_quit_job(quit_chance[i]) for i, person in enumerate(self.persons)]
        #############
        # Household #
        #############
        [household.update_skilled() for household in self.households]
        hh_leaving = list()
        for household in self.households:
            # step fn includes: update list for sale. Listing of houses, intent to buy or to sell.
            # step fn returns true if both 'to leave' is true and no house
            if household.step():
                hh_leaving.append(household)
                [self.persons.remove(person) for person in household.members]
        self.households_graveyard += hh_leaving
        for hh in hh_leaving:
            hh.move_to(None)
            self.households.remove(hh)  # hh will sell the house and still participate in the housing mkt after leaving
        ##################
        # Housing market #
        ##################
        [household.choose_house_on_market() for household in self.households if household.to_buy_house]
        self.housing_market.match_process()
        self.housing_market.update_house_for_sale_list()
        #################
        # Neighbourhood #
        #################
        self.update_neighbourhoods()
        if self.params['REPORT_NEIGHBOURHOOD_DATA'] or self.params['PLOT_NEIGHBOURHOOD_DATA']:
            self.housing_market.update_nbhd_housing_price()
        # Neighbourhoods jobs are update within the labour market, just before clearing jobs data for the month
        #####################################
        # Saving data and generating report #
        #####################################
        self.output()
        self.logger.info(f'step {self.current_step}, population = {len(self.persons)}')
        unemployment_rate = len([person for person in self.persons if person.industry.name == 'unemployed']) / len(
            self.persons)
        self.logger.info(f'unemployment rate = {unemployment_rate}')

    def output(self):
        self.reporter.loc[self.current_step, 'current_step'] = self.current_step
        self.reporter.loc[self.current_step, 'population'] = len(self.persons)
        self.reporter.loc[self.current_step, 'unemployment'] = calculate_statistics.calculate_unemployment(self)
        incomes = [h.current_income for h in self.households]
        avg_hh_income = self.np.nanmean(incomes)
        self.reporter.loc[self.current_step, 'hh_income'] = avg_hh_income
        gini = calculate_statistics.calculate_gini(incomes, self)
        self.reporter.loc[self.current_step, 'gini'] = gini
        gdp = calculate_statistics.calculate_gdp(incomes)
        self.reporter.loc[self.current_step, 'gdp'] = gdp
        wealth_income = [h.get_house_gains() for h in self.households]
        wealth_gini = calculate_statistics.calculate_gini(wealth_income, self)
        self.reporter.loc[self.current_step, 'wealth_gini'] = wealth_gini
        tom_buy = [household.buyer_tom for household in self.households if household.to_buy_house]
        avg_tom_buy = self.np.nanmean(tom_buy) if tom_buy else None
        tom_sell = [house_for_sale.tom for house_for_sale in self.housing_market.house_for_sale_list]
        avg_tom_sell = self.np.nanmean(tom_sell) if tom_sell else None
        self.reporter.loc[self.current_step, 'tom_buy'] = avg_tom_buy
        self.reporter.loc[self.current_step, 'tom_sell'] = avg_tom_sell
        industry_names = list((self.params['INDUSTRIES']['name']))
        self.reporter.loc[self.current_step, f'skill'] \
            = self.np.nanmean([person.skill for person in self.persons])
        self.reporter.loc[self.current_step, f'person_income'] \
            = self.np.nanmean([person.income for person in self.persons])
        self.reporter.loc[self.current_step, f'tom_labour'] \
            = self.np.nanmean([person.TOM for person in self.persons
                               if person.industry.name == 'unemployed'])
        for i in industry_names:
            people_in_industry = [person for person in self.persons if person.industry.name == i]
            avg_skill = self.np.nanmean([person.skill for person in people_in_industry])
            avg_person_income = self.np.nanmean([person.income for person in people_in_industry])
            self.reporter.loc[self.current_step, f'n_workers_{i}'] = len(people_in_industry)
            self.reporter.loc[self.current_step, f'skill_{i}'] = avg_skill
            self.reporter.loc[self.current_step, f'person_income_{i}'] = avg_person_income

        self.logger.info(f'Gini coefficient for step {self.current_step} is {gini:.4f}')

    ######################################
    # INITIALIZING PROCEDURES
    ######################################
    def initialize(self):
        # Create agents. Allocate people into households. Households into neighbourhoods.
        # Create businesses from industries allocate them into neighbourhoods
        self.initialize_industry()
        self.initialize_persons()
        self.initialize_business()
        self.initialize_households()
        self.initialize_nbhd()
        self.allocate_business_into_hexagons()
        self.initialize_houses()
        self.allocate_renters_to_nbhds()
        self.update_neighbourhoods()
        self.set_max_distance()
        if not os.path.exists(self.params['SAVING_DIRECTORY']):
            os.mkdir(self.params['SAVING_DIRECTORY'])
        self.get_nbhd_data().to_csv('output\init_nbhd.csv', index=False)
        self.logger.info(f"Simulation started...: {len(self.industries)} industries, "
                         f"{len(self.persons)} people in {len(self.households)} households.")
        if self.params['PLOT_EACH_RUN']:
            self.print_map_info()

    def get_nbhd_data(self):
        keys = ['housing_price_indexes', 'income_score', 'service_score', 'avg_income', 'p_skilled', 'population',
                'unemployment rate', 'n_service']
        col_names = ['nbhd'] + ['init_h_price'] + keys
        dta = []
        for neighbourhood in self.neighbourhoods.values():
            dta.append(
                [neighbourhood.name] + [neighbourhood.init_avg_price] + [neighbourhood.data[key][0] for key in keys])
        return pd.DataFrame(dta, columns=col_names)

    def print_map_info(self, time_id='initial'):
        shape_name = self.params['SHAPE_NAME']
        plotting.plot_geo_business(background=self.space, businesses=self.businesses,
                                   path=self.params['SAVING_DIRECTORY'], time_id=time_id,
                                   run_id=self.id)

    def initialize_industry(self):
        print('Initialize industry...')
        for i in range(len(self.params['INDUSTRIES']['name'])):
            name = self.params['INDUSTRIES']['name'][i]
            industry = Industry(
                name=name,
                size=self.params['INDUSTRIES']['initial_size'][name],
                skill_min=self.params['INDUSTRIES']['skill_min'][name],
                income_min=self.params['INDUSTRIES']['income_min'][name],
                income_max=self.params['INDUSTRIES']['income_max'][name],
                # Growth rate and std. is a tuple, RATE is the second term -- index 1
                growth_rate_mean=self.params['INDUSTRIES']['growth_rate_mean'][name]['step-rate'][0]
                if name in self.params['INDUSTRIES']['growth_rate_mean'] else 0,
                growth_rate_sd=self.params['INDUSTRIES']['growth_rate_sd'][name]['step-rate'][0]
                if name in self.params['INDUSTRIES']['growth_rate_sd'] else 0,
                model=self
            )
            self.industries[name] = industry

    def initialize_persons(self):  # during initialization
        print('Initialize_persons...')
        new_comers = list()
        n = int(self.params['N_PEOPLE'])
        ages = self.seed.randint(self.params['MIN_AGE'], self.params['MAX_AGE'], size=int(n))
        females = self.seed.choice([True, False], size=int(n))
        industries = self.seed.choice(list(self.industries.values()),
                                      p=[i.size for i in self.industries.values()], size=int(n))
        income = [industry.draw_income() for industry in industries]
        for i in range(n):
            min_skill = industries[i].skill_min
            max_skill = 1 if industries[i].skill_min > self.params['SKILLED_THRESHOLD'] \
                else self.params['SKILLED_THRESHOLD']
            skill = self.seed.uniform(low=min_skill, high=max_skill)
            person = Person(_id=str(self.persons_id), age=ages[i], female=females[i], industry=industries[i],
                            skill=skill, income=income[i], model=self)
            self.persons_id += 1
            new_comers.append(person)
        self.persons += new_comers
        print(f'{len(self.persons)} people initialized')
        return new_comers

    def add_new_persons(self, n=None, ages=None, industry=None, females=None):  # during simulation
        new_comers = list()
        if not ages:
            ages = self.seed.randint(self.params['MIN_AGE'], self.params['MAX_AGE'], size=n)
        if not females:
            females = self.seed.choice([True, False], size=n)
        if not industry:
            industry = self.industries["unemployed"]
            mean_skill = [0.5 for i in range(n)]
        else:
            mean_skill = (industry.skill_min + 1) / 2 if industry.skill_min >= self.params['SKILLED_THRESHOLD'] \
                else (industry.skill_min + self.params['SKILLED_THRESHOLD']) / 2
            # mean = 0.75 for skilled, 0.25 for unskilled
        skill = np.clip(self.seed.normal(mean_skill, self.params['INITIAL_SKILL_VARIANCE'], size=n), 0, 1)
        for i in range(n):
            person = Person(_id=str(self.persons_id), age=ages[i], female=females[i], industry=industry,
                            skill=skill[i], income=industry.draw_income(), model=self)
            self.persons_id += 1
            new_comers.append(person)
        self.persons += new_comers
        return new_comers

    def initialize_households(self):  # during intialization
        print('Initialize households...')
        new_comers = self.persons
        # Marrying couples.
        females, males = list(), list()
        [females.append(p) if p.female else males.append(p) for p in new_comers]
        n_couples = min(len(females), len(males),
                        int((self.params['PERC_MARRIED'] * len(new_comers) / 2)))
        married_males, married_females = list(self.seed.choice(males, size=n_couples, replace=False)), \
                                         list(self.seed.choice(females, size=n_couples, replace=False))
        married_females.sort(key=lambda f: f.age)
        married_males.sort(key=lambda m: m.age)
        # Marrying couples and adding to Households
        [married_females[i].marry(married_males[i]) for i in range(n_couples)]
        for i in range(n_couples):
            self.households.append(Household(self, [females[i], males[i]]))
        # Populating households of singles
        singles = [p for p in new_comers if p.household is None]
        for single_person in singles:
            self.households.append(Household(self, [single_person]))
        print(f'{len(self.households)} households initialised')
        print(f'max hh income: {max([household.current_income for household in self.households])}')
        print(f'min hh income: {min([household.current_income for household in self.households])}')

    def add_new_households(self, new_comers):  # during simulation
        # Marrying couples.
        n_weights = 5  # This has to be a fixed number as it is.  int(self.params['N_WEIGHTS'])
        n = len(new_comers)
        is_married = self.seed.choice([True, False], p=[self.params['PERC_MARRIED'], 1 - self.params['PERC_MARRIED']],
                                      size=n)
        for i in range(n):
            new_hh = None
            if is_married[i]:
                female = not new_comers[i].female
                age = new_comers[i].age
                partner = self.add_new_persons(1, ages=[age], females=[female])[0]  # partner unemployed initially
                new_comers[i].marry(partner)
                new_hh = Household(self, [new_comers[i], partner])
            else:
                new_hh = Household(self, [new_comers[i]])
            self.households.append(new_hh)
            new_hh.renter_move()

    def initialize_nbhd(self):
        print('Initialize neighbourhoods')
        self.space = read_data.read_shapes(self.params['SHAPE_NAME'])
        initial_prices = pd.read_csv("data/initial_price.csv")
        dta = self.space.merge(initial_prices, left_on='Name', right_on='name')
        for index, row in dta.iterrows():
            name = row.Name
            initial_p = row.initial_price
            init_unemployment_index = row.initial_unemployement_index
            self.neighbourhoods[name] = Neighbourhood(name, self.space.geometry, self, initial_p,
                                                      init_unemployment_index)
            # print(f'nbhd {name} created with init_p = {initial_p}')
        self.logger.info(f'Read {len(self.neighbourhoods)} neighbourhoods from file')

    def initialize_houses(self):
        print('Initialize houses...')
        n_houses = int(self.params['N_HOUSES'])
        # equal number of houses in each nbhd
        neighbourhoods = self.seed.choice(list(self.neighbourhoods.values()), size=int(n_houses))
        var = self.params['VAR_INIT_PRICE']
        fluct = self.seed.uniform(-1 * var, var, size=n_houses)
        # random address or location for houses
        addresses = [self.find_house_address(n, n.minx, n.miny, n.maxx, n.maxy) for n in neighbourhoods]
        new_houses = []
        for i in range(n_houses):
            # Houses need owners. After creating, let's give them to owners.
            neighbourhood = neighbourhoods[i]
            init_p = neighbourhood.init_avg_price + fluct[i]
            new_house = House(neighbourhood=neighbourhood, model=self, location=addresses[i], initial_price=init_p)
            new_houses.append(new_house)
        new_houses.sort(key=lambda f: f.initial_price, reverse=True)  # sort from high to low
        for new_house in new_houses:
            owner = self.get_initial_owner(new_house)
            if owner:
                owner.set_house(new_house, 'owned')  #
                owner.move_to(new_house.neighbourhood)
                self.houses.append(new_house)
        self.logger.info(f'Intend to create {n_houses} houses, actually created {len(self.houses)} houses')

    def allocate_renters_to_nbhds(self):
        [household.renter_move() for household in self.households if not household.house]

    # Raises an error when cannot return an owner
    def get_initial_owner(self, house):
        prospect_owners = [household for household in self.households if
                           household.house is None and household.can_afford(house.initial_price)]
        if prospect_owners:
            return self.seed.choice(prospect_owners)
        else:
            print(f'House not created, house price = {house.initial_price}')
            return None

    def find_house_address(self, n, minx, miny, maxx, maxy):
        while True:
            # Trying to get a point within the envelope of the region. Uniform distribution
            address = shapely.geometry.Point(self.seed.uniform(minx, maxx), self.seed.uniform(miny, maxy))
            if n.geometry.contains(address).iloc[0]:
                return address

    # will assign location later in initialize_business_location
    def initialize_business(self):
        print('Initialize businesses')
        ids = 0
        for industry_name in self.industries.keys():
            n_businesses = self.params['INDUSTRIES']['n_businesses'][industry_name]
            if n_businesses > 0:
                workers_in_industry = [person for person in self.persons if person.industry.name == industry_name]
                self.seed.shuffle(workers_in_industry)
                business_in_industry = []
                for i in range(n_businesses):
                    # Business location for this model is set at fn 'self.allocate_business_into_hexagons'
                    this_business = Business(ids, industry=self.industries[industry_name], location=None, model=self,
                                             neighbourhood=None)
                    business_in_industry.append(this_business)
                    self.businesses.append(this_business)
                    ids += 1
                # random allocation of workers to businesses
                businesses_assigned = self.seed.choice(business_in_industry, size=len(workers_in_industry))
                for i in range(len(workers_in_industry)):
                    business = businesses_assigned[i]
                    worker = workers_in_industry[i]
                    business.hire(worker, worker.income)
        for business in self.businesses:
            business.target_size = len(business.workers)
        self.logger.info(f'Created {len(self.businesses)} businesses and assigned workers to them...')

    def allocate_business_into_hexagons(self):
        print('Allocate business into hexagons...')
        for business in self.businesses:
            if business.industry.name == 'old':
                nbhd_name = self.seed.choice(['ring1a', 'ring1b', 'ring1c'])
            elif business.industry.name == 'new':
                nbhd_name = self.seed.choice(['ring1d', 'ring1e', 'ring1f'])
            elif business.industry.name == 'service':
                nbhd_name_list = [neighbourhood.name for neighbourhood in self.neighbourhoods.values()]
                nbhd_name_list.append('center')  # center is twice as dense
                nbhd_name = self.seed.choice(nbhd_name_list)
            if nbhd_name:
                poly = self.space.loc[self.space['Name'] == nbhd_name]['geometry']
                pt = self.get_random_point_in_polygon(1, poly)[0]
                business.location = pt
                business.neighbourhood = self.neighbourhoods[nbhd_name]


def main(params, param=None, value=None, seed=0, scene_params=None, verbose=False):
    my_model = Model(params, param, value, seed, scene_params, verbose=verbose)
    my_model.run()
    my_model.logger.info('All done...')
