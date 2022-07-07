# Generating abstract points for businesses and polygons for neighbourhoods
from collections import defaultdict


class Neighbourhood:

    def __init__(self, name, geometry, model, initial_avg_price, init_unemployment_index):
        self.name = name
        self.geometry = geometry
        # Make it here to avoid doing it everytime
        self.minx, self.miny, self.maxx, self.maxy = self.geometry.total_bounds
        self.model = model
        self.data = defaultdict(list)
        self.init_avg_price = initial_avg_price
        self.init_unemployment_index = init_unemployment_index
        self.data['p'] = [1]

    def update_income_score(self, value):
        self.data['income_score'].append(value)

    def update_service_score(self, value):
        self.data['service_score'].append(value)

    # update income scores for income and p_skilled
    def update_avg_income(self, households):
        if households:
            avg_income = self.model.np.nanmean([i.current_income for i in households])
        self.data['avg_income'].append(avg_income if households else 0)

    def update_p_skill(self, households):
        if households:
            p_skilled = sum([i.skilled for i in households]) / len(households)
        self.data['p_skilled'].append(p_skilled if households else 0)

    def update_population(self, households):
        population = sum([len(household.members) for household in households])
        self.data['population'].append(population)

    def update_n_households(self, households):
        self.data['n_households'].append(len(households))

    def update_unemployment(self, households):
        try:
            self.data['unemployment rate'].append(len([m for h in households for m in h.members.values()
                                                if m.industry.name == 'unemployed']) / self.data['population'][-1])
        except ZeroDivisionError:
            print(f'No population at {self.name} at step {self.model.current_step}')
            self.data['unemployment rate'].append(0)

    def update_service(self):
        business_here = [business for business in self.model.businesses if business.neighbourhood == self]
        service = sum([len(b.workers) for b in business_here if b.industry.name == "service"])
        self.data['n_service'].append(service)

    def update_all(self):
        households = [hh for hh in self.model.households if hh.neighbourhood == self]
        self.update_population(households)
        self.update_service()
        self.update_p_skill(households)
        self.update_avg_income(households)
        self.update_unemployment(households)
        self.update_n_households(households)
